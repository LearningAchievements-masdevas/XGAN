import numpy as np

import os
import shutil
import gc
import hashlib
import time
import logging
from datetime import datetime
from tqdm.auto import tqdm
import json
import time
import inspect

import torch
from torch.utils.data import BatchSampler, SequentialSampler
from torch.profiler import profile, record_function, ProfilerActivity

from torchvision.utils import save_image
from torchsummary import summary

from xgan.xai import GradCam, LIME, LimeRandomForest, GenSpace, ShapGen
from xgan.utils import check_for_key, prepare_batches, gan_labels, set_flush_data_limit, flush_data

class GAN:
    def get_str_datetime(self):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y")
        return dt_string

    def __init__(self, device, generator_config, discriminator_config, *, verbose=False, explanation_config=None, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        self.generator_config = generator_config
        self.generator = self.generator_config['model']
        self.generator_optimizer = self.generator_config['optimizer']
        if 'noize_generator' in self.generator_config.keys():
            self.noize_generator = self.generator_config['noize_generator']
        else:
            self.noize_generator = lambda generator, shape: np.random.normal(0, 1, size=shape)
        if 'scheduler' in self.generator_config.keys():
            self.generator_scheduler = self.generator_config['scheduler']
        else:
            self.generator_scheduler = None


        self.discriminator_config = discriminator_config
        self.discriminator = self.discriminator_config['model']
        self.discriminator_optimizer = self.discriminator_config['optimizer']
        if 'scheduler' in self.discriminator_config.keys():
            self.discriminator_scheduler = self.discriminator_config['scheduler']
        else:
            self.discriminator_scheduler = None

        self.generator_config['model'].to(device)
        self.discriminator_config['model'].to(device)
        self.verbose = verbose
        self.explanation_config = explanation_config
        self.datetime_prefix = self.get_str_datetime()

        self.explain_methods_prepared = False

    def _prepare_explain_methods(self):
        explanation_config = self.explanation_config
        if 'grad_cam' in explanation_config.keys():
            if explanation_config['grad_cam'] == True:
                self.grad_cam_run = True
            else:
                value = explanation_config['grad_cam']
                raise XAIConfigException(f'Unknown parameter \'grad_cam\' in explanation_config : {value}')
        else:
            self.grad_cam_run = False
        if 'lime' in explanation_config.keys():
            self.lime = explanation_config['lime']['model']
        if 'genspace' in explanation_config.keys():
            self.genspace_run = True
        else:
            self.genspace_run = False
        if 'shap_gen' in explanation_config.keys():
            self.shap_gen_run = True
        else:
            self.shap_gen_run = False
        self.explain_methods_prepared = True

    def _calculate_iters_count(self, train_config):
        discr_iters = train_config['discr_per_gener_iters']
        gen_iters = 1
        if type(discr_iters) == float:
            multiplier = int(1 / discr_iters)
            gen_iters *= multiplier
            discr_iters = 1
        return discr_iters, gen_iters

    def save_gan(self, result_dir, postfix):
        models_dir = os.path.join(result_dir, 'models')
        os.makedirs(models_dir)
        torch.save(self.generator.state_dict(), os.path.join(models_dir, f'generator_{postfix}.pt'))
        torch.save(self.discriminator.state_dict(), os.path.join(models_dir, f'discriminator_{postfix}.pt'))

    def _save_images(self, images, images_dir, content_type_name, samples_number, start_index=0):
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        for idx, image in enumerate(images):
            formatted_idx_value = '0' * (len(str(samples_number)) - len(str(start_index + idx))) + str(start_index + idx)
            save_image(image, os.path.join(images_dir, f'{formatted_idx_value}_{content_type_name}.png'))
        return images_dir

    def clear_result_dir(self, generation_config):
        result_dir = generation_config['result_dir']
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)

    def _print_summary(self, generator_config, discriminator_config):
        if self.verbose:
            print('### Generator summary')
            summary(self.generator, self.generator_config['input_shape'])
            print('### Discriminator summary')
            summary(self.discriminator, self.discriminator_config['input_shape'])

    def _save_examples(self, samples_number, data, labels, postfix, result_dir, batch_size):
        batch_generator = prepare_batches(data, labels, batch_size)
        result_dir = os.path.join(result_dir, postfix)
        remained = samples_number
        for batch_idx, (batch_data, _) in batch_generator:
            if remained == 0:
                break
            reduced_batch_data = batch_data[:remained]
            self._save_images(reduced_batch_data, result_dir, postfix, samples_number, samples_number - remained)
            remained -= reduced_batch_data.shape[0]

    def train(self, train_config, generation_config=None):
        funcname = inspect.currentframe().f_code.co_name
        if 'gc_run_prob' in train_config.keys():
            set_flush_data_limit(inspect.currentframe().f_code.co_name, train_config['gc_run_prob'])
        else:
            set_flush_data_limit(inspect.currentframe().f_code.co_name, 1.)
        if not self.explain_methods_prepared:
            self._prepare_explain_methods()
        
        if check_for_key(generation_config, 'result_dir'):
            result_dir = generation_config['result_dir']
        else:
            result_dir = 'result_'+self.datetime_prefix

        train_data = train_config['train_dataset']
        train_labels = train_config['train_labels'] if 'train_labels' in train_config.keys() and isinstance(train_data, torch.Tensor) else None
        test_data = train_config['test_dataset']
        test_labels = train_config['test_labels'] if 'test_labels' in train_config.keys() and isinstance(test_data, torch.Tensor) else None
        
        self._print_summary(self.generator_config, self.discriminator_config)
        torch.set_printoptions(profile="full")
        criterion = self.discriminator_config['criterion']

        discr_iters, gen_iters = self._calculate_iters_count(train_config)

        # Save examples of the data is needed
        if check_for_key(generation_config, 'save_examples', True):
            samples_number = generation_config['samples_number']
            batch_size = generation_config['batch_size']
            self._save_examples(samples_number, train_data, train_labels, 'train_data_example', result_dir, batch_size)
            self._save_examples(samples_number, test_data, test_labels, 'test_data_example', result_dir, batch_size)
            flush_data(funcname)
        
        batch_size = train_config['batch_size']
        max_epochs = train_config['epochs']
        max_epochs_field_size = len(str(max_epochs))
        progress_bar_epoch = tqdm(range(max_epochs))
        
        for epoch in range(max_epochs):
            formatted_epoch_value = '0' * (max_epochs_field_size - len(str(epoch))) + str(epoch)
            if generation_config is not None and epoch % int(train_config['iterations_between_saves']) == 0:
                progress_bar_epoch.set_description(f'# Generating. Processing #{epoch} epoch')
                self.generate(f'epoch_{formatted_epoch_value}', generation_config, train_config=train_config)
            progress_bar_epoch.set_description(f'# Training. Processing #{epoch} epoch')
            batch_generator = prepare_batches(train_data, train_labels, batch_size)
            continue_flag = True
            while continue_flag:
                for discr_iter in range(discr_iters):
                    # Prepare real data
                    self.discriminator.zero_grad()
                    try:
                        indices, images = next(batch_generator)
                    except StopIteration:
                        continue_flag = False
                        break
                    device_data = images[0].to(self.device)
                    data_batch_size = device_data.size(0)

                    # Prepare fake data
                    generator_out = self._internal_generate(data_batch_size).detach()

                    # Concat data
                    data = torch.cat((device_data, generator_out))

                    # Prepare labels                
                    label_real = torch.full((data_batch_size,), gan_labels['real'], dtype=self.dtype, device=self.device)
                    label_fake = torch.full((data_batch_size,), gan_labels['fake'], dtype=self.dtype, device=self.device)
                    label = torch.cat((label_real, label_fake))

                    # Calculate predictions of discriminator and update discriminator's weights by real and fake data
                    output = self.discriminator(data).view(-1)
                    err_discriminant_real = criterion(output, label)
                    err_discriminant_real.backward()
                    if 'grad_norm' in self.discriminator_config.keys():
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.discriminator_config['grad_norm'])
                    
                    self.discriminator_optimizer.step()
                    del indices, images, data_batch_size, device_data, generator_out, data, label_real, label_fake, label, output, err_discriminant_real
                    flush_data(funcname)

                for gen_iter in range(gen_iters):
                    # Calculate generations and update generator's weights
                    self.generator.zero_grad()
                    label = torch.full((batch_size,), gan_labels['real'], dtype=self.dtype, device=self.device) # fake labels are real for generator cost
                    generator_out = self._internal_generate(batch_size)
                    output = self.discriminator(generator_out).view(-1)
                    err_discriminator = criterion(output, label)
                    err_discriminator.backward()
                    if 'grad_norm' in self.generator_config.keys():
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.generator_config['grad_norm'])
                    self.generator_optimizer.step()
                    del label, generator_out, output, err_discriminator
                    flush_data(funcname)
            if self.generator_scheduler is not None:
                self.generator_scheduler.step()
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()
            progress_bar_epoch.update(1)

    def _get_noize(self, batch_size):
        noize_shape = (batch_size, ) + self.generator_config['input_shape']
        noize_np = self.noize_generator(self.generator, noize_shape)
        return noize_np

    def _internal_generate(self, batch_size):
        noize_np = self._get_noize(batch_size)
        noize = torch.from_numpy(noize_np).to(dtype=self.dtype, device=self.device)
        return self.generator(noize)

    def generate(self, postfix, generation_config, train_config=None):
        funcname = inspect.currentframe().f_code.co_name
        if 'gc_run_prob' in generation_config.keys():
            set_flush_data_limit(funcname, generation_config['gc_run_prob'])
        else:
            set_flush_data_limit(funcname, 1.)
        if not self.explain_methods_prepared:
            self._prepare_explain_methods()
            
        if check_for_key(generation_config, 'result_dir'):
            result_dir = generation_config['result_dir']
        else:
            result_dir = 'result_'+self.datetime_prefix
        result_dir = os.path.join(result_dir, postfix)
        os.makedirs(result_dir)

        if check_for_key(generation_config, 'save_models', True):
            self.save_gan(result_dir, postfix)

        batch_size = generation_config['batch_size']
        generated_set = []
        
        if self.grad_cam_run:
            grad_cam_real = []
            grad_cam_fake = []
            grad_cam_prob_real = {}
            grad_cam_prob_fake = {}

        samples_number = generation_config['samples_number']
        samples_indices = BatchSampler(SequentialSampler(range(samples_number)), batch_size=batch_size, drop_last=False)
        max_field_size = len(str(samples_number))
        generation_explanation = {}
        check_nodes_count = isinstance(self.lime, LimeRandomForest) and 'nodes_count' in self.explanation_config['lime']['features']
        if check_nodes_count:
            lime_nodes_counts = []
        for sample_idx, batch_indices in enumerate(samples_indices):
            generated_data = self._internal_generate(len(batch_indices))
            for subout in generated_data:
                generated_set.append(subout.detach())
            if self.grad_cam_run:
                grad_cam = GradCam(self.explanation_config, self)
                for idx, subout in enumerate(generated_data):
                    grad_cam_real_local, grad_cam_fake_local, real_prob, fake_prob = grad_cam.apply(subout.unsqueeze(dim=0), generation_config)
                    grad_cam_real.append(grad_cam_real_local.detach())
                    grad_cam_fake.append(grad_cam_fake_local.detach())
                    grad_cam_prob_real[idx + sample_idx * batch_size] = real_prob
                    grad_cam_prob_fake[idx + sample_idx * batch_size] = fake_prob
                del grad_cam
                flush_data(funcname)
            if self.lime is not None:
                with torch.no_grad():
                    train_data = train_config['train_dataset']
                    train_labels = train_config['train_labels'] if 'train_labels' in train_config.keys() and isinstance(train_data, torch.Tensor) else None
                    
                    lime_helper = LIME(self, self.explanation_config['lime'])
                    X, y, weights_batch, features = lime_helper.generate_data_for_ml(train_data, train_labels, batch_size, generated_data)
                    
                    for idx, subout in enumerate(generated_data):
                        formatted_idx = '0' * (max_field_size - len(str(idx + sample_idx * batch_size))) + str(idx + sample_idx * batch_size)
                        weights = weights_batch[:, idx]
                        explanation = self.lime.explain_sample(subout, X, y, weights, features, result_dir, formatted_idx, self.explanation_config, generation_config)
                        if check_nodes_count:
                            lime_nodes_counts.append(explanation['nodes_count'])
                    del X, y, weights_batch, weights, features, lime_helper
                flush_data(funcname)
            del generated_data, subout
        if self.genspace_run:
            with torch.no_grad():
                genspace = GenSpace(self, self.explanation_config['genspace'])
                train_data = train_config['train_dataset']
                train_labels = train_config['train_labels'] if 'train_labels' in train_config.keys() and isinstance(train_data, torch.Tensor) else None
                X, y = genspace.generate_data_for_ml(train_data, train_labels, batch_size)
                genspace_counts = genspace.explain_space(X, y, batch_size, result_dir)
                del X, y, genspace
            flush_data(funcname)
        if self.shap_gen_run:
            with torch.no_grad():
                shap_gen = ShapGen(self, self.generator, self.explanation_config['shap_gen']['model'])
                train_data = train_config['train_dataset']
                train_labels = train_config['train_labels'] if 'train_labels' in train_config.keys() and isinstance(train_data, torch.Tensor) else None
                if 'columns' not in self.explanation_config['shap_gen']:
                    num_columns = np.prod(np.array(self.generator_config['input_shape']))
                    columns = ['feat_'+str(column_idx) for column_idx in range(num_columns)]
                else:
                    columns = self.explanation_config['shap_gen']['columns']
                X, y = shap_gen.generate_data_for_ml(train_data, train_labels, batch_size)
                shap_gen.fit_ml(X, y)
                shap_gen.explain(self.explanation_config['shap_gen'], self._get_noize, columns, result_dir)
                del X, y, columns, shap_gen
            flush_data(funcname)

        if check_nodes_count:
            generation_explanation['lime_mean_nodes_count'] = sum(lime_nodes_counts) // len(lime_nodes_counts)

        if self.genspace_run:
            generation_explanation['genspace'] = {}
            generation_explanation['genspace']['counts'] = genspace_counts

        if len(generation_explanation.keys()) > 0:
            with open(os.path.join(result_dir, 'generation_explanation.json'), 'w') as f:
                f.write(json.dumps(generation_explanation, indent=4, sort_keys=True))

        self._save_images(generated_set, result_dir, 'generated', len(generated_set))
        if self.grad_cam_run:
            images_dir = self._save_images(grad_cam_real, result_dir, 'grad_cam_real', len(grad_cam_real))
            with open(os.path.join(images_dir, 'real_probabilities.json'), 'w') as f:
                f.write(json.dumps(grad_cam_prob_real, indent=4, sort_keys=True))
            images_dir = self._save_images(grad_cam_fake, result_dir, 'grad_cam_fake', len(grad_cam_fake))
            with open(os.path.join(images_dir, 'fake_probabilities.json'), 'w') as f:
                f.write(json.dumps(grad_cam_prob_fake, indent=4, sort_keys=True))
            del grad_cam_real, grad_cam_fake
