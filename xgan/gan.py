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

import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from torchvision.utils import save_image
from torchsummary import summary

from xgan.xai import GradCam
from xgan.utils import check_for_key

real_label = 1.
fake_label = 0.

def create_batch_generator(data_loader):
    for index, (data, target) in enumerate(data_loader):
        yield index, (data, target)

def create_tensor_batch_generator(data_tensor, indices, data_labels=None):
    for batch_idx, samples_indices in enumerate(indices):
        batch_data = data_tensor[samples_indices]
        if data_labels is None:
            yield batch_idx, (batch_data, 0)
        else:
            batch_labels = data_labels[samples_indices]
            yield batch_idx, (batch_data, batch_labels)

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
        self.discriminator_config = discriminator_config
        self.discriminator = self.discriminator_config['model']

        self.discriminator_optimizer = self.discriminator_config['optimizer']
        self.generator_config['model'].to(device)
        self.discriminator_config['model'].to(device)
        self.verbose = verbose
        self.explanation_config = explanation_config
        self.datetime_prefix = self.get_str_datetime()
        
        self.grad_cam = None
        self.lime = None

        self.explain_methods_prepared = False

    def _prepare_explain_methods(self):
        explanation_config = self.explanation_config
        if 'grad_cam' in explanation_config.keys():
            if explanation_config['grad_cam'] == True:
                self.grad_cam = GradCam(explanation_config, self)
            else:
                value = explanation_config['grad_cam']
                raise XAIConfigException(f'Unknown parameter \'grad_cam\' in explanation_config : {value}')
        if 'lime' in explanation_config.keys():
            self.lime = explanation_config['lime']['model']
        self.explain_methods_prepared = True

    def _calculate_iters_count(self, train_config):
        discr_iters = train_config['discr_per_gener_iters']
        gen_iters = 1
        if type(discr_iters) == float:
            multiplier = int(1 / discr_iters)
            gen_iters *= multiplier
            discr_iters = 1
        return discr_iters, gen_iters

    def save_gan(self, epoch='final'): # TODO
        result_dir = self.generation_config['result_dir']
        models_dir = os.path.join(result_dir, 'model')
        # if not os.path.exists(models_dir):
        #     os.makedirs(models_dir)
        torch.save(self.generator, os.path.join(models_dir, f'generator_{epoch}_epoch.pt'))
        torch.save(self.discriminator, os.path.join(models_dir, f'discriminator_{epoch}_epoch.pt'))

    def _save_images(self, images, images_dir, content_type_name):
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        if isinstance(torch.Tensor, type(images)):
            field_size = len(str(images.shape[0]))
        else:
            field_size = len(str(len(images)))
        for idx, image in enumerate(images):
            formatted_idx_value = '0' * (field_size - len(str(idx))) + str(idx)
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
            summary(generator_config['model'], generator_config['z_shape'])
            print('### Discriminator summary')
            summary(discriminator_config['model'], discriminator_config['input_shape'])

    def _save_examples(self, samples_number, data, postfix, result_dir):
        if isinstance(torch.Tensor, type(data)):
            sample_indices = RandomSampler(range(samples_number))
            sampled_data = train_data[sample_indices]
            batch_generator = create_tensor_batch_generator(data, sample_indices)
        else:
            train_dataloader = torch.utils.data.DataLoader(data, batch_size=samples_number, shuffle=True)
            batch_generator = create_batch_generator(train_dataloader)
        counter = 0
        result_dir = os.path.join(result_dir, postfix)
        while counter < samples_number:
            indices, (images_batch, _) = next(batch_generator)
            self._save_images(images_batch, result_dir, postfix)
            counter += images_batch.shape[0]

    def train(self, train_config, generation_config=None):
        if not self.explain_methods_prepared:
            self._prepare_explain_methods()
        
        if check_for_key(generation_config, 'result_dir'):
            result_dir = generation_config['result_dir']
        else:
            result_dir = 'result_'+self.datetime_prefix

        batch_size = train_config['batch_size']
        train_data = train_config['train_dataset']
        test_data = train_config['test_dataset']
        
        self._print_summary(self.generator_config, self.discriminator_config)
        torch.set_printoptions(profile="full")

        if isinstance(torch.Tensor, type(train_data)):
            train_indices = BatchSampler(RandomSampler(range(train_data.shape[0])), batch_size=batch_size, drop_last=False)
            test_indices = BatchSampler(RandomSampler(range(test_data.shape[0])), batch_size=batch_size, drop_last=False)
        else:
            train_dataloader = torch.utils.data.DataLoader(train_config['train_dataset'], batch_size=batch_size, shuffle=True)
        # test_dataloader = torch.utils.data.DataLoader(train_config['testset'], batch_size=batch_size, shuffle=True)
        criterion = self.discriminator_config['criterion']
        
        discr_iters, gen_iters = self._calculate_iters_count(train_config)

        # Save examples of the data is needed
        if check_for_key(generation_config, 'save_examples', True):
            samples_number = generation_config['samples_number']
            self._save_examples(samples_number, train_data, 'train_data_example', result_dir)
            self._save_examples(samples_number, test_data, 'test_data_example', result_dir)
            gc.collect()
            torch.cuda.empty_cache()
        
        max_epochs = train_config['epochs']
        max_epochs_field_size = len(str(max_epochs))
        progress_bar_epoch = tqdm(range(max_epochs))
        for epoch in range(max_epochs):
            formatted_epoch_value = '0' * (max_epochs_field_size - len(str(epoch))) + str(epoch)
            progress_bar_epoch.set_description(f'# Training. Processing {epoch+1} epoch')
            if generation_config is not None and epoch != 0 and epoch % int(train_config['iterations_between_saves']) == 0:
                self.generate(f'epoch_{formatted_epoch_value}', generation_config, train_config=train_config)
            if isinstance(torch.Tensor, type(train_data)):
                train_labels = train_config['train_labels']
                batch_generator = create_tensor_batch_generator(train_data, train_indices, train_labels)
            else:
                batch_generator = create_batch_generator(train_dataloader)
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
                    batch_size = device_data.size(0)

                    # Prepare fake data
                    noize_np = np.random.normal(0, 1, size=self.generator.get_input_shape(batch_size))
                    noize = torch.from_numpy(noize_np).float().to(self.device) 
                    generator_out = self.generator(noize).detach()

                    # Concat data
                    data = torch.cat((device_data, generator_out))

                    # Prepare labels                
                    label_real = torch.full((batch_size,), real_label, dtype=self.dtype, device=self.device)
                    label_fake = torch.full((batch_size,), fake_label, dtype=self.dtype, device=self.device)
                    label = torch.cat((label_real, label_fake))

                    # Calculate predictions of discriminator and update discriminator's weights by real and fake data
                    output = self.discriminator(data).view(-1)
                    err_discriminant_real = criterion(output, label)
                    err_discriminant_real.backward()
                    
                    self.discriminator_optimizer.step()
                    
                    del device_data, noize_np, noize, generator_out, data, label_real, label_fake, label, output, err_discriminant_real
                    gc.collect()
                    torch.cuda.empty_cache()

                for gen_iter in range(gen_iters):
                    # Calculate generations and update generator's weights
                    self.generator.zero_grad()
                    label = torch.full((batch_size,), real_label, dtype=self.dtype, device=self.device) # fake labels are real for generator cost
                    noize_np = np.random.normal(0, 1, size=self.generator.get_input_shape(batch_size))
                    noize = torch.from_numpy(noize_np).float().to(self.device)
                    generator_out = self.generator(noize)
                    output = self.discriminator(generator_out).view(-1)
                    err_discriminator = criterion(output, label)
                    err_discriminator.backward()
                    self.generator_optimizer.step()
                    del label, noize_np, noize, generator_out, output, err_discriminator
                    gc.collect()
                    torch.cuda.empty_cache()
            progress_bar_epoch.update(1)

    def generate(self, postfix, generation_config, train_config=None):
        if not self.explain_methods_prepared:
            self._prepare_explain_methods()
            
        if check_for_key(generation_config, 'result_dir'):
            result_dir = generation_config['result_dir']
        else:
            result_dir = 'result_'+self.datetime_prefix
        result_dir = os.path.join(result_dir, postfix)
        os.makedirs(result_dir)

        batch_size = generation_config['batch_size']
        generated_set = []
        
        if self.grad_cam:
            grad_cam_real = []
            grad_cam_fake = []
            grad_cam_prob_real = {}
            grad_cam_prob_fake = {}

        samples_number = generation_config['samples_number']
        samples_indices = BatchSampler(SequentialSampler(range(samples_number)), batch_size=batch_size, drop_last=False)

        max_field_size = len(str(samples_number))
        for sample_idx, batch_indices in enumerate(samples_indices): 
            noize_np = np.random.normal(0, 1, size=self.generator.get_input_shape(batch_size))
            noize = torch.from_numpy(noize_np).float().to(self.device)
            out = self.generator(noize)
            for subout in out:
                generated_set.append(subout)
            if self.grad_cam is not None:
                for idx, subout in enumerate(out):
                    grad_cam_real_local, grad_cam_fake_local, real_prob, fake_prob = self.grad_cam.apply(subout.unsqueeze(dim=0), generation_config)
                    grad_cam_real.append(grad_cam_real_local)
                    grad_cam_fake.append(grad_cam_fake_local)
                    grad_cam_prob_real[idx + sample_idx * batch_size] = real_prob
                    grad_cam_prob_fake[idx + sample_idx * batch_size] = fake_prob
            if self.lime is not None:
                with torch.no_grad():
                    for idx, subout in enumerate(out):
                        formatted_idx = '0' * (max_field_size - len(str(idx))) + str(idx)
                        train_data = train_config['train_dataset']
                        test_data = train_config['test_dataset']
                        if isinstance(torch.Tensor, type(train_data)):
                            train_labels = train_config['train_labels']
                            train_indices = BatchSampler(RandomSampler(range(train_data.shape[0])), batch_size=batch_size, drop_last=False)
                            test_indices = BatchSampler(RandomSampler(range(test_data.shape[0])), batch_size=batch_size, drop_last=False)
                            batch_generator = create_tensor_batch_generator(train_data, train_indices, train_labels)
                        else:
                            train_dataloader = torch.utils.data.DataLoader(train_config['train_dataset'], batch_size=batch_size, shuffle=True)
                            batch_generator = create_batch_generator(train_dataloader)
                        self.lime.explain_sample(subout, formatted_idx, batch_generator, result_dir, self.explanation_config, generation_config)

            del noize_np, noize, out, subout
            gc.collect()
            torch.cuda.empty_cache()

        self._save_images(generated_set, result_dir, 'generated')
        if self.grad_cam:
            images_dir = self._save_images(grad_cam_real, result_dir, 'grad_cam_real')
            with open(os.path.join(images_dir, 'real_probabilities.json'), 'w') as f:
                f.write(json.dumps(grad_cam_prob_real, indent=4, sort_keys=True))
            images_dir = self._save_images(grad_cam_fake, result_dir, 'grad_cam_fake')
            with open(os.path.join(images_dir, 'fake_probabilities.json'), 'w') as f:
                f.write(json.dumps(grad_cam_prob_fake, indent=4, sort_keys=True))
