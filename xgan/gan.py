import numpy as np
import torch
import os
import shutil
import gc
import hashlib
import time
from torchvision.utils import save_image
from torchsummary import summary
import logging
from datetime import datetime
import cv2
import json

real_label = 1.
fake_label = 0.

class GanException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

def create_batch_generator(data_loader):
    for index, (data, target) in enumerate(data_loader):
        yield index, (data, target)

def discriminator_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class GAN:
    def get_str_datetime(self):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
        return dt_string

    def __init__(self, device, generator_config, discriminator_config, *, verbose=False, generation_config=None, explanation_config=None):
        self.device = device
        self.generator_config = generator_config
        self.generator = self.generator_config['model']

        self.generator_optimizer = self.generator_config['optimizer']
        self.discriminator_config = discriminator_config
        self.discriminator = self.discriminator_config['model']
        #print(dir(self.discriminator.named_parameters()))
        # for name, param in self.discriminator.named_parameters():
        #     print(name, param.size())
        #     print(type(param))


        # os.exit(0)
        self.discriminator_optimizer = self.discriminator_config['optimizer']
        self.generator_config['model'].to(device)
        self.discriminator_config['model'].to(device)
        self.verbose=verbose
        self.generation_config = generation_config
        self.explanation_config = explanation_config
        self.datetime_prefix = self.get_str_datetime()
        self.grad_cam = False
        self.explanation_prepared = False

    def __calculate_iters_count(self, train_config):
        discr_iters = train_config['discr_per_gener_iters']
        gen_iters = 1
        if type(discr_iters) == float:
            multiplier = int(1 / discr_iters)
            gen_iters *= multiplier
            discr_iters = 1
        return discr_iters, gen_iters

    def __save_models(self, epoch):
        result_dir = self.generation_config['result_dir']
        models_dir = os.path.join(result_dir, self.datetime_prefix, 'model')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        torch.save(self.generator, os.path.join(models_dir, f'generator_{epoch}_epoch.pt'))
        torch.save(self.discriminator, os.path.join(models_dir, f'discriminator_{epoch}_epoch.pt'))

    def __save_images(self, images, content_type_name):
        result_dir = self.generation_config['result_dir']
        images_dir = os.path.join(result_dir, self.datetime_prefix, content_type_name)
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        for idx, image in enumerate(images):
            save_image(image, os.path.join(images_dir, f'{idx}.png'))
        return images_dir
                
    def clear_result_dir(self):
        result_dir = self.generation_config['result_dir']
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)

    def train(self, train_config):
        if not self.explanation_prepared:
            self.__prepare_explanation()
            self.explanation_prepared = True
        base_batch_size = train_config['batch_size']
        self.discriminator.apply(discriminator_init)
        if self.verbose:
            print('### Generator summary')
            print((1,) + self.generator_config['z_shape'])
            summary(self.generator, self.generator_config['z_shape'])
            print('### Discriminator summary')
            summary(self.discriminator, self.discriminator_config['input_shape'])
        torch.set_printoptions(profile="full")
        train_dataloader = torch.utils.data.DataLoader(train_config['trainset'], batch_size=base_batch_size, shuffle=True, num_workers=train_config['workers'])
        test_dataloader = torch.utils.data.DataLoader(train_config['testset'], batch_size=base_batch_size, shuffle=True, num_workers=train_config['workers'])
        criterion = self.discriminator_config['criterion']
        discr_iters, gen_iters = self.__calculate_iters_count(train_config)
        were_examples_saved = False
        for epoch in range(train_config['epochs']):
            if epoch != 0 and epoch % int(train_config['iterations_between_saves']) == 0:
                self.generate(postfix=f'generated_epoch_{epoch}')
                # self.__save_models(epoch)
            print('Epoch:', epoch)
            batch_generator = create_batch_generator(train_dataloader)
            continue_flag = True
            while continue_flag:
                for discr_iter in range(discr_iters):
                    # Prepare real data
                    self.discriminator.zero_grad()
                    try:
                        indices, images = next(batch_generator)
                        #print('Get batch', indices)
                    except StopIteration:
                        continue_flag = False
                        break
                    device_data = images[0].to(self.device)
                    batch_size = device_data.size(0)

                    # Save examples of dataset
                    result_dir = self.generation_config['result_dir']
                    if self.verbose and self.generation_config is not None and not were_examples_saved:
                        self.__save_images(device_data, 'data')
                        were_examples_saved = True

                    # Prepare fake data
                    noize_np = np.random.normal(0, 1, size=self.generator.get_input_shape(batch_size))
                    noize = torch.from_numpy(noize_np).float().to(self.device) 
                    generator_out = self.generator(noize).detach()

                    # Concat data
                    data = torch.cat((device_data, generator_out))

                    # Prepare labels                
                    label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                    label_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)
                    label = torch.cat((label_real, label_fake))

                    # Calculate predictions of discriminator and update discriminator's weights by real and fake data
                    output = self.discriminator(data).view(-1)
                    err_discriminant_real = criterion(output, label)
                    err_discriminant_real.backward()
                    
                    self.discriminator_optimizer.step()
                    #print('Prefinal discr')
                    
                    del device_data, noize_np, noize, generator_out, data, label_real, label_fake, label, output, err_discriminant_real
                    gc.collect()
                    torch.cuda.empty_cache() # TODO: if device is cpu?
                    #print('Final discr')

                for gen_iter in range(gen_iters):
                    # Calculate generations and update generator's weights
                    self.generator.zero_grad()
                    label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device) # fake labels are real for generator cost
                    noize_np = np.random.normal(0, 1, size=self.generator.get_input_shape(batch_size))
                    noize = torch.from_numpy(noize_np).float().to(self.device)
                    generator_out = self.generator(noize)
                    output = self.discriminator(generator_out).view(-1)
                    err_discriminator = criterion(output, label)
                    err_discriminator.backward()
                    self.generator_optimizer.step()
                    del label, noize_np, noize, generator_out, output, err_discriminator
                    gc.collect()
                    torch.cuda.empty_cache() # TODO: if device is cpu?

    def __find_target_conv_layer(self, grad_cam_layer_number):
        conv_counter = 0
        for name, module in self.discriminator.named_modules():
            #print(module)
            #print(type(module))
            #print('CHECK')
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                #print('INSIDE IF')
                #print('@@@ SEARCHING SHAPE:', module.weight.grad.shape)
                grad_cam_target_module = module
                grad_cam_target_module_name = name
                if conv_counter == grad_cam_layer_number:
                    #print('## conv_counter', conv_counter)
                    return grad_cam_target_module, grad_cam_target_module_name
                conv_counter += 1
            #print('*' * 50)
        if conv_counter == 0:
            raise GanException('No conv layers in network')
        if grad_cam_layer_number != -1 and grad_cam_layer_number != conv_counter:
            raise GanException(f'Unexpected error in __find_target_conv_layer(). grad_cam_layer_number:{grad_cam_layer_number}, conv_counter:{conv_counter}')
        #print('## final conv_counter (+1)', conv_counter)
        return grad_cam_target_module, grad_cam_target_module_name

    def __hook_fabric(self, storage, key):
        def __hook_fn(self, input, output):
            # print('--', module)
            if not isinstance(output, torch.Tensor):
                GanException(f"Error with output tensor. Type: {type(output)}")
            # input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = output.detach()
            storage[key] = output
        return __hook_fn

    def __prepare_forward_hooks(self, model, model_name):
        if not hasattr(self, 'hooks'):
            self.hooks = {}
        self.hooks[model_name] = {}
        for name, module in model.named_modules():
            module.register_forward_hook(self.__hook_fabric(self.hooks[model_name], name))
        # print(self.hooks)
        # so

    def __get_module_by_name(self, model, name):
        return model.named_modules()[name]

    def __prepare_explanation(self):
        explanation_config = self.explanation_config
        print('@@@', type(self.discriminator.named_modules()))
        if 'grad_cam' in explanation_config.keys() and explanation_config['grad_cam'] == True:
            self.__prepare_forward_hooks(self.discriminator, 'discriminator')
            self.grad_cam = True
            if 'grad_cam_layer_name' in explanation_config.keys():
                self.grad_cam_target_module = self.__get_module_by_name(self.discriminator, explanation_config['grad_cam_layer_name'])
                self.grad_cam_target_module_name = explanation_config['grad_cam_layer_name']
            else:
                if 'grad_cam_layer_number' in explanation_config.keys():
                    self.grad_cam_layer_number = explanation_config['grad_cam_layer_number']
                else:
                    self.grad_cam_layer_number = -1
                self.grad_cam_target_module, self.grad_cam_target_module_name = self.__find_target_conv_layer(self.grad_cam_layer_number)
        self.explanation_prepared = True
    
    def __apply_heatmap(self, generated, heatmap, net_value):
        heatmap = cv2.resize(heatmap.cpu().numpy(), (generated.shape[1], generated.shape[2]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #
        #print(heatmap.shape)
        
        #print(heatmap[0][0])
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        #print(heatmap[0][0])
        heatmap = heatmap.astype(float)
        heatmap /= 255.
        #print(heatmap)
        #er

        new_dim = generated.shape[0]
        heatmap = torch.from_numpy(heatmap).to(self.device)
        # heatmap = heatmap.view(heatmap.shape[0], heatmap.shape[1], 1).expand(heatmap.shape[0], heatmap.shape[1], new_dim) #.expand(-1, -1, generated.shape[0])
        heatmap = torch.permute(heatmap, (2, 0, 1))
        # print(heatmap.shape)
        #er
        if generated.shape[0] == 1:
            generated = generated.expand(3, generated.shape[1], generated.shape[2])
        #generated = generated.cpu().numpy()
        #heatmap = heatmap.cpu().numpy()
        #print('@@1', generated.shape)
        #print('@@2', heatmap.shape)
        
        #print(heatmap)
        #er
        #heatmap = cv2.resize(heatmap, (generated.shape[0], generated.shape[1], new_dim))
        #print('^^^', type(heatmap))
        
        # heatmap_expanded_list = []
        # for channel in range(generated.shape[0]):
        #     heatmap_expanded_list.append(heatmap)
        #heatmap = np.array(heatmap_expanded_list)
        #print('@@3', heatmap.shape)

        #heatmap = np.uint8(255 * heatmap)
        superimposed_img = heatmap * 0.5 * net_value + generated
        mx = torch.max(superimposed_img)
        if abs(mx) > 1e-6:
            superimposed_img /= mx
        return superimposed_img

    def generate(self, postfix):
        if not self.explanation_prepared:
            self.__prepare_explanation()
            self.explanation_prepared = True
        batch_size = 1 # TODO if not 1 need changes in appending to generated_set            
        generated_set = []
        if self.grad_cam:
            grad_cam_real = []
            grad_cam_fake = []
            grad_cam_prob = {}

        result_dir = self.generation_config['result_dir']
        for sample_number in range(self.generation_config['samples_number']):
            noize_np = np.random.normal(0, 1, size=self.generator.get_input_shape(batch_size))
            noize = torch.from_numpy(noize_np).float().to(self.device)
            out = self.generator(noize)
            generated_set.append(out)
            label_map = {}
            if self.grad_cam:
                grad_cam_out = out.detach().clone()
                criterion = self.discriminator_config['criterion']
                discr_out = self.discriminator(grad_cam_out).view(-1)
                
                for label_value, label_postfix in [(1., 'real'), [0., 'fake']]:
                    self.discriminator.zero_grad()
                    label = torch.full((batch_size,), label_value, dtype=torch.float, device=self.device)
                    err_discriminant = criterion(discr_out, label)
                    err_discriminant.backward(retain_graph=True)
                    grad = self.grad_cam_target_module.weight.grad
                    pooled_gradients = torch.mean(grad, dim=[1, 2, 3])
                    activations_copy = self.hooks['discriminator'][self.grad_cam_target_module_name].detach().clone()
                    for idx in range(activations_copy.shape[1]):
                        activations_copy[:, idx, :, :] *= pooled_gradients[idx]
                    grad_cam_heatmap = torch.mean(activations_copy, dim=1).squeeze()
                    relu = torch.nn.ReLU()
                    grad_cam_heatmap = relu(grad_cam_heatmap)
                    max_val = torch.max(grad_cam_heatmap).item()
                    # print('@@ MAX VAL', max_val)
                    if abs(max_val) > 1e-6:
                        grad_cam_heatmap /= max_val
                    
                    net_value = discr_out[0].item()
                    grad_cam_prob[sample_number] = f'{net_value:.3f}'
                    # print('### Label:', label_value, ' SampleIdx:', sample_number)
                    # print(grad_cam_heatmap)
                    # print('=' * 40)
                    if label_postfix == 'real':
                        applied_heatmap = self.__apply_heatmap(grad_cam_out[0], grad_cam_heatmap, net_value)
                        grad_cam_real.append(applied_heatmap)
                    else:
                        applied_heatmap = self.__apply_heatmap(grad_cam_out[0], grad_cam_heatmap, 1 - net_value)
                        grad_cam_fake.append(applied_heatmap)

            del noize_np, noize, out
            gc.collect()
            torch.cuda.empty_cache() # TODO: if device is cpu?
        self.__save_images(generated_set, postfix)
        if self.grad_cam:
            images_dir = self.__save_images(grad_cam_real, postfix+'_grad_cam_real')
            with open(os.path.join(images_dir, 'real_probabilities.json'), 'w') as f:
                f.write(json.dumps(grad_cam_prob, indent=4, sort_keys=True))
            images_dir = self.__save_images(grad_cam_fake, postfix+'_grad_cam_fake')
            with open(os.path.join(images_dir, 'real_probabilities.json'), 'w') as f:
                f.write(json.dumps(grad_cam_prob, indent=4, sort_keys=True))
        # er
            
