import torch
import cv2
import numpy as np

class GradCam:
    def __init__(self, explanation_config, gan):
        self.gan = gan
        self.explanation_config = explanation_config
        if 'grad_cam' in explanation_config.keys() and explanation_config['grad_cam'] == True:
            self._prepare_forward_hooks(gan.discriminator, 'discriminator')
            self.grad_cam = True
            if 'grad_cam_layer_name' in explanation_config.keys():
                self.grad_cam_target_module = gan.discriminator.named_modules()[explanation_config['grad_cam_layer_name']]
                self.grad_cam_target_module_name = explanation_config['grad_cam_layer_name']
            else:
                if 'grad_cam_layer_number' in explanation_config.keys():
                    self.grad_cam_layer_number = explanation_config['grad_cam_layer_number']
                else:
                    self.grad_cam_layer_number = -1
                self.grad_cam_target_module, self.grad_cam_target_module_name = self._find_target_conv_layer(self.grad_cam_layer_number)

    def apply(self, x, generator_config):
        discr_input = x.detach().clone()
        criterion = self.gan.discriminator_config['criterion']
        discr_out = self.gan.discriminator(discr_input).view(-1)
        batch_size = discr_input.shape[0]

        for label_value, label_postfix in [(1., 'real'), [0., 'fake']]:
            self.gan.discriminator.zero_grad()
            label = torch.full((batch_size,), label_value, dtype=self.gan.dtype, device=self.gan.device)
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
            if abs(max_val) > 1e-6:
                grad_cam_heatmap /= max_val
            
            net_value = discr_out[0].item()
            if label_postfix == 'real':
                real_prob = net_value
                applied_heatmap = self._apply_heatmap(discr_input[0], grad_cam_heatmap, real_prob)
                grad_cam_real = applied_heatmap
            else:
                fake_prob = 1. - net_value
                applied_heatmap = self._apply_heatmap(discr_input[0], grad_cam_heatmap, fake_prob)
                grad_cam_fake = applied_heatmap
        return grad_cam_real, grad_cam_fake, real_prob, fake_prob

    def _apply_heatmap(self, generated, heatmap, net_value):
        heatmap = cv2.resize(heatmap.cpu().numpy(), (generated.shape[1], generated.shape[2]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(float)
        heatmap /= 255.

        new_dim = generated.shape[0]
        heatmap = torch.from_numpy(heatmap).to(self.gan.device)
        heatmap = torch.permute(heatmap, (2, 0, 1))

        if generated.shape[0] == 1:
            generated = generated.expand(3, generated.shape[1], generated.shape[2])

        superimposed_img = heatmap * 0.5 * net_value + generated
        mx = torch.max(superimposed_img)
        if abs(mx) > 1e-6:
            superimposed_img /= mx
        return superimposed_img

    def _find_target_conv_layer(self, grad_cam_layer_number):
        conv_counter = 0
        for name, module in self.gan.discriminator.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                grad_cam_target_module = module
                grad_cam_target_module_name = name
                if conv_counter == grad_cam_layer_number:
                    return grad_cam_target_module, grad_cam_target_module_name
                conv_counter += 1
        if conv_counter == 0:
            raise GanException('No conv layers in network')
        if grad_cam_layer_number != -1 and grad_cam_layer_number != conv_counter:
            raise GanException(f'Unexpected error in _find_target_conv_layer(). grad_cam_layer_number:{grad_cam_layer_number}, conv_counter:{conv_counter}')
        return grad_cam_target_module, grad_cam_target_module_name

    def _hook_fabric(self, storage, key):
        def _hook_fn(self, input, output):
            if not isinstance(output, torch.Tensor):
                GanException(f"Error with output tensor. Type: {type(output)}")
            output = output.detach()
            storage[key] = output
        return _hook_fn

    def _prepare_forward_hooks(self, model, model_name):
        if not hasattr(self, 'hooks'):
            self.hooks = {}
        self.hooks[model_name] = {}
        for name, module in model.named_modules():
            module.register_forward_hook(self._hook_fabric(self.hooks[model_name], name))
