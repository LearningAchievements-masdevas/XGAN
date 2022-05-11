import numpy as np
import pandas as pd
import torch

from xgan.utils import prepare_batches, gan_labels

class LimeModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = None

    def get_model(self):
        return self.model

class LIME:
    def __init__(self, gan, explanation_config_lime):
        self.gan = gan
        self.explanation_config_lime = explanation_config_lime

    def _generate_features(self, sample_shape):
        features = []
        for index in np.ndindex(sample_shape):
            features.append('x'.join([str(value) for value in index]))
        return features

    def generate_data_for_ml(self, data, labels, batch_size, generated_data):
        features = self._generate_features(generated_data[0].shape)
        generated_data = generated_data.reshape(generated_data.shape[0], -1)
        generated_count = 0
        samples_per_class = self.explanation_config_lime['samples_per_class']
        X_list = []
        distances_list = []

        while generated_count < samples_per_class:
            samples_to_gen = batch_size if generated_count + batch_size <= samples_per_class else samples_per_class - generated_count
            generated_samples = self.gan._internal_generate(samples_to_gen)
            generated_samples = generated_samples.reshape(generated_samples.shape[0], -1)
            X_list.append(generated_samples.cpu().numpy())
            local_distances = torch.cdist(generated_samples, generated_data, p=2).cpu().numpy()
            distances_list.append(local_distances)
            generated_count += samples_to_gen

        real_batch_generator = prepare_batches(data, labels, batch_size)
        remained = self.explanation_config_lime['samples_per_class']
        for batch_idx, (batch_data, _) in real_batch_generator:
            if remained == 0:
                break
            reduced_batch_data = batch_data[:remained].to(self.gan.device)
            reduced_batch_data = reduced_batch_data.reshape(reduced_batch_data.shape[0], -1)
            local_distances = torch.cdist(reduced_batch_data, generated_data, p=2).cpu().numpy()
            X_list.append(reduced_batch_data.cpu().numpy())
            distances_list.append(local_distances)
            remained -= batch_size
        
        X = pd.DataFrame(data=np.concatenate(X_list), columns=features)
        distances = np.concatenate(distances_list)
        distances = np.where(distances < 1e-6, distances + 1e-6, distances)
        weights = 1 / distances
        max_weight = weights.max(axis=0)
        weights /= max_weight
        label_real = torch.full((samples_per_class,), gan_labels['real'], dtype=self.gan.dtype, device=self.gan.device)
        label_fake = torch.full((samples_per_class,), gan_labels['fake'], dtype=self.gan.dtype, device=self.gan.device)
        labels = torch.cat([label_fake, label_real], dim=0).cpu().numpy().flatten()
        return X, labels, weights, features

