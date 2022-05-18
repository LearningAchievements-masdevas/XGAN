import gc
import torch
import numpy as np
import pandas as pd
import os
from collections import Counter

import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import BatchSampler, SequentialSampler
from sklearn.manifold import TSNE

from xgan.utils import prepare_batches

sns.set_style('whitegrid')

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except:
    pass

class GenSpace:
    def __init__(self, gan, genspace_config):
        self.gan = gan
        self.genspace_config = genspace_config
        self.model = self.genspace_config['model']

    def _generate_features(self, sample_shape):
        features = []
        for index in np.ndindex(sample_shape):
            features.append('x'.join([str(value) for value in index]))
        return features

    def generate_data_for_ml(self, data, labels, batch_size):
        X_list = []
        y_list = []
        real_batch_generator = prepare_batches(data, labels, batch_size)
        for batch_idx, (batch_data, batch_labels) in real_batch_generator:
            batch_data = batch_data.to(self.gan.device)
            self.features = self._generate_features(batch_data[0].shape)
            batch_data = batch_data.reshape(batch_data.shape[0], -1)
            X_list.append(batch_data.cpu().numpy())
            y_list.append(batch_labels.cpu().numpy())
        
        X = pd.DataFrame(data=np.concatenate(X_list), columns=self.features)
        y = np.concatenate(y_list)
        
        del batch_data, batch_labels, X_list, y_list
        return X, y

    def explain_space(self, X, y, batch_size, path):
        self.model.fit(X, y)
        indices = BatchSampler(SequentialSampler(range(self.genspace_config['samples_to_generate'])), batch_size=batch_size, drop_last=False)
        X_list = []
        for batch_idx, batch_indices in enumerate(indices):
            generated_data = self.gan._internal_generate(len(batch_indices))
            generated_data = generated_data.reshape(generated_data.shape[0], -1)
            X_list.append(generated_data.cpu().numpy())
        X_test = pd.DataFrame(data=np.concatenate(X_list), columns=self.features)
        y_predicted = self.model.predict(X_test).astype(np.long)
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_test)
        data = pd.DataFrame.from_dict({'x' : X_embedded[:, 0], 'y' : X_embedded[:, 1], 'labels' : y_predicted})
        label_counts = dict(Counter(y_predicted))
        label_counts = {str(k):v for k, v in label_counts.items()}
        hue_order = sorted([int(k) for k,v in label_counts.items()])
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.scatterplot(data=data, x='x', y='y', hue='labels', hue_order=hue_order, ax=ax)
        plt.legend()
        plt.savefig(os.path.join(path, f'genspace.png'))
        fig.clear()
        plt.close(fig)
        del indices, X_list, X_test, y_predicted, X_embedded, data, hue_order
        return label_counts
