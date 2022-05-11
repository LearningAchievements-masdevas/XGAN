import numpy as np
import pandas as pd
import torch

class LimeModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = None

    def get_model(self):
        return self.model

    def _generate_features(self, sample_shape):
        features = []
        for index in np.ndindex(sample_shape):
            features.append('x'.join([str(value) for value in index]))
        return features

    def _convert_data_to_pandas(self, explained_sample, X, y):
        # print('#INS1', explained_sample.unsqueeze(0).shape, X.shape, y.shape)
        sample_shape = explained_sample.shape
        self.features = self._generate_features(sample_shape)
        explained_sample = explained_sample.reshape(1, -1)
        X_flatten = X.reshape(X.shape[0], -1)
        distances = torch.cdist(X_flatten, explained_sample, p=2).cpu().numpy()
        X_pandas = pd.DataFrame(data=X_flatten.cpu().numpy(), columns=self.features)
        y_numpy = y.cpu().numpy()
        # 
        # 
        
        # data_parts = []
        # labels_parts = []
        # distance_parts = []
        # while True:
        #     try:
        #         indices, (data, target) = next(batch_generator)
        #     except StopIteration:
        #         continue_flag = False
        #         break
        #     data = torch.reshape(data, (data.shape[0], -1)).to(explained_sample.device)
        #     distances = torch.cdist(data, explained_sample, p=2).cpu().numpy()
        #     distance_parts.append(distances)
        #     data_numpy = data.cpu().numpy()
        #     data_parts.append(data_numpy)
        #     labels_numpy = target.cpu().numpy()
        #     labels_parts.append(labels_numpy)
        
        # data = pd.DataFrame(data=np.concatenate(data_parts, axis=0), columns=self.features)
        # labels = np.concatenate(labels_parts, axis=0).flatten()
        # distances = np.concatenate(distance_parts, axis=0).flatten()
        distances = np.where(distances < 1e-6, distances + 1e-6, distances).flatten()
        weights = 1 / distances
        max_weight = weights.max()
        weights /= max_weight
        return X_pandas, y_numpy, weights

class LIME:
    def __init__(self):
        pass
