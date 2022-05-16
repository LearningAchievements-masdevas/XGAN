import gc
import torch
from torch.utils.data import BatchSampler, SequentialSampler
from sklearn.manifold import TSNE

class GenSpace:
	def __init__(self, gan, genspace_config):
		self.gan = gan
		self.genspace_config = genspace_config

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
        
        X = pd.DataFrame(data=np.concatenate(X_list), columns=features)
        y = np.concatenate(y_list)
        
        del batch_data, batch_labels
        gc.collect()
        torch.cuda.empty_cache()
        return X, y

	def explain_space(self, X, y, batch_size):
		self.model.fit(X, y)
		indices = BatchSampler(SequentialSampler(range(self.genspace_config['samples_to_generate'])), batch_size=batch_size, drop_last=False)
		X_list = []
		for batch_idx, batch_indices in enumerate(indices):
			generated_data = self.gan._internal_generate(len(batch_indices))
			generated_data = generated_data.reshape(generated_data.shape[0], -1)
			X_list.append(generated_data.cpu().numpy())
		X_test = pd.DataFrame(data=np.concatenate(X_list), columns=self.features)
		y_predicted = self.model.predict(X_test)
		print('# PREDICTED', y_predicted.shape, type(y_predicted))
		X_embedded = TSNE(n_components=self.n_components, learning_rate='auto', init='random').fit_transform(X_test)
		print('# EMBEDDED', X_embedded.shape, type(X_embedded))



