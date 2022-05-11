from matplotlib import pyplot as plt
import os
import numpy as np

from .lime import LimeModel
from xgan.utils import check_for_key
from xgan.utils import gan_labels

try:
	from sklearnex import patch_sklearn
	patch_sklearn()
except:
	pass

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

class LimeRandomForest(LimeModel):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = RandomForestClassifier(*args, **kwargs)

	# def _find_superpixel(self, explained_sample):
	# 	for label_name, label_value in gan_labels.items():

	def explain_sample(self, explained_sample, X, y, formatted_idx, prefix_path, explanation_config, generation_config):
		X, y, weights = self._convert_data_to_pandas(explained_sample, X, y)
		# explained_sample = explained_sample.flatten().cpu().numpy()

		# sampled_index = np.random.choice(len(X.index), explanation_config['lime']['samples_to_explain'])
		# X_sampled = X.iloc[sampled_index]
		# y_samples = y[sampled_index]
		# weights_sampled = weights[sampled_index]
		self.model.fit(X, y, sample_weight=weights)
		# if check_for_key(explanation_config['lime'], 'explain_model', True):
		# 	self.explain_model(generation_config)
		if 'features' in explanation_config['lime'].keys():
			# if 'superpixel' in explanation_config['lime']['features']:
			# 	self._find_superpixel(explained_sample)
			if 'explain_model' in explanation_config['lime']['features']:
				self._explain_model(generation_config, formatted_idx, prefix_path)


	def _explain_model(self, generation_config, formatted_idx, prefix_path):
		shape = (3, 2)
		fig, axes = plt.subplots(*shape, figsize=(12, 8))
		shuffled_estimators = np.random.choice(self.model.estimators_, np.prod(np.array(shape)))
		for estimator_idx, plot_idx in enumerate(np.ndindex(shape)):
			estimator = shuffled_estimators[estimator_idx]
			plot_tree(estimator, ax=axes[plot_idx], feature_names=self.features)
		plt.savefig(os.path.join(prefix_path, f'{formatted_idx}_lime_model.png'), dpi=800)
		plt.close(fig)

			
			# dot_data = export_graphviz(estimator, 
   #              feature_names = self.features,
   #              rounded = True, proportion = False, 
   #              precision = 2, filled = True)
			# (graph,) = pydot.graph_from_dot_data(dot_data)

		# call(['dot', '-Tpng', 'lime_rf_tree.dot', '-o', 'tree.png', '-Gdpi=600'])
