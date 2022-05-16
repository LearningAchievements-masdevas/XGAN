from matplotlib import pyplot as plt
import os
import numpy as np

from .lime import LimeModel
from xgan.utils import check_for_key
from xgan.utils import gan_labels

# try:
# 	from sklearnex import patch_sklearn
# 	patch_sklearn()
# except:
# 	pass

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

class LimeRandomForest(LimeModel):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = RandomForestClassifier(*args, **kwargs)

	def explain_sample(self, explained_sample, X, y, weights, features, prefix_path, formatted_idx, explanation_config, generation_config):
		self.model.fit(X, y, sample_weight=weights)
		explanation = {}
		if 'features' in explanation_config['lime'].keys():
			if 'explain_model' in explanation_config['lime']['features']:
				self._explain_model(generation_config, formatted_idx, features, prefix_path)
			if 'nodes_count' in explanation_config['lime']['features']:
				explanation['nodes_count'] = self._calculate_nodes_count()
		return explanation

	def _calculate_nodes_count(self):
		nodes_count = 0
		for estimator in self.model.estimators_:
			nodes_count += estimator.tree_.node_count
		return nodes_count

	def _explain_model(self, generation_config, formatted_idx, features, prefix_path):
		shape = (3, 2)
		fig, axes = plt.subplots(*shape, figsize=(12, 8))
		shuffled_estimators = np.random.choice(self.model.estimators_, np.prod(np.array(shape)))
		class_names = [str(key) for key in gan_labels.keys()]
		for estimator_idx, plot_idx in enumerate(np.ndindex(shape)):
			estimator = shuffled_estimators[estimator_idx]
			plot_tree(estimator, ax=axes[plot_idx], feature_names=features, class_names=class_names)
		plt.savefig(os.path.join(prefix_path, f'{formatted_idx}_lime_model.png'), dpi=800)
		plt.close(fig)
