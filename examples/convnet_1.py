import numpy as np
import gc

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except:
    pass

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision

from xgan.models import Convnet1
from xgan import GAN
from xgan.xai import LimeRandomForest
from xgan.utils import prepare_batches

from sklearn.ensemble import RandomForestClassifier

def convert_data_to_tensor(dataset, dtype):
    cpu = torch.device('cpu')
    batches = prepare_batches(dataset, None, 1024)
    X_list = []
    y_list = []
    for _, (batch_data, batch_labels) in batches:
        X_list.append(batch_data)
        y_list.append(batch_labels)
    X = torch.cat(X_list).to(device=cpu, dtype=dtype)
    y = torch.cat(y_list).to(device=cpu, dtype=dtype)
    print(X.shape)
    print(y.shape)
    return X, y

def main():
    data_root = '_datasets'
    result_dir = '_convnet_1'
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    dtype = torch.float32
    trainset = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=transform)
    train_X, train_y = convert_data_to_tensor(trainset, dtype)
    testset = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=transform)
    test_X, test_y = convert_data_to_tensor(testset, dtype)
    gc.collect()
    torch.cuda.empty_cache()
    gan_lr = 0.0002
    criterion = torch.nn.BCELoss()
    convnet = Convnet1()

    generator_z_shape = (20,)
    generator_config = {
        'input_shape': generator_z_shape,
        'model': convnet.generator(generator_z_shape),
        'noize_generator': lambda generator, shape: np.random.normal(0, 1, size=shape),
        'output_shape': (1, 28, 28), # CHW
        # 'grad_norm': 50
    }

    discriminator_input_shape = (1, 28, 28)
    discriminator_config = {
        'input_shape': discriminator_input_shape,
        'model': convnet.discriminator(),
        'criterion': criterion,
        # 'grad_norm': 50
    }
    
    explanation_config = {
        # 'grad_cam': True,
        # 'lime': {
        #     'model' : LimeRandomForest(n_estimators=10, max_depth=4),
        #     'samples_per_class': 10000,
        #     'features' : ['explain_model', 'nodes_count']
        # }
        # 'genspace': {
        #     'model' : RandomForestClassifier(n_estimators=100, max_depth=12),
        #     'samples_to_generate' : 1000
        # }
        'shap_gen': {
            'model': RandomForestClassifier(n_estimators=100, max_depth=12),
            'background_samples_to_gen': 100,
            'test_samples_to_gen': 5,
            'shap_nsamples': 175,
            'features': ['summary', 'waterfall'],
            # 'waterfall_samples_count': 2
        }
    }
    generation_config = {
        'samples_number': 3,
        'batch_size' : 256,
        'save_examples' : True,
        'result_dir': result_dir
    }
    
    
    generator_config['optimizer'] = optim.Adam(generator_config['model'].parameters(), lr=gan_lr, betas=(0.5, 0.999))
    # generator_config['scheduler'] = ExponentialLR(generator_config['optimizer'], gamma=0.995)
    discriminator_config['optimizer'] = optim.Adam(discriminator_config['model'].parameters(), lr=gan_lr, betas=(0.5, 0.999))
    # discriminator_config['scheduler'] = ExponentialLR(discriminator_config['optimizer'], gamma=0.995)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gan = GAN(device, generator_config, discriminator_config, verbose=True, explanation_config=explanation_config, dtype=dtype)
    gan.clear_result_dir(generation_config)
    train_config = {
        'epochs' : 200,
        'discr_per_gener_iters' : 3,
        'iterations_between_saves': 10,
        'batch_size': 1024,
        'train_dataset': train_X,
        'train_labels': train_y,
        'test_dataset': test_X,
        'test_labels': test_y,
    }
    
    gan.train(train_config, generation_config)
    
    gan.generate('result', generation_config, train_config=train_config)
    # gan.get_explaination_for_sample(x)

if __name__ == '__main__':
    main()
