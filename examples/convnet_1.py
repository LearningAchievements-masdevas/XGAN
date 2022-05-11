import torch
import torch.optim as optim
import torchvision

from xgan.models import Convnet1
from xgan import GAN
from xgan.xai import LimeRandomForest

def main():
    data_root = '_datasets'
    result_dir = '_convnet_1'
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=transform)
    gan_lr = 2e-4
    beta1 = 0.5
    criterion = torch.nn.BCELoss()

    generator_config = {
        'z_shape': (100,), 
        'output_shape': (1, 28, 28) # CHW
    }
    generator_config['model'] = Convnet1().generator(generator_config)

    discriminator_config = {
        'input_shape': (1, 28, 28),
        'criterion': criterion
    }
    discriminator_config['model'] = Convnet1().discriminator(discriminator_config)
    
    explanation_config = {
        'grad_cam': True,
        'lime': {
            'model' : LimeRandomForest(n_estimators=10, max_depth=4),
            'samples_per_class': 30000,
            'features' : ['explain_model']
        }
    }
    generation_config = {
        'samples_number': 2,
        'batch_size' : 2,
        'save_examples' : True,
        'result_dir': result_dir
    }
    
    
    generator_config['optimizer'] = optim.Adam(generator_config['model'].parameters(), lr=gan_lr, betas=(beta1, 0.999))
    discriminator_config['optimizer'] = optim.Adam(discriminator_config['model'].parameters(), lr=gan_lr, betas=(beta1, 0.999))

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gan = GAN(device, generator_config, discriminator_config, verbose=True, explanation_config=explanation_config)
    gan.clear_result_dir(generation_config)
    train_config = {
        'epochs' : 200,
        'discr_per_gener_iters' : 3,
        'iterations_between_saves': 1,
        'batch_size': 200,
        'train_dataset': trainset,
        'test_dataset': testset,
        'workers': 1
    }
    
    gan.train(train_config, generation_config)
    
    gan.generate('result', generation_config, train_config=train_config)
    # gan.get_explaination_for_sample(x)

if __name__ == '__main__':
    main()
