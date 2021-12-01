import torch
import torch.optim as optim
import torchvision
from xgan.models import Convnet2
from xgan import GAN

def main():
    data_root = '_datasets'
    result_dir = '_convnet_2'
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
    discriminator_config = {
        'input_shape': (1, 28, 28),
        'criterion': criterion
    }
    explanation_config = {
        'grad_cam': True
    }
    generation_config = {
        'samples_number': 12,
        'result_dir': result_dir
    }
    generator_config['model'] = Convnet2().generator(generator_config)
    discriminator_config['model'] = Convnet2().discriminator(discriminator_config)
    generator_config['optimizer'] = optim.Adam(generator_config['model'].parameters(), lr=gan_lr, betas=(beta1, 0.999))
    discriminator_config['optimizer'] = optim.Adam(discriminator_config['model'].parameters(), lr=gan_lr, betas=(beta1, 0.999))

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gan = GAN(device, generator_config, discriminator_config, verbose=True, generation_config=generation_config, explanation_config=explanation_config)
    gan.clear_result_dir()
    # gan.prepare_explain(explain_config)
    train_config = {
        'epochs' : 200,
        'discr_per_gener_iters' : 2,
        'iterations_between_saves': 5,
        'batch_size': 200,
        'trainset': trainset,
        'testset': testset,
        'workers': 1
    }
    
    pretrained = False
    if pretrained:
        pass
        # load model
    else:
        gan.train(train_config)
    
    gan.generate('result')
    # gan.get_explaination_for_sample(x)

if __name__ == '__main__':
    main()
