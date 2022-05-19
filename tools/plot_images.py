import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import json

def _plot_grid(images, out_dir, grid_name, dim, figsize, text=None):
    font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 16,
    }
    fig, axes = plt.subplots(*dim, figsize=figsize)
    for linear_idx, idx in enumerate(np.ndindex(dim)):
        x_idx = idx[0]
        y_idx = idx[1]
        axes[x_idx][y_idx].imshow(images[linear_idx][1], interpolation='nearest', cmap='gray_r')
        if text is not None:
            axes[x_idx][y_idx].text(0, -1, text[linear_idx], fontdict=font)
        axes[x_idx][y_idx].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{grid_name}.png'))

def plotGeneratedImages(directory, out_dir, args, dim=(2, 2), figsize=(4, 4)):
    os.makedirs(out_dir, exist_ok=True)
    if args.generated:
        generated_images = []
        for address, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('_generated.png'):
                    generated_images.append((file, plt.imread(os.path.join(address, file))))
        generated_images = sorted(generated_images, key=lambda x:x[0])
        print([item[0] for item in generated_images])
        _plot_grid(generated_images, out_dir, 'generated_images', dim, figsize)
    if args.grad_cam:
        grad_cam_dir = os.path.join(directory, 'grad_cam')
        real, fake = [], []
        for address, dirs, files in os.walk(grad_cam_dir):
            for file in files:
                if file.endswith('_grad_cam_fake.png'):
                    fake.append((file, plt.imread(os.path.join(address, file))))
                elif file.endswith('_grad_cam_real.png'):
                    real.append((file, plt.imread(os.path.join(address, file))))
        real = sorted(real, key=lambda x:x[0])
        fake = sorted(fake, key=lambda x:x[0])
        print([item[0] for item in real])
        print([item[0] for item in fake])
        real_probs, fake_probs = [], []
        for probs_name, container, images in [('real_probabilities.json', real_probs, real), ('fake_probabilities.json', fake_probs, fake)]:
            with open(os.path.join(grad_cam_dir, probs_name), 'r') as f:
                probs_file = json.load(f)
            for image in images:
                filename = image[0]
                sample_number = filename.split('_')[0]
                probs_val = round(float(probs_file[sample_number]), 2)
                container.append(str(probs_val))
        print([item for item in real_probs])
        print([item for item in fake_probs])
        _plot_grid(real, out_dir, 'grad_cam_real', dim, figsize, real_probs)
        _plot_grid(fake, out_dir, 'grad_cam_fake', dim, figsize, fake_probs)
                

def main():
    parser = argparse.ArgumentParser(description='Tool for generation grid of images.')
    parser.add_argument('images', type=str, help='Input dir for images.')
    parser.add_argument('--output', default='./_result', type=str, help='Input dir for images.')
    parser.add_argument('--generated', action='store_true', help='Should app create grid with generated images.')
    parser.add_argument('--grad-cam', action='store_true', help='Should app create grids for grad_cam.')
    args = parser.parse_args()

    plotGeneratedImages(args.images, args.output, args)

if __name__ == '__main__':
    main()
