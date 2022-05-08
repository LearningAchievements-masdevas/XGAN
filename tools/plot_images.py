import matplotlib.pyplot as plt
import os
import argparse

def plotGeneratedImages(directory, result, dim=(3, 3), figsize=(3, 3)):
    #noise = np.random.normal(0, 1, size=[examples, randomDim])
    #generatedImages = generator.predict(noise)
    images = []
    for address, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('png'):
                images.append((file, plt.imread(os.path.join(address, file))))
    plt.figure(figsize=figsize)
    for i in range(9):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(images[i][0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{result}.png')

def main():
    parser = argparse.ArgumentParser(description='Tool for generation grid of images.')
    parser.add_argument('images', type=str, help='Input dir for images.')
    args = parser.parse_args()
    plotGeneratedImages(args.images, os.path.join(args.images, 'grid'))

if __name__ == '__main__':
    main()
