import matplotlib.pyplot as plt
import os

def plotGeneratedImages(directory, result, dim=(3, 3), figsize=(3, 3)):
    #noise = np.random.normal(0, 1, size=[examples, randomDim])
    #generatedImages = generator.predict(noise)
    images = []
    for address, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('png'):
                images.append(plt.imread(os.path.join(address, file)))
    plt.figure(figsize=figsize)
    for i in range(9):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{result}.png')

for number in [10, 25, 50, 75, 100, 125, 150, 175]:
    for postfix in ['', '_grad_cam_real', '_grad_cam_fake']:
        string = f'generated_epoch_{number}{postfix}'
        plotGeneratedImages(f'/home/masdevas/Documents/Projects/_convnet_2_v1/24_11_2021-01_24_19/{string}', string)