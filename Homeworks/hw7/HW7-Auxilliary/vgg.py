"""vgg.py: Please see the bottom for sample operations to extract the feature map using a pretrained VGG network."""


import numpy as np
import torch
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # encode 1-1
            nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 1-1
            # encode 2-1
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 2-1
            # encoder 3-1
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 3-1
            # encoder 4-1
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 4-1
            # rest of vgg not used
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 5-1
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True)
        )

    def load_weights(self, path_to_weights):
        vgg_model = torch.load(path_to_weights)
        # Don't care about the extra weights
        self.model.load_state_dict(vgg_model, strict=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        # Input is numpy array of shape (H, W, 3)
        # Output is numpy array of shape (N_l, H_l, W_l)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        out = self.model(x)
        out = out.squeeze(0).numpy()
        return out


def plot_gram(img_path: str):
    """
    Plot Gram matrix as an image, to the right of the original image
    :param img_path:
    :return:
    """
    import matplotlib.pyplot as plt
    import cv2
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    vgg = VGG19()
    vgg.load_weights('vgg_normalized.pth')
    ft = vgg(img)
    ft = ft.reshape(ft.shape[0], -1)
    gram = ft @ ft.T

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(gram)
    plt.show()


if __name__ == '__main__':
    # img_path = 'data/training/cloudy1.jpg'
    # img_path = 'data/training/rain1.jpg'
    # img_path = 'data/training/shine1.jpg'
    img_path = 'data/training/sunrise1.jpg'

    plot_gram(img_path)
    exit()

    # Load the model and the provided pretrained weights
    vgg = VGG19()
    vgg.load_weights('vgg_normalized.pth')

    training_path = 'data/training'
    testing_path = 'data/testing'

    # Gram matrix random feature
    # training_csv = 'train_gram.csv'
    # testing_csv = 'test_gram.csv'

    # Gram matrix uniform feature
    training_csv = 'train_gram_uniform.csv'
    testing_csv = 'test_gram_uniform.csv'

    # AdaIN feature
    # training_csv = 'train_adain.csv'
    # testing_csv = 'test_adain.csv'

    # extract texture feature vector, store it in csv file along with its image label
    import glob
    import csv
    import tqdm
    import os
    import cv2

    use_gram = True  # switch between Gram matrix features and AdaIN features
    pick_random = False  # whether pick random values from Gram matrix
    indices = None  # indices of picked values from flatten upper triangular Gram matrix
    C = 1024  # number of picked features from Gram matrix
    for img in tqdm.tqdm(glob.glob(testing_path + '/*.jpg')):
        print(f"Processing {img} ...")
        # Read an image into numpy array
        x = cv2.imread(img)
        # Resize the input image
        x = cv2.resize(x, (256, 256))
        # Obtain the output feature map
        ft = vgg(x)
        # print(ft.shape)
        # flatten last 2 dimensions
        ft = ft.reshape(ft.shape[0], -1)
        # print(ft.shape)

        if use_gram:
            # get Gram matrix
            gram = ft @ ft.T
            # print(gram.shape)
            # get flattened upper triangular matrix of Gram matrix
            upper = gram[np.triu_indices(gram.shape[0])]
            # print(upper.shape)
            # sample features from gram vector at fixed indices
            if indices is None:  # use the same set of indices for all images
                if pick_random:
                    indices = np.random.randint(low=0, high=len(upper), size=C)  # select C features randomly
                else:
                    indices = np.linspace(0, len(upper), num=C, endpoint=False, dtype=int)  # select C features "uniformly"
            feature = upper[indices]
            print(f"{feature.shape=}")
        else:
            # get AdaIN (mean and variance) of each channel as features
            mean = np.mean(ft, axis=1)
            var = np.var(ft, axis=1)
            feature = np.concatenate((mean, var))
            print(f"{feature.shape=}")

        # append label to the end of feature vector
        base_name = os.path.basename(img)
        if 'cloudy' in base_name:
            label = 0
        elif 'rain' in base_name:
            label = 1
        elif 'shine' in base_name:
            label = 2
        elif 'sunrise' in base_name:
            label = 3
        else:
            print(f"Cannot find label from image name {img}!")
            continue
        feature = np.append(feature, label)
        print(f"{feature}")

        with open(testing_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(feature)

    print(f"All features and labels have been written to csv file!")


