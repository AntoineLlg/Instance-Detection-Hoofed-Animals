import numpy as np
import pandas as pd
import PIL.Image as Img
import torch
import torch.nn.functional as f
from torch import Tensor
from tqdm import tqdm

def load_data(max_area=300000):
    L = np.asarray([[*Img.open(f'./HoofedAnimals/org/{i}.png').size, i] for i in range(1, 201)])
    unusable = [
        Img.open(f'./HoofedAnimals/org/{i}.png').size != Img.open(f'./HoofedAnimals/mask/{i}_mask.png').size
        for i in range(1, 201)]  # Images with size different from mask to drop later
    df = pd.DataFrame(
        {'Width': L[:, 0], 'Height': L[:, 1], 'Area': L[:, 0] * L[:, 1], 'number': L[:, 2], 'unusable': unusable})
    df.drop(df.loc[df.unusable].index, inplace=True)
    df.drop(df.loc[df.Area >= max_area].index, inplace=True)  # Drop the images that are too large
    biggest = [(df['Height'].max() // 16 + 1) * 16, (df['Width'].max() // 16 + 1) * 16]
    # More than maximum Height/Width images can take, multiples of 16

    images, target_masks = [], []
    minmax = TorchMinMaxScaler()
    for i in tqdm(df.number.to_list()):
        img = np.asarray(Img.open(f'./HoofedAnimals/org/{i}.png'))
        mask = np.asarray(Img.open(f'./HoofedAnimals/mask/{i}_mask.png'))
        size = np.array(img.shape)
        correction = (biggest - size).astype(int)
        pad = [correction[1] // 2, correction[1] - correction[1] // 2, correction[0] // 2,
               correction[0] - correction[0] // 2]
        img = torch.tensor(img)
        mask = torch.tensor(mask)

        img = f.pad(input=img, pad=pad)
        mask = f.pad(input=mask, pad=[0, 0,
                                      *pad])  # la fonction pad s'applique en premier aux dernière
        # dimensions, ie la couleur pour mask, ce qui n'est pas souhaitable
        images.append(img.detach().numpy())
        target_masks.append(mask.detach().numpy())
    target_masks = np.array(target_masks)
    images = np.array(images)
    # images = np.moveaxis(np.array([images]), 0, -1) créer un channel
    #  images_RGB = np.moveaxis(np.array([images, images, images]), 0, -1)  # conversion en RGB
    return images, target_masks


def get_mask_per_type(masks):
    masks_simplified = (masks > 0.5).astype(int)
    type_colors = {0: (1, 0, 0),
                   1: (0, 1, 0),
                   2: (0, 0, 1),
                   3: (1, 0, 1),
                   4: (1, 1, 0),
                   5: (0, 1, 1)}
    masks_simplified = np.array([np.all(masks_simplified == type_colors[i], axis=-1).astype(int) for i in range(6)])
    masks = np.moveaxis(masks_simplified, 0, 1)
    return masks


def masks_to_color_img(masks):
    colors = np.asarray([(255, 0, 0),
                         (0, 255, 0),
                         (0, 0, 255),
                         (255, 0, 255),
                         (255, 255, 0),
                         (0, 255, 255)])

    color_img = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.float32)
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:, y, x] > 0.5]

            if len(selected_colors) > 0:
                color_img[y, x, :] = np.mean(selected_colors, axis=0)

    return color_img.astype(np.uint8)


class TorchMinMaxScaler:
    def __init__(self):
        self.maxs = np.inf
        self.mins = -np.inf

    def fit(self, X):
        self.maxs = X.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        self.mins = X.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values

    def transform(self, X):
        return (X - self.mins) / (self.maxs - self.mins)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def make_agregated_masks(masks):
    return (masks.sum(axis=-1) > 0).astype(np.float64)


def loss(predictions, masks):
    intersections = predictions * masks  # pour pénaliser le manque de certitude dans les zones à détecter
    unions = predictions + masks - intersections  # pénaliser le surplus de certitude dans les zones vides
    return 1 - (intersections.sum(dim=-1).sum(dim=-1) / unions.sum(dim=-1).sum(dim=-1)).mean()
