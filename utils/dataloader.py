import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from torchvision.transforms import InterpolationMode


# ---------------------------------------------------------------------------
# Augmentations. `edge` may be None (morph boundary mode): the boundary is then
# derived from the final transformed GT instead of being augmented separately.
# Masks always use nearest-neighbour interpolation so they stay binary.
# ---------------------------------------------------------------------------
def cv_random_flip(img, label, edge=None):
    if random.randint(0, 1) == 1:
        img   = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if edge is not None:
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge


def randomCrop(image, label, edge=None):
    border = 30
    image_width  = image.size[0]
    image_height = image.size[1]
    crop_win_width  = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1, (image_height + crop_win_height) >> 1)
    return (image.crop(random_region), label.crop(random_region),
            edge.crop(random_region) if edge is not None else None)


def randomRotation(image, label, edge=None):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, Image.BICUBIC)
        label = label.rotate(random_angle, Image.NEAREST)          # masks: nearest
        if edge is not None:
            edge = edge.rotate(random_angle, Image.NEAREST)
    return image, label, edge


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for _ in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        img[randX, randY] = 0 if random.randint(0, 1) == 0 else 255
    return Image.fromarray(img)


def randomPeper_eg(img, edge):
    """Legacy: salt-and-pepper noise applied identically to GT and edge map."""
    img  = np.array(img)
    edge = np.array(edge)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for _ in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        v = 0 if random.randint(0, 1) == 0 else 255
        img[randX, randY]  = v
        edge[randX, randY] = v
    return Image.fromarray(img), Image.fromarray(edge)


def binarize(mask_pil, thr=127):
    """PIL 'L' mask -> PIL 'L' mask with values in {0, 255}."""
    return Image.fromarray(((np.array(mask_pil) > thr).astype(np.uint8)) * 255)


def morph_boundary(mask_pil, kernel_size=3):
    """
    Morphological-gradient boundary (dilate - erode) of a binary mask.
    Returns a PIL 'L' image with values in {0, 255}.
    """
    m = (np.array(mask_pil) > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    b = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
    return Image.fromarray((b > 0).astype(np.uint8) * 255)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _list(root, exts):
    return sorted(root + f for f in os.listdir(root) if f.lower().endswith(exts))


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


# ---------------------------------------------------------------------------
# dataset for training
# ---------------------------------------------------------------------------
class PolypObjDataset(data.Dataset):
    """
    boundary_mode:
        'file'  -> legacy: load precomputed edge maps from edge_root (control)
        'morph' -> H1: derive boundary from the final transformed binary GT
    gt_noise: apply salt-and-pepper noise to the GT (legacy behaviour). In
        'morph' mode the boundary is derived BEFORE the noise, so noise never
        creates spurious boundary pixels.
    """

    def __init__(self, image_root, gt_root, edge_root, trainsize,
                 boundary_mode='file', boundary_kernel_size=3, gt_noise=True):
        assert boundary_mode in ('file', 'morph'), boundary_mode
        self.trainsize = trainsize
        self.boundary_mode = boundary_mode
        self.boundary_kernel_size = boundary_kernel_size
        self.gt_noise = gt_noise

        self.images = _list(image_root, ('.jpg',))
        self.gts    = _list(gt_root, ('.jpg', '.png'))
        if boundary_mode == 'file':
            assert edge_root is not None, "boundary_mode='file' requires edge_root"
            self.egs = _list(edge_root, ('.jpg', '.png'))
        else:
            self.egs = None

        self.filter_files()

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # masks: nearest-neighbour resize
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.eg_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()])

        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt    = self.binary_loader(self.gts[index])

        if self.boundary_mode == 'morph':
            gt = binarize(gt)
            image, gt, _ = cv_random_flip(image, gt, None)
            image, gt, _ = randomCrop(image, gt, None)
            image, gt, _ = randomRotation(image, gt, None)
            image = colorEnhance(image)

            # final binary mask at training resolution -> boundary target
            gt = binarize(gt.resize((self.trainsize, self.trainsize), Image.NEAREST))
            eg = morph_boundary(gt, self.boundary_kernel_size)
            if self.gt_noise:
                gt = randomPeper(gt)
        else:
            eg = self.binary_loader(self.egs[index])
            image, gt, eg = cv_random_flip(image, gt, eg)
            image, gt, eg = randomCrop(image, gt, eg)
            image, gt, eg = randomRotation(image, gt, eg)
            image = colorEnhance(image)
            if self.gt_noise:
                gt, eg = randomPeper_eg(gt, eg)

        image = self.img_transform(image)
        gt    = self.gt_transform(gt)
        eg    = self.eg_transform(eg)
        return image, gt, eg

    def filter_files(self):
        assert len(self.images) == len(self.gts), \
            f'image/GT count mismatch: {len(self.images)} vs {len(self.gts)}'
        if self.egs is not None:
            assert len(self.egs) == len(self.images), \
                f'image/edge count mismatch: {len(self.images)} vs {len(self.egs)}'

        images, gts, egs = [], [], []
        for i, (img_path, gt_path) in enumerate(zip(self.images, self.gts)):
            assert _stem(img_path) == _stem(gt_path), f'pairing mismatch: {img_path} vs {gt_path}'
            if self.egs is not None:
                assert _stem(img_path) == _stem(self.egs[i]), f'pairing mismatch: {img_path} vs {self.egs[i]}'
            with Image.open(img_path) as img, Image.open(gt_path) as gt:
                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
                    if self.egs is not None:
                        egs.append(self.egs[i])
        self.images, self.gts = images, gts
        if self.egs is not None:
            self.egs = egs

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, eg_root, batchsize, trainsize,
               shuffle=True, num_workers=0, pin_memory=False,
               boundary_mode='file', boundary_kernel_size=3, gt_noise=True, seed=None):
    dataset = PolypObjDataset(image_root, gt_root, eg_root, trainsize,
                              boundary_mode=boundary_mode,
                              boundary_kernel_size=boundary_kernel_size,
                              gt_noise=gt_noise)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return data.DataLoader(dataset=dataset,
                           batch_size=batchsize,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           worker_init_fn=worker_init_fn,
                           generator=generator)


# ---------------------------------------------------------------------------
# test dataset and loader
# ---------------------------------------------------------------------------
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = os.path.basename(self.images[self.index])

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


class tt_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts    = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt    = self.binary_loader(self.gts[self.index])
        name  = os.path.basename(self.images[self.index])

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')
