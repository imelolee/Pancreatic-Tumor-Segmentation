import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
from scipy import ndimage
import SimpleITK as sitk

# random.seed(2022)

def load_itk_image(filename):

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing



class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
       
        D = random.randint(0, sample['image'].shape[0] - 32)
        H = random.randint(0, sample['image'].shape[1] - 128)
        W = random.randint(0, sample['image'].shape[2] - 128)

        image = image[D: D + 32, H: H + 128, W: W + 128]
        label = label[D: D + 32, H: H + 128, W: W + 128]
        
        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[image.shape[0], image.shape[1], image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[image.shape[0], image.shape[1], image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}





class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image)
        label = sample['label']
        label = np.ascontiguousarray(label)
        label = np.array(label, dtype=np.int64)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label)

        return {'image': image, 'label': label}


def transform(sample, mode='train'):
    if mode in ['train', 'val']:
        trans = transforms.Compose([
            # Random_rotate(),  # time-consuming
            Random_Crop(),
            Random_Flip(),
            Random_intencity_shift(),
            ToTensor()
        ])
    if mode == 'test':
        trans = transforms.Compose([
        # MaxMinNormalization(),
        ToTensor()
    ])
    

    return trans(sample)

    



class PancreaticDataset(Dataset):
    def __init__(self, ct_list, seg_list, mode='train'):
        self.img_path = ct_list
        self.label_path = seg_list
        self.mode = mode
        

    def __getitem__(self, item):
        if self.mode == 'train':
            image, _, _ = load_itk_image(self.img_path[item])
            label, _, _ = load_itk_image(self.label_path[item])
            if image.shape != label.shape: 
                print(self.img_path[item])       
                print(image.shape, label.shape)
           
            sample = {'image': image, 'label': label}
            sample = transform(sample, mode=self.mode)
           
            return sample['image'], sample['label']

        elif self.mode == 'val':
            image, _, _ = load_itk_image(self.img_path[item])
            label, _, _ = load_itk_image(self.label_path[item])
            if image.shape != label.shape: 
                print(self.img_path[item])       
                print(image.shape, label.shape)
           
            sample = {'image': image, 'label': label}
            sample = transform(sample, mode=self.mode)
            
            return sample['image'], sample['label']

    def __len__(self):
        return len(self.img_path)
    
    @staticmethod
    def load(filename):
        image, _, _ = load_itk_image(filename)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

        return image
        
    @staticmethod
    def collate_fn(batch):
      
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        
        return images, labels



