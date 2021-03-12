import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import json
from utils.utils import set_random_seed
import copy
from PIL import Image

DATA_PATH = '~/data/'
IMAGENET_PATH = '../../data/ImageNet'


CIFAR10_SUPERCLASS = list(range(10))  # one class
STL10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(10))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]




class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)
    
    
class ImageNetSubset(Dataset):
    def __init__(self, root, transform, split='train'):
        super(ImageNetSubset, self).__init__()

        self.root = IMAGENET_PATH
        self.transform = transform
        self.split = split
        self.labels = []
        
        # Read the subset of classes to include (sorted)
        train_list = []
        test_list = []
        
        self.img = []
        
        
        for i in range(0, 10):
            dir_name = os.path.join(IMAGENET_PATH, str(i))
            file_list = os.listdir(dir_name)
            train = np.random.choice(file_list, 1000)
            test = list(set(file_list) - set(train))
            
            if self.split == 'train':
                for file in train:
                    file_name = os.path.join(dir_name, file)
                    image = copy.deepcopy(Image.open(file_name).convert('RGB'))
                    image = np.array(image)
                    self.img.append(image)
                    self.labels.append(i)
                
            else:
                for file in test:
                    file_name = os.path.join(dir_name, file)
                    image = copy.deepcopy(Image.open(file_name))
                    image = np.array(image)
                    self.img.append(image)
                    self.labels.append(i)   
        
        self.img = np.asarray(self.img)
                

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img = self.img[index]
        img = Image.fromarray(img)
        label = self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        
        return img, label


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform

def load_mc_dataset(data_path, normal_class, known_outlier_class, n_known_outlier_classes = 0,
                 ratio_known_normal = 0.0, ratio_known_outlier = 0.0, ratio_pollution = 0.0,
                 random_state=None,train_transform=None, test_transform=None,valid_transform=None):
    dataset = MC_Dataset(root=data_path,
                              ratio_known_normal=ratio_known_normal,
                              ratio_known_outlier=ratio_known_outlier,
                              ratio_pollution=ratio_pollution,
                              train_transform=train_transform,
                              test_transform=test_transform,
                             valid_transform= valid_transform)
    
    return dataset


def create_semisupervised_setting(labels, normal_classes, outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    Create a semi-supervised data setting. 
    :param labels: np.array with labels of all dataset samples
    :param normal_classes: tuple with normal class labels
    :param outlier_classes: tuple with anomaly class labels
    :param known_outlier_classes: tuple with known (labeled) anomaly class labels
    :param ratio_known_normal: the desired ratio of known (labeled) normal samples
    :param ratio_known_outlier: the desired ratio of known (labeled) anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :return: tuple with list of sample indices, list of original labels, and list of semi-supervised labels
    """
    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
    idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)

    # Get number of samples
    n_known_normal = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_known_outlier = int(x[3])
    
    print("# of known normal: ", n_known_normal)
    print("# of known outlier: ", n_known_outlier)

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
    idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
    idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()
    torch.save(idx_unlabeled_outlier, 'known_unlabeled.variable')
    
    # Get original class labels
    labels_known_normal = labels[idx_known_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_known_outlier = labels[idx_known_outlier].tolist()

    # Get semi-supervised setting labels
    semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
    semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

    # Create final lists
    list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_known_outlier)
    print("# of training set: ", len(list_idx))

#     if SAMPLE_CHECK:
#         print('Dumping to chosen samples to ~/eco/main_train_samples')
#         with open('./main_train_samples', 'w') as f:
#             f.write('list_idx:\n')
#             f.write(str(list_idx))
#             f.write('\n\n')

#             f.write('list_labels:\n')
#             f.write(str(list_labels))
#             f.write('\n\n')

#             f.write('list_semi_labels:\n')
#             f.write(str(list_semi_labels))
#             f.write('\n\n')

#         assert(0)

    return list_idx, list_labels, list_semi_labels




class MC_Dataset():
    def __init__(self, root='../../ood_data', ratio_known_normal = 0.0, ratio_known_outlier = 0.0, ratio_pollution = 0.0, 
                 train_transform=None, test_transform=None,valid_transform=None):

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([0])
        self.outlier_classes = tuple([1])
        self.known_outlier_classes = tuple([1])

        # CIFAR-10 preprocessing: feature scaling to [0, 1]
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        
        cifar_data = datasets.CIFAR10(root, train=True, download=True).data
        # Get train set
        train_set = MC_Data(cifar_data, 'LSUN_resize', transform=train_transform)
        false_valid_set =  MC_Data(cifar_data, 'LSUN_resize', transform=test_transform)
        cifar_data = datasets.CIFAR10(root, train=False, download=True).data
        test_set =  MC_Data(cifar_data, 'LSUN_resize', transform=test_transform)
        train_set.transform2 = valid_transform

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes , self.outlier_classes, self.known_outlier_classes, ratio_known_normal, ratio_known_outlier, ratio_pollution)
        
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels
#         false_valid_set.semi_targets[idx] = torch.tensor(semi_targets) 
        
        
        #seperation from train set
#         valset_ratio = 0.05
#         valset_size = int(len(np.array(idx)[np.array(semi_targets)==0]) * valset_ratio)
#         val_idx = list(np.random.choice(np.array(idx)[np.array(semi_targets)==0],size=valset_size,replace=False))
        
#         train_idx = list(set(idx).difference(set(val_idx)))
        train_idx = idx
        print(len(train_idx))
        test_idx = [i for i in range(0, 60000)]
        test_idx = list(set(test_idx).difference(set(idx)))
        
        test_idx.pop(0)
        
        test_idx_o = list(map(lambda x: x - 40000, test_idx))
        test_idx_n = [i for i in range(0, 10000)]
        
        
#         print('val dataset:',len(val_idx))
        print("train dataset:",len(train_idx))
        print("test dataset:",len(test_idx))
        
        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, train_idx)
#         self.valid_set = Subset(train_set, val_idx)
#         self.false_valid_set = Subset(false_valid_set, idx)
        self.test_set_n = Subset(test_set, test_idx_n)
        self.test_set_o = Subset(test_set, test_idx_o)
        
        
    def loaders(self, batch_size, shuffle_train=True, shuffle_test=False, num_workers=4):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
#         false_valid_loader = DataLoader(dataset=self.false_valid_set, batch_size=batch_size, shuffle=shuffle_test,
#                                   num_workers=num_workers, drop_last=False)        
#         valid_loader = DataLoader(dataset=self.valid_set, batch_size=batch_size, shuffle=shuffle_test,
#                                   num_workers=num_workers, drop_last=False)        
        test_loader_n = DataLoader(dataset=self.test_set_n, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        test_loader_o = DataLoader(dataset=self.test_set_o, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
#         return train_loader, false_valid_loader,valid_loader, test_loader
        return train_loader, test_loader


class MC_Data(Dataset):
    def __init__(self, c_dataset, o_dataset, transform=None):
        o_dataset_root = os.path.join('../../ood_data', o_dataset)
        self.c_dataset = c_dataset
        self.o_dataset = []
        self.targets = []
        
        self.transform = transform
        
        
        for idx in os.listdir(o_dataset_root):
            image = copy.deepcopy(Image.open(os.path.join(o_dataset_root, idx)))
            image = np.array(image)
            self.o_dataset.append(image)
        
        self.o_dataset  = np.asarray(self.o_dataset)
        self.data = np.concatenate((self.c_dataset, self.o_dataset))
        
        for i in range(0, self.c_dataset.shape[0]):
            self.targets.append(0)
        
        for i in range(0, self.o_dataset.shape[0]):
            self.targets.append(1) 
        
        print(len(self.targets))
        
        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)
        
        
    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, index): 
        img, semi_target, target = self.data[index], int(self.semi_targets[index]), self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
#         raw = self.raw_tf(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target


def get_dataset(P, dataset, test_only=False, image_size=None, download=False, eval=False):
    if dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                   'places365', 'food_101', 'caltech_256', 'dtd', 'pets']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=image_size)
 
    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
        
    elif dataset == 'stl10':
        image_size = (64, 64, 3)
        n_classes = 10
        train_set = datasets.STL10(DATA_PATH, split="train", download=True, transform=train_transform)
        test_set = datasets.STL10(DATA_PATH, split="test", download=True, transform=test_transform)

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, split="train", download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, split="test", download=download, transform=test_transform)

    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 10 
        ImageNetSubset
        train_set = ImageNetSubset(root = IMAGENET_PATH, transform=train_transform, split = 'train')
        test_set = ImageNetSubset(root = IMAGENET_PATH, transform=test_transform, split = 'test')

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    elif dataset == 'stl10':
        return STL10_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.labels):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def get_subclass_contaminated_dataset(dataset, normal_classes, known_outlier_classes, ratio_known_normal, ratio_known_outlier, ratio_pollution):
    
    
    outlier_classes = list(set(dataset.labels))
    
    for normal_cls in normal_classes:
        outlier_classes.remove(normal_cls)
    
    idx_normal = np.argwhere(np.isin(dataset.labels, normal_classes)).flatten()
    idx_outlier = np.argwhere(np.isin(dataset.labels, outlier_classes)).flatten()
    idx_known_outlier_candidates = np.argwhere(np.isin(dataset.labels, known_outlier_classes)).flatten()

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)

    # Get number of samples
    n_known_normal = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_known_outlier = int(x[3])
    
    print("# of known normal: ", n_known_normal)
    print("# of known outlier: ", n_known_outlier)

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
    idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
    idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()

    # Get original class labels
    labels_known_normal = np.array(dataset.labels)[idx_known_normal].tolist()        
    labels_unlabeled_normal = np.array(dataset.labels)[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = np.array(dataset.labels)[idx_unlabeled_outlier].tolist()
    labels_known_outlier = np.array(dataset.labels)[idx_known_outlier].tolist()


    # Get semi-supervised setting labels
    semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
    semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

    # Create final lists
    list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_known_outlier)
    print("# of training set: ", len(list_idx))
    
    dataset = Subset(dataset, list_idx)
    dataset.labels = list_semi_labels
    
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform


