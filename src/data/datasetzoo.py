import numpy as np
from torchvision import datasets 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision import datasets, transforms
import torch


class DatasetZoo(Dataset):
    def __init__(self, root, dataset='cifar10', dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False, p_data=1.0, seed=2023):
        
        self.root = root
        self.dataset = dataset
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.p_data = p_data
        self.seed = seed
        
        self.data, self.target, self.dataobj, self.mode = self.__init_dataset__()

    def load_tinyimagenet_data(self, datadir):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4807, 0.4574, 0.4083), (0.2056, 0.2035, 0.2041)),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4807, 0.4574, 0.4083), (0.2056, 0.2035, 0.2041)),
        ])
        
        # Load TinyImageNet datasets
        train_ds = datasets.ImageNet(root=datadir , split='train', transform=transform_train)
        val_ds = datasets.ImageNet(root=datadir , split='val', transform=transform_val)
        
        # Define the 100 labels to select
        labels = [847, 874, 471, 476, 764, 138,  49, 226, 100, 426, 815, 836, 338,
                669, 743, 912, 320, 843, 796, 322, 261, 136, 841, 460, 699, 935,
                949, 877,  61, 332, 416,  35, 227, 493,  32, 478, 660,  13, 451,
                438, 323, 867, 168,  40, 156, 455, 691, 223, 354, 495, 799, 432,
                158, 866, 657, 768, 183, 852, 727, 249, 402, 507,  12, 880, 995,
                233, 176, 776, 830, 586, 865, 475, 610, 534, 953, 490, 160, 386,
                117, 942, 675,  24, 538, 494, 266, 295, 272,  11,   9, 154, 967,
                901, 123, 649, 737, 121,  20, 439, 641, 398]
        
        subset_train = np.array([], dtype='int')
        train_targets = np.array(train_ds.targets)
        for label in labels:
            subset_train = np.hstack([subset_train, np.where(train_targets == label)[0][0:500]])
        public_ds_train = torch.utils.data.Subset(train_ds, subset_train)
        
        subset_val = np.array([], dtype='int')
        val_targets = np.array(val_ds.targets)
        for label in labels:
            subset_val = np.hstack([subset_val, np.where(val_targets == label)[0]])
        public_ds_val = torch.utils.data.Subset(val_ds, subset_val)
        
        return public_ds_train, public_ds_val

    def __init_dataset__(self):
        
        if self.dataset == 'mnist':
            dataobj = datasets.MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'L'
        elif self.dataset == 'usps':
            dataobj = datasets.USPS(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'L'
        elif self.dataset == 'fmnist':
            dataobj = datasets.FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'L'
        elif self.dataset == 'cifar10':
            dataobj = datasets.CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'RGB'
        elif self.dataset == 'cifar100':
            dataobj = datasets.CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
            mode = 'RGB'
        elif self.dataset == 'svhn':
            if self.train:
                dataobj = datasets.SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = datasets.SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
            mode = 'RGB'
        elif self.dataset == 'stl10':
            if self.train:
                dataobj = datasets.STL10(self.root, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = datasets.STL10(self.root, 'test', self.transform, self.target_transform, self.download)
            mode = 'RGB'
        elif self.dataset == 'celeba':
            X_train, y_train, X_test, y_test = load_celeba_data(datadir)
            mode = 'RGB'
        elif self.dataset == 'femnist':
            X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
            mode = 'L'
        elif self.dataset == 'tinyimagenet':
            train_ds, val_ds = self.load_tinyimagenet_data("/home/mahdi/codes/data/imagenet_resized/")
            mode = 'RGB'
            dataobj = train_ds if self.train else val_ds
            data = np.array([np.array(img) for img, _ in dataobj])
            target = np.array([label for _, label in dataobj])
            print(data.dtype)
            print("length of the data",len(data))
            
            print("imagenet")
            print(data.dtype)
            print("length of the data",len(data))
            print(data.shape)
           

            if self.dataidxs is not None:
                data = data[self.dataidxs]
                target = target[self.dataidxs]
            
            if self.dataidxs is None: 
                idxs_data = np.arange(len(data))
                idxs_target = np.arange(len(target))
                
                perm_data = np.random.RandomState(seed=self.seed).permutation(len(target))
                #perm_data = idxs_data  #np.random.permutation(idxs_data) 

                p_data1 = int(len(idxs_data)*self.p_data)
                perm_data = perm_data[0:p_data1]

                data = data[perm_data] 
                target = target[perm_data]
           

            return data, target, dataobj, mode
        elif self.dataset == 'cinic10':
            cinic_mean = [0.47889522, 0.47227842, 0.43047404]
            cinic_std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean, std=cinic_std)
            ])
            mode="RGB"
            
            mode = "RGB"
    
            train_ds = datasets.ImageFolder(root='/home/mahdi/codes/data/cinic-10/train', transform=transform)
            val_ds = datasets.ImageFolder(root='/home/mahdi/codes/data/cinic-10/valid', transform=transform)
            test_ds = datasets.ImageFolder(root='/home/mahdi/codes/data/cinic-10/test', transform=transform)
            whole_ds = torch.utils.data.ConcatDataset([train_ds, val_ds, test_ds])

            train_size = 100000
            length = len(whole_ds)
            indices = list(range(length))
            
            # Shuffle indices with fixed seed
            np.random.seed(42)
            np.random.shuffle(indices)

            train_indices = indices[:train_size]
            test_indices = indices[train_size:train_size + 30000]


            train_ds_global = torch.utils.data.Subset(whole_ds, train_indices)
            test_ds_global = torch.utils.data.Subset(whole_ds, test_indices)
            
            # Assign dataobj for train and test
            dataobj = train_ds_global if self.train else test_ds_global
            
            # Extract data and targets
            x_data, y_data = zip(*[(img, label) for img, label in dataobj])
            x_data = torch.stack(x_data)
            y_data = torch.tensor(y_data)
            
            print(f"x_data: {len(x_data)}")
            print(f"y_data: {len(y_data)}")
            print(f"x_data: {x_data.shape}")
            
            # Apply permutation and filtering
            idxs_data = np.arange(len(x_data))
            perm_data = np.random.RandomState(seed=self.seed).permutation(len(idxs_data))
            p_data1 = min(int(len(idxs_data) * self.p_data), len(idxs_data))
            perm_data = perm_data[:p_data1]
            
            x_data = x_data[perm_data]
            y_data = y_data[perm_data]
            
            return x_data, y_data, dataobj, mode

        data = np.array(dataobj.data)
        try:
            target = np.array(dataobj.targets)
        except:
            target = np.array(dataobj.labels)
        
        if data.shape[2]==data.shape[3]:
            data = data.transpose(0,2,3,1) ## STL-10

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
            
        if self.dataidxs is None: 
            idxs_data = np.arange(len(data))
            idxs_target = np.arange(len(target))
            
            perm_data = np.random.RandomState(seed=self.seed).permutation(len(target))
            #perm_data = idxs_data  #np.random.permutation(idxs_data) 

            p_data1 = int(len(idxs_data)*self.p_data)
            perm_data = perm_data[0:p_data1]

            data = data[perm_data] 
            target = target[perm_data]

        return data, target, dataobj, mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        if(self.dataset=="tinyimagenet"):
            if img.shape[0] == 3:  
                img = np.transpose(img, (1, 2, 0))
        
        if(self.dataset=="tinyimagenet"):
            img = Image.fromarray((img).astype(np.uint8), mode=self.mode)
        elif(self.dataset=="cinic10"):
            img = Image.fromarray(np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        else:
            img = Image.fromarray(img, mode=self.mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            target.type(torch.LongTensor)

        return img, target

    def __len__(self):
        return len(self.data)