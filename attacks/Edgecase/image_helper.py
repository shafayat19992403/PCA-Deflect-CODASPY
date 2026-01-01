from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.utils.data

from attacks.Badnet.helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18
import torchvision.transforms.functional as tf

import config
from config import device
import copy
import pickle
# import cv2

from torchvision.datasets import EMNIST
from PIL import Image
import torch




import yaml

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime
from PIL import Image



class ImageHelper(Helper):

    def create_model(self):
        local_model=None
        target_model=None
        print('entered model creation')
        if self.params['type']==config.TYPE_CIFAR:
            local_model = ResNet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = ResNet18(name='Target',
                                   created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_MNIST or self.params['type']==config.TYPE_FMNIST or self.params['type']==config.TYPE_EMNIST:
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'])
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_TINYIMAGENET:

            local_model= resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])

        local_model=local_model.to(device)
        target_model=target_model.to(device)
   
        self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
        print(f"{self.params['type']} model has been created.")

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes


    def poison_test_dataset(self):
        print('get poison test loader')

        # Build dictionary of test data indices by label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        # Remove poison label indices from the normal test dataset indices
        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)

        # Create clean test loader without poison label data
        clean_test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(range_no_id)
        )

        # Create poison test loader using ARDIS poisoned dataset (ARDIS "7"s)
        if config.TYPE_EMNIST == self.params['type'] or config.TYPE_MNIST == self.params['type'] or config.TYPE_FMNIST == self.params['type']:
            ardis_poisoned_dataset = self.get_ardis_dataset_poisoned(isTest = True)
        else:
            ardis_poisoned_dataset = self.get_sw_ds_poisoned(isTest = True)
            
        poison_test_loader = torch.utils.data.DataLoader(
            ardis_poisoned_dataset,
            batch_size=self.params['batch_size'],
            shuffle=True  # shuffle is fine here since it's a separate poisoned dataset
        )

        return  poison_test_loader, clean_test_loader

    def load_data(self):
        print('Loading data')
        dataPath = './data'
        trigger_path = None
        if self.params['type'] == config.TYPE_CIFAR:
            tf_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),
                                    (0.2023,0.1994,0.2010))])
            tf_test  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),
                                    (0.2023,0.1994,0.2010))])
            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                             transform=tf_train)

            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=tf_test)
            self.tf_train = tf_train
            self.tf_test = tf_test
            
            if self.params["is_poison"]:
                sw_poisoned = self.get_sw_ds_poisoned()
                self.ardis_poison_loader = torch.utils.data.DataLoader(
                    sw_poisoned,
                    batch_size = self.params['poisoning_per_batch'],
                    shuffle    = True,
                    drop_last  = True)
                self._ardis_iter        = iter(self.ardis_poison_loader)
                self.poisoning_per_batch = self.params['poisoning_per_batch']
            
            
        elif self.params['type'] == config.TYPE_MNIST:
            # Set transforms for MNIST
            self.tf_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.tf_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.tf_train)
            self.test_dataset = datasets.MNIST('./data', train=False, transform=self.tf_test)
        
            if self.params['is_poison']:
                ardis_poisoned = self.get_ardis_dataset_poisoned()
                self.ardis_poison_loader = torch.utils.data.DataLoader(
                    ardis_poisoned,
                    batch_size = self.params['poisoning_per_batch'],
                    shuffle = True,
                    drop_last=True
                )
                self._ardis_iter = iter(self.ardis_poison_loader)
                self.poisoning_per_batch = self.params['poisoning_per_batch']



        elif self.params['type'] == config.TYPE_EMNIST:
            # Set transforms for EMNIST
            self.tf_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.tf_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.train_dataset = datasets.EMNIST('./data',split='byclass', train=True, download=True, transform=self.tf_train)
            self.test_dataset = datasets.EMNIST('./data',split='byclass', train=False, transform=self.tf_test)
            if self.params['is_poison']:
                ardis_poisoned = self.get_ardis_dataset_poisoned()
                self.ardis_poison_loader = torch.utils.data.DataLoader(
                    ardis_poisoned,
                    batch_size = self.params['poisoning_per_batch'],
                    shuffle = True,
                    drop_last=True
                )
                self._ardis_iter = iter(self.ardis_poison_loader)
                self.poisoning_per_batch = self.params['poisoning_per_batch']

        elif self.params['type'] == config.TYPE_FMNIST:
            # Set transforms for FMNIST
            self.tf_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.tf_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            self.train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                               transform=self.tf_train)
            self.test_dataset = datasets.FashionMNIST('./data', train=False, transform=self.tf_test)
            trigger_path = 'attacks/Badnet/triggers/trigger_white.png'
            
            if self.params['is_poison']:
                ardis_poisoned = self.get_ardis_dataset_poisoned()
                self.ardis_poison_loader = torch.utils.data.DataLoader(
                    ardis_poisoned,
                    batch_size = self.params['poisoning_per_batch'],
                    shuffle = True,
                    drop_last=True
                )
                self._ardis_iter = iter(self.ardis_poison_loader)
                self.poisoning_per_batch = self.params['poisoning_per_batch']
          
        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            _data_transforms = {
                'train': transforms.Compose([
                    # transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    # transforms.Resize(224),
                    transforms.ToTensor(),
                ]),
            }
            _data_dir = './data/tiny-imagenet-200/'
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                    _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                   _data_transforms['val'])
            print('reading data done')

        self.classes_dict = self.build_classes_dict()
        print('build_classes_dict done')
        if self.params['sampling_dirichlet']:
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'], 
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            ## sample indices for participants that are equally
            print('Homogenous')
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        print('train loaders done')
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_data_poison ,self.test_targetlabel_data = self.poison_test_dataset()
        self.advasarial_namelist = self.params['adversary_list']

        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        # random.shuffle(self.participants_list)
        self.benign_namelist =list(set(self.participants_list) - set(self.advasarial_namelist))


    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),pin_memory=True, num_workers=8)
        return train_loader
    
    def get_train_poison(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset_poison,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),pin_memory=True, num_workers=8)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target
    
    def get_poison_batch_test(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target, data.size(0)

    def get_poison_batch(self, bptt, adversarial_index=-1, evaluation=False):
        """Return a batch where poisoning_per_batch samples are replaced
        by ARDIS images labelled poison_label_swap."""
        data, target = bptt                      # clean MNIST
        data   = data.to(device)
        target = target.to(device)
        # print('edge-poisoning')
        
        # -- do not poison evaluation batches unless explicitly wanted
        if (not evaluation) and self.params['is_poison']:
            n_poison = min(self.poisoning_per_batch, data.size(0))

            try:
                poison_data, poison_target = next(self._ardis_iter)
            except StopIteration:
                # re‑shuffle when we exhaust the ARDIS loader
                self._ardis_iter = iter(self.ardis_poison_loader)
                poison_data, poison_target = next(self._ardis_iter)

            poison_data   = poison_data.to(device)
            poison_target = poison_target.to(device)

            # choose random positions inside the clean batch to overwrite
            idx = torch.randperm(data.size(0))[:n_poison]
            data[idx]   = poison_data[:n_poison]
            target[idx] = poison_target[:n_poison]

            poison_count = n_poison
        else:
            poison_count = 0

        # turn off grads if evaluation
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)

        return data, target, poison_count

    # ============= NAB DEFENSE SUPPORT METHODS =============

    def get_client_loss_statistics(self, model, data_loader):
        """
        Compute loss statistics for a client's data using the given model
        Required by NAB defense for loss-guided detection
        """
        print(f"[NAB] Computing client loss statistics...")
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        all_losses = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_id, batch in enumerate(data_loader):
                try:
                    data, targets = self.get_batch(data_loader, batch, evaluation=True)
                    outputs = model(data)
                    losses = criterion(outputs, targets)
                    
                    all_losses.extend(losses.cpu().numpy().tolist())
                    sample_count += len(data)
                    
                    # Limit samples for efficiency
                    if sample_count >= 1000:  # Reasonable limit
                        break
                        
                except Exception as e:
                    print(f"[NAB] Error in loss computation batch {batch_id}: {e}")
                    continue
        
        if len(all_losses) == 0:
            print(f"[NAB] WARNING: No loss statistics computed")
            return {'mean': 0.0, 'std': 0.0, 'sample_count': 0, 'percentiles': {'1': 0.0, '5': 0.0, '10': 0.0}}
        
        # Compute statistics
        losses_array = np.array(all_losses)
        stats = {
            'mean': float(np.mean(losses_array)),
            'std': float(np.std(losses_array)),
            'sample_count': len(all_losses),
            'percentiles': {
                '1': float(np.percentile(losses_array, 1)),
                '5': float(np.percentile(losses_array, 5)),
                '10': float(np.percentile(losses_array, 10))
            }
        }
        
        model.train()
        return stats

    def add_nab_stamp(self, data):
        """
        Add NAB defensive stamp to input data
        Sets top-left corner pixels to 0 (following original NAB implementation)
        """
        stamped_data = data.clone()
        stamp_size = getattr(config, 'NAB_STAMP_SIZE', 3)  # Default 3x3 stamp
        
        # Set top-left corner to 0
        if len(stamped_data.shape) == 4:  # Batch of images
            stamped_data[:, :, :stamp_size, :stamp_size] = 0.0
        elif len(stamped_data.shape) == 3:  # Single image
            stamped_data[:, :stamp_size, :stamp_size] = 0.0
        else:
            print(f"[NAB] WARNING: Unexpected data shape for stamp: {stamped_data.shape}")
        
        return stamped_data

    def get_nab_poison_batch_with_stamp(self, batch, isolation_mask, pseudo_labels, evaluation=False):
        """
        Get batch with NAB defensive modifications applied to isolated samples
        Following original NAB train_nab.py logic
        """
        data, targets = batch
        data = data.to(device)
        targets = targets.to(device)
        
        if evaluation:
            data.requires_grad_(False)
            targets.requires_grad_(False)
        
        # Apply NAB modifications to isolated samples
        if isolation_mask and pseudo_labels:
            batch_size = data.size(0)
            
            for i in range(min(batch_size, len(isolation_mask))):
                if i < len(isolation_mask) and isolation_mask[i] and i < len(pseudo_labels):
                    # Apply defensive stamp to isolated sample
                    stamped_sample = self.add_nab_stamp(data[i:i+1])
                    data[i] = stamped_sample[0]
                    
                    # Use pseudo label for isolated sample
                    if isinstance(pseudo_labels[i], (int, float)):
                        targets[i] = int(pseudo_labels[i])
                    elif hasattr(pseudo_labels[i], 'item'):
                        targets[i] = pseudo_labels[i].item()
        
        return data, targets

    def get_ardis_dataset(self, isTest=False):
        # load the data from csv's
        
        if isTest:
            ardis_images=np.loadtxt('./attacks/Edgecase/utils/Adris/ARDIS_test_2828.csv', dtype='float')
            ardis_labels=np.loadtxt('./attacks/Edgecase/utils/Adris/ARDIS_test_labels.csv', dtype='float')
        else:
            ardis_images=np.loadtxt('./attacks/Edgecase/utils/Adris/ARDIS_train_2828.csv', dtype='float')
            ardis_labels=np.loadtxt('./attacks/Edgecase/utils/Adris/ARDIS_train_labels.csv', dtype='float')


        #### reshape to be [samples][width][height]
        ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')

        # labels are one-hot encoded
        indices_seven = np.where(ardis_labels[:,7] == 1)[0]
        images_seven = ardis_images[indices_seven,:]
        # images_seven = torch.tensor(images_seven).type(torch.uint8)
        images_seven = torch.tensor(images_seven).float() / 255.0
        images_seven = images_seven.type(torch.float32)

        # labels_seven = torch.tensor([7 for y in ardis_labels])
        labels_seven = torch.full((images_seven.shape[0],), 7, dtype=torch.long)


        # ardis_dataset = datasets.MNIST('./data', train=True, download=True,
        #                 transform=transforms.Compose([
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.1307,), (0.3081,))
        #                 ]))
        ardis_dataset = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))

        ardis_dataset.data = images_seven
        ardis_dataset.targets = labels_seven
        # print(len(ardis_dataset))

        return ardis_dataset
            
    def get_ardis_dataset_poisoned(self, isTest=False):
        target_label = self.params['poison_label_swap']
        ardis_dataset = self.get_ardis_dataset(isTest=isTest)  # Your existing function
        ardis_dataset_poisoned = copy.deepcopy(ardis_dataset)
        if not isTest:
            ardis_dataset_poisoned.targets = torch.full_like(ardis_dataset_poisoned.targets, target_label)
        return ardis_dataset_poisoned
    
    def _load_blob(self, split):       # split ∈ {"train","test"}
        fname = f"southwest_images_new_{split}.pkl"  # edge‑case file names
        path  = os.path.join("./attacks/Edgecase/utils/Southwest", fname)
        with open(path, "rb") as f:
            return pickle.load(f)  
        
    def get_sw_ds(self, isTest = False):
        blob = self._load_blob("test" if isTest else "train")
        # data = torch.tensor(blob)
        labels = torch.full((blob.shape[0],), self.params['poison_label_swap'], dtype=torch.long)
        
        tf = self.tf_test if isTest else self.tf_train
        dummy = datasets.CIFAR10("./data", train=True, download=True, transform = tf)
        sw_ds = copy.deepcopy(dummy)
        # sw_ds.data = data.numpy()
        sw_ds.data = blob
        sw_ds.targets = labels.numpy().tolist()
        return sw_ds
    
    def get_sw_ds_poisoned(self, isTest=False):
        return self.get_sw_ds(isTest = isTest)