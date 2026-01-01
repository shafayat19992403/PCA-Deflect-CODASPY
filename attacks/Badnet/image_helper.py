from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn

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
# import cv2

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

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0]) #for cifar: 5000
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)
        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
        return per_participant_list

    def draw_dirichlet_plot(self,no_classes,no_participants,image_nums,alpha):
        fig= plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par=[0,10,20,30]
        for k in range(no_participants):
        # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
           
            left += s[k]
        plt.legend(ncol=20,loc='lower left',  bbox_to_anchor=(0, 1),fontsize=4) #
        # plt.legend(ncol=len(vis_par), bbox_to_anchor=(0, 1),
        #            loc='lower left', fontsize='small')
        plt.xlabel("Number of Images", fontsize=16)
        # plt.ylabel("Label 0 ~ 199", fontsize=16)
        # plt.yticks([])
        fig.tight_layout(pad=0.1)
        # plt.ylabel("Label",fontsize='small')
        fig.savefig(self.folder_path+'/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        print('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range_no_id)), \
               torch.utils.data.DataLoader(self.test_dataset,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds))
    def load_data(self):
        print('Loading data')
        dataPath = './data'
        trigger_path = None
        if self.params['type'] == config.TYPE_CIFAR:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                             transform=transform_train)

            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)
            trigger_path = 'attacks/Badnet/triggers/trigger_10.png'
            

        elif self.params['type'] == config.TYPE_MNIST:

            self.train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            self.test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]))
            trigger_path = 'attacks/Badnet/triggers/trigger_white.png'
        elif self.params['type'] == config.TYPE_EMNIST:

            self.train_dataset = datasets.EMNIST('./data',split='byclass', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            self.test_dataset = datasets.EMNIST('./data',split='byclass', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]))
            trigger_path = 'attacks/Badnet/triggers/trigger_white.png'

        elif self.params['type'] == config.TYPE_FMNIST:
            self.train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            self.test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]))
            trigger_path = 'attacks/Badnet/triggers/trigger_white.png'
          
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

        
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = self.params['trigger_size']
        self.trigger_img = self.trigger_img.resize((self.trigger_size, self.trigger_size))


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

    def get_poison_batch(self, bptt,adversarial_index=-1, evaluation=False):

        images, targets = bptt

        poison_count = 0
        new_images=images
        new_targets=targets

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                poison_count+=1

            else: # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count

    def add_pixel_pattern(self,ori_image,adversarial_index):
        img_w, img_h = ori_image.shape[1], ori_image.shape[2]
        ori_image = tf.to_pil_image(ori_image)
        ori_image.paste(self.trigger_img, (img_w - self.trigger_size, img_h - self.trigger_size))
        return transforms.ToTensor()(ori_image)

    # ==================== NAB DEFENSE INTEGRATION METHODS ====================
    
    def get_client_loss_statistics(self, model, data_loader):
        """
        Compute loss statistics for LGA detection
        Required by NAB defense
        """
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        losses = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_id, batch in enumerate(data_loader):
                data, targets = self.get_batch(data_loader, batch, evaluation=True)
                outputs = model(data)
                batch_losses = criterion(outputs, targets)
                
                losses.extend(batch_losses.cpu().numpy().tolist())
                sample_count += len(data)
                
                # Limit samples for efficiency
                if sample_count >= 1000:
                    break
        
        losses = np.array(losses)
        
        # Compute statistics
        stats = {
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses)),
            'sample_count': len(losses),
            'percentiles': {
                '1': float(np.percentile(losses, 1)),
                '5': float(np.percentile(losses, 5)),
                '10': float(np.percentile(losses, 10))
            }
        }
        
        model.train()
        return stats
    
    def add_nab_stamp(self, data):
        """
        Add NAB defensive stamp to data
        Sets top-left corner pixels to 0 (black) to disrupt triggers
        """
        stamped_data = data.clone()
        
        # For MNIST family datasets (1-channel)
        if len(data.shape) == 4:  # Batch of images
            stamped_data[:, :, :config.NAB_STAMP_SIZE, :config.NAB_STAMP_SIZE] = 0.0
        elif len(data.shape) == 3:  # Single image
            stamped_data[:, :config.NAB_STAMP_SIZE, :config.NAB_STAMP_SIZE] = 0.0
        
        return stamped_data
    
    def get_nab_poison_batch_with_stamp(self, batch, isolation_mask, pseudo_labels, evaluation=False):
        """
        Get batch with NAB defensive modifications
        Applied to isolated samples during NAB training
        """
        images, targets = batch
        
        poison_count = 0
        new_images = images.clone()
        new_targets = targets.clone()
        
        # Apply NAB modifications to isolated samples
        sample_idx = 0
        for index in range(len(images)):
            if sample_idx < len(isolation_mask) and isolation_mask[sample_idx]:
                # This sample is isolated (suspicious)
                if sample_idx < len(pseudo_labels):
                    # Apply defensive stamp
                    new_images[index] = self.add_nab_stamp(images[index])
                    # Use pseudo label
                    new_targets[index] = pseudo_labels[sample_idx]
                    poison_count += 1
            
            sample_idx += 1
        
        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        
        return new_images, new_targets, poison_count