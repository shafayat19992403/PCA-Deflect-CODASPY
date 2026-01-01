from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.utils.data

from attacks.DBA.helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18
logger = logging.getLogger("logger")
import config
from config import device
import copy
# import cv2

import yaml

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime
import json


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
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
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
            # for y, (x, c) in enumerate(zip(xcenters, widths)):
            #     plt.text(x, y, str(int(c)), ha='center', va='center',
            #              color=text_color,fontsize='small')
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
        logger.info('get poison test loader')
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
        logger.info('Loading data')
        dataPath = './data'
        if self.params['type'] == config.TYPE_CIFAR:
            ### data load
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                             transform=transform_train)

            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)

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
            logger.info('reading data done')

        self.classes_dict = self.build_classes_dict()
        logger.info('build_classes_dict done')
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'], #100
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            ## sample indices for participants that are equally
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        logger.info('train loaders done')
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
        
        # PCA-Deflect: Track poison indices per participant
        if self.params.get('track_detection_metrics', False):
            self.poison_indices_per_participant = self._identify_poison_indices()

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

        poison_count= 0
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
        image = copy.deepcopy(ori_image)
        poison_patterns= []
        if adversarial_index==-1:
            for i in range(0,self.params['trigger_num']):
                poison_patterns = poison_patterns+ self.params[str(i) + '_poison_pattern']
        else :
            poison_patterns = self.params[str(adversarial_index) + '_poison_pattern']
        if self.params['type'] == config.TYPE_CIFAR or self.params['type'] == config.TYPE_TINYIMAGENET:
            for i in range(0,len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1


        elif self.params['type'] == config.TYPE_MNIST or self.params['type'] == config.TYPE_FMNIST or self.params['type'] == config.TYPE_EMNIST:

            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1

        return image


    # PCA-Deflect specific methods
    def _identify_poison_indices(self):
        """Identify which training samples are poisoned for each participant."""
        poison_indices_dict = {}
        
        for participant_id in self.params['adversary_list']:
            poison_indices = []
            _, data_loader = self.train_data[participant_id]
            
            sample_count = 0
            for batch_id, batch in enumerate(data_loader):
                batch_size = len(batch[0])
                # First poisoning_per_batch samples in each batch are poisoned
                for i in range(min(self.params['poisoning_per_batch'], batch_size)):
                    poison_indices.append(sample_count + i)
                sample_count += batch_size
            
            poison_indices_dict[participant_id] = set(poison_indices)
            print(f"[PCA-DEFLECT] Participant {participant_id} has {len(poison_indices)} poisoned samples")
        
        return poison_indices_dict

    def get_client_trigger_indices(self, client_id):
        """Get ground truth trigger indices for a client."""
        if hasattr(self, 'poison_indices_per_participant'):
            return self.poison_indices_per_participant.get(client_id, set())
        return set()

    def get_filtered_batch(self, batch, indices_to_exclude, evaluation=False):
        """Get batch with certain indices filtered out."""
        images, targets = batch
        mask = torch.ones(len(images), dtype=torch.bool)
        
        for idx in indices_to_exclude:
            if idx < len(images):
                mask[idx] = False
        
        filtered_images = images[mask]
        filtered_targets = targets[mask]
        
        filtered_images = filtered_images.to(device)
        filtered_targets = filtered_targets.to(device).long()
        
        if evaluation:
            filtered_images.requires_grad_(False)
            filtered_targets.requires_grad_(False)
            
        return filtered_images, filtered_targets

    def split_data_by_filtering(self, data_loader, indices_to_exclude):
        """Split data loader into filtered and excluded parts."""
        all_indices = list(range(sum(len(batch[0]) for batch in data_loader)))
        filtered_indices = [i for i in all_indices if i not in indices_to_exclude]
        excluded_indices = list(indices_to_exclude)
        
        # Create subset data loaders
        filtered_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(filtered_indices)
        )
        
        excluded_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(excluded_indices)
        )
        
        return filtered_loader, excluded_loader

    def create_client_test_dataset(self, client_id):
        """Create a test dataset for trigger detection (subset of training data)."""
        _, data_loader = self.train_data[client_id]
        
        # Collect a subset of data for testing
        test_images = []
        test_labels = []
        max_samples = 1000  # Limit for efficiency
        
        sample_count = 0
        for batch in data_loader:
            images, labels = batch
            test_images.append(images)
            test_labels.append(labels)
            sample_count += len(images)
            if sample_count >= max_samples:
                break
        
        test_images = torch.cat(test_images[:max_samples], dim=0)
        test_labels = torch.cat(test_labels[:max_samples], dim=0)
        
        return torch.utils.data.TensorDataset(test_images, test_labels)

    def get_client_data_stats(self, client_id):
        """Get statistics about client's data distribution."""
        _, data_loader = self.train_data[client_id]
        
        label_counts = {}
        total_samples = 0
        
        for batch in data_loader:
            _, labels = batch
            for label in labels:
                label_item = label.item()
                label_counts[label_item] = label_counts.get(label_item, 0) + 1
                total_samples += 1
        
        return {
            'total_samples': total_samples,
            'label_distribution': label_counts,
            'is_adversary': client_id in self.params['adversary_list']
        }


    # Add these methods to the ImageHelper class after existing methods claude

    def add_nab_stamp(self, images):
        """
        Add NAB defensive stamp to images (set top-left 2x2 pixels to 0)
        Following original NAB implementation
        """
        print(f"[NAB] Adding defensive stamp to {len(images)} images")
        stamped_images = images.clone()
        stamped_images[:, :, :config.NAB_STAMP_SIZE, :config.NAB_STAMP_SIZE] = 0.0
        return stamped_images

    def get_nab_clean_batch(self, batch, evaluation=False):
        """
        Get clean batch for NAB training (similar to get_batch but for clean data)
        Used for clean model training in NAB
        """
        data, target = batch
        data = data.to(config.device)
        target = target.to(config.device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        print(f"[NAB] Prepared clean batch: {len(data)} samples")
        return data, target

    def get_nab_poison_batch_with_stamp(self, batch, isolation_mask, pseudo_labels, evaluation=False):
        """
        Get batch with NAB defensive modifications
        Apply pseudo labels and stamps to isolated samples
        Following original NAB train.py logic
        """
        images, targets = batch
        batch_size = len(images)
        
        # Convert to tensors
        new_images = images.clone()
        new_targets = targets.clone()
        
        isolation_count = 0
        stamp_count = 0
        
        for idx in range(batch_size):
            if idx < len(isolation_mask) and isolation_mask[idx]:
                # This sample is isolated - apply NAB logic
                isolation_count += 1
                
                # Get pseudo label for this sample
                if idx < len(pseudo_labels):
                    pseudo_label = pseudo_labels[idx]
                    
                    # Check if we need to add stamp (when label != pseudo_label)
                    if targets[idx] != pseudo_label:
                        # Add defensive stamp and change label
                        new_images[idx, :, :config.NAB_STAMP_SIZE, :config.NAB_STAMP_SIZE] = 0.0
                        new_targets[idx] = pseudo_label
                        stamp_count += 1
                        print(f"[NAB] Sample {idx}: Applied stamp and changed label {targets[idx]} -> {pseudo_label}")
                    else:
                        # Just change label, no stamp needed
                        new_targets[idx] = pseudo_label
                        print(f"[NAB] Sample {idx}: Changed label {targets[idx]} -> {pseudo_label} (no stamp)")

        new_images = new_images.to(config.device)
        new_targets = new_targets.to(config.device).long()
        
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        
        print(f"[NAB] Batch processing: {isolation_count} isolated samples, {stamp_count} stamped samples")
        return new_images, new_targets

    def create_nab_test_loaders(self):
        """
        Create test loaders for NAB evaluation with and without stamps
        Following original NAB evaluate_filter.py
        """
        print(f"[NAB] Creating test loaders for stamp-based filtering")
        
        # Regular test loader (already exists)
        regular_test_loader = self.test_data
        
        # Create stamped test loader
        stamped_test_dataset = copy.deepcopy(self.test_dataset)
        
        # We'll apply stamps during testing, not to the dataset itself
        # This matches original NAB implementation
        
        print(f"[NAB] Test loaders created for filtering evaluation")
        return regular_test_loader, regular_test_loader  # We'll apply stamps during forward pass

    def get_client_loss_statistics(self, model, data_loader):
        """
        Compute loss statistics for a client's data
        Used for federated LGA detection
        Following original backdoor_detection_lga.py
        """
        print(f"[NAB] Computing loss statistics for client data")
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        losses = []
        with torch.no_grad():
            for batch_id, batch in enumerate(data_loader):
                data, targets = self.get_batch(data_loader, batch, evaluation=True)
                outputs = model(data)
                batch_losses = criterion(outputs, targets)
                losses.extend(batch_losses.cpu().numpy().tolist())
        
        losses = np.array(losses)
        stats = {
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses)),
            'min': float(np.min(losses)),
            'max': float(np.max(losses)),
            'percentiles': {
                '1': float(np.percentile(losses, 1)),
                '5': float(np.percentile(losses, 5)),
                '10': float(np.percentile(losses, 10)),
                '25': float(np.percentile(losses, 25)),
                '50': float(np.percentile(losses, 50)),
                '75': float(np.percentile(losses, 75)),
                '90': float(np.percentile(losses, 90)),
                '95': float(np.percentile(losses, 95)),
                '99': float(np.percentile(losses, 99))
            },
            'sample_count': len(losses)
        }
        
        print(f"[NAB] Loss statistics computed: mean={stats['mean']:.4f}, std={stats['std']:.4f}, samples={stats['sample_count']}")
        model.train()
        return stats

if __name__ == '__main__':
    np.random.seed(1)
    with open(f'./utils/cifar_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                        name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    pars= list(range(100))
    # show the data distribution among all participants.
    count_all= 0
    for par in pars:
        cifar_class_count = dict()
        for i in range(10):
            cifar_class_count[i] = 0
        count=0
        _, data_iterator = helper.train_data[par]
        for batch_id, batch in enumerate(data_iterator):
            data, targets= batch
            for t in targets:
                cifar_class_count[t.item()]+=1
            count += len(targets)
        count_all+=count
        print(par, cifar_class_count,count,max(zip(cifar_class_count.values(), cifar_class_count.keys())))

    print('avg', count_all*1.0/100)