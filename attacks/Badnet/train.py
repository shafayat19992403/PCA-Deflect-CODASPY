

import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import attacks.Badnet.image_train as image_train_badnet
import config
import random

def train(helper, start_epoch, local_model, target_model, is_poison,agent_name_keys):
    epochs_submit_update_dict={}
    num_samples_dict={}
    
    if helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type']==config.TYPE_TINYIMAGENET or helper.params['type'] == config.TYPE_FMNIST or helper.params['type'] == config.TYPE_EMNIST:
        epochs_submit_update_dict, num_samples_dict = image_train_badnet.ImageTrain(helper, start_epoch, local_model,
                                                                             target_model, is_poison, agent_name_keys)
    return epochs_submit_update_dict, num_samples_dict
