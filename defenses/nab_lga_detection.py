import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
import config
from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18

class NABLGADetection:
    """
    LGA (Loss-Guided Augmentation) Detection for NAB Defense
    Adapted from original backdoor_detection_lga.py for federated setting
    """
    
    def __init__(self, helper):
        print(f"[NAB_LGA] Initializing LGA Detection")
        self.helper = helper
        self.dataset_type = helper.params['type']
        self.num_classes = helper.params.get('num_classes', 10)
        
        # LGA parameters following original implementation
        self.gamma = config.NAB_LGA_GAMMA
        self.epochs = config.NAB_LGA_EPOCHS
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        print(f"[NAB_LGA] LGA parameters:")
        print(f"[NAB_LGA]   - Gamma: {self.gamma}")
        print(f"[NAB_LGA]   - Epochs: {self.epochs}")
        print(f"[NAB_LGA]   - Learning rate: {self.lr}")
        print(f"[NAB_LGA] Dataset: {self.dataset_type}, Classes: {self.num_classes}")
    
    def train_lga_model(self, global_model, agent_keys):
        """
        Train LGA model for detection following original implementation
        """
        print(f"[NAB_LGA] === TRAINING LGA MODEL ===")
        print(f"[NAB_LGA] Training LGA model for {len(agent_keys)} agents")
        
        # Create LGA model (copy of current global model)
        lga_model = self._create_lga_model()
        lga_model.load_state_dict(global_model.state_dict())
        lga_model = lga_model.to(config.device)
        
        if torch.cuda.device_count() > 1:
            lga_model = torch.nn.DataParallel(lga_model)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(lga_model.parameters(), lr=self.lr, 
                             momentum=self.momentum, weight_decay=self.weight_decay)
        
        print(f"[NAB_LGA] Starting LGA training for {self.epochs} epochs")
        
        # Training loop following original LGA implementation
        for epoch in range(self.epochs):
            start_time = time.time()
            epoch_loss, epoch_acc = self._train_epoch(lga_model, criterion, optimizer, agent_keys, epoch)
            epoch_time = time.time() - start_time
            
            print(f"[NAB_LGA] Epoch {epoch+1}/{self.epochs}: "
                  f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%, Time={epoch_time:.2f}s")
            
            # Adjust learning rate following original implementation
            self._adjust_lr(optimizer, epoch)
        
        print(f"[NAB_LGA] LGA model training completed")
        return lga_model
    
    def _train_epoch(self, model, criterion, optimizer, agent_keys, epoch):
        """
        Train one epoch with LGA loss modification
        Following original train function in backdoor_detection_lga.py
        """
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        # Train on data from all selected agents
        for agent_key in agent_keys:
            _, data_loader = self.helper.train_data[agent_key]
            
            for batch_id, batch in enumerate(data_loader):
                data, targets = self.helper.get_batch(data_loader, batch, evaluation=False)
                
                optimizer.zero_grad()
                outputs = model(data)
                
                # Apply LGA loss modification following original implementation
                loss = criterion(outputs, targets)
                lga_loss = (loss - self.gamma).abs() + self.gamma
                
                lga_loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += lga_loss.item() * len(data)
                pred = outputs.max(1)[1]
                correct += pred.eq(targets).sum().item()
                total_samples += len(data)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def compute_isolation_statistics(self, lga_model, agent_keys):
        """
        Compute loss statistics for isolation following original implementation
        """
        print(f"[NAB_LGA] === COMPUTING ISOLATION STATISTICS ===")
        print(f"[NAB_LGA] Computing isolation statistics for {len(agent_keys)} agents")
        
        lga_model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        agent_isolation_data = {}
        
        with torch.no_grad():
            for agent_key in agent_keys:
                print(f"[NAB_LGA] Processing agent {agent_key}")
                _, data_loader = self.helper.train_data[agent_key]
                
                losses = []
                indices = []
                
                for batch_id, batch in enumerate(data_loader):
                    data, targets = self.helper.get_batch(data_loader, batch, evaluation=True)
                    outputs = lga_model(data)
                    batch_losses = criterion(outputs, targets)
                    
                    losses.extend(batch_losses.cpu().numpy().tolist())
                    # Generate indices for this batch
                    batch_indices = list(range(batch_id * data_loader.batch_size, 
                                             batch_id * data_loader.batch_size + len(data)))
                    indices.extend(batch_indices)
                
                # Store isolation data for this agent
                agent_isolation_data[agent_key] = {
                    'losses': np.array(losses),
                    'indices': indices,
                    'sample_count': len(losses)
                }
                
                print(f"[NAB_LGA] Agent {agent_key}: {len(losses)} samples processed")
        
        lga_model.train()
        return agent_isolation_data
    
    def generate_isolation_masks(self, isolation_data, ratios=None):
        """
        Generate isolation masks for different ratios
        Following original isolation function
        """
        if ratios is None:
            ratios = config.NAB_ISOLATION_RATIOS
        
        print(f"[NAB_LGA] === GENERATING ISOLATION MASKS ===")
        print(f"[NAB_LGA] Isolation ratios: {ratios}")
        
        all_isolation_masks = {}
        
        for agent_key, data in isolation_data.items():
            losses = data['losses']
            sample_count = data['sample_count']
            
            agent_masks = {}
            
            for ratio in ratios:
                # Select samples with lowest losses (following original NAB)
                num_isolated = int(ratio * sample_count)
                if num_isolated > 0:
                    # Get indices of samples with lowest losses
                    isolated_indices = losses.argsort()[:num_isolated]
                    
                    # Create boolean mask
                    isolation_mask = np.zeros(sample_count, dtype=bool)
                    isolation_mask[isolated_indices] = True
                    
                    agent_masks[ratio] = isolation_mask.tolist()
                    
                    print(f"[NAB_LGA] Agent {agent_key}, ratio {ratio}: "
                          f"{num_isolated}/{sample_count} samples isolated")
                else:
                    agent_masks[ratio] = [False] * sample_count
                    print(f"[NAB_LGA] Agent {agent_key}, ratio {ratio}: No samples isolated")
            
            all_isolation_masks[agent_key] = agent_masks
        
        return all_isolation_masks
    
    def _create_lga_model(self):
        """
        Create LGA model based on dataset type using DBA's model structure
        """
        current_time = self.helper.current_time
        
        if self.dataset_type == config.TYPE_CIFAR:
            model = ResNet18(name='NAB_LGA', created_time=current_time)
            print(f"[NAB_LGA] Created LGA model: ResNet18 for CIFAR")
        elif self.dataset_type in [config.TYPE_MNIST, config.TYPE_FMNIST, config.TYPE_EMNIST]:
            model = MnistNet(name='NAB_LGA', created_time=current_time)
            print(f"[NAB_LGA] Created LGA model: MnistNet for {self.dataset_type}")
        elif self.dataset_type == config.TYPE_TINYIMAGENET:
            model = resnet18(name='NAB_LGA', created_time=current_time)
            print(f"[NAB_LGA] Created LGA model: ResNet18 for Tiny ImageNet")
        else:
            # Default fallback
            model = MnistNet(name='NAB_LGA', created_time=current_time)
            print(f"[NAB_LGA] Created default LGA model: MnistNet")
        
        return model
    
    def _adjust_lr(self, optimizer, epoch):
        """
        Adjust learning rate following original implementation
        """
        # Cosine annealing as in original implementation
        lr = 0.5 * (1 + math.cos(math.pi * epoch / (self.epochs + 80))) * self.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if epoch % 5 == 0:  # Log every 5 epochs
            print(f"[NAB_LGA] Learning rate adjusted to: {lr:.6f}")