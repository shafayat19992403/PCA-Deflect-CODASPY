import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import config
from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18

class NABPseudoLabel:
    """
    Pseudo Label Generation for NAB Defense
    Adapted from original pseudo_label_vd.py for federated setting
    """
    
    def __init__(self, helper):
        print(f"[NAB_PSEUDO] Initializing Pseudo Label Generator")
        self.helper = helper
        self.dataset_type = helper.params['type']
        self.num_classes = helper.params.get('num_classes', 10)
        
        # Training parameters following original implementation
        self.epochs = config.NAB_CLEAN_MODEL_EPOCHS
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.batch_size = helper.params.get('batch_size', 64)
        
        # Clean model for pseudo label generation
        self.clean_model = None
        
        print(f"[NAB_PSEUDO] Parameters:")
        print(f"[NAB_PSEUDO]   - Epochs: {self.epochs}")
        print(f"[NAB_PSEUDO]   - Learning rate: {self.lr}")
        print(f"[NAB_PSEUDO]   - Dataset: {self.dataset_type}")
        print(f"[NAB_PSEUDO]   - Classes: {self.num_classes}")
    
    def train_clean_model(self, global_model, agent_keys, isolation_masks):
        """
        Train clean model using only non-isolated samples
        Following original pseudo_label_vd.py implementation
        """
        print(f"[NAB_PSEUDO] === TRAINING CLEAN MODEL ===")
        print(f"[NAB_PSEUDO] Training clean model for {len(agent_keys)} agents")
        
        # Create clean model
        self.clean_model = self._create_clean_model()
        
        # Initialize with global model weights
        self.clean_model.load_state_dict(global_model.state_dict())
        self.clean_model = self.clean_model.to(config.device)
        
        if torch.cuda.device_count() > 1:
            self.clean_model = torch.nn.DataParallel(self.clean_model)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.clean_model.parameters(), lr=self.lr,
                             momentum=self.momentum, weight_decay=self.weight_decay)
        
        print(f"[NAB_PSEUDO] Starting clean model training for {self.epochs} epochs")
        
        # Training loop following original implementation
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Train only on non-isolated samples
            train_acc, train_loss = self._train_clean_epoch(
                self.clean_model, criterion, optimizer, agent_keys, isolation_masks, epoch
            )
            
            # Validate on clean test data
            val_acc = self._validate_clean_model(self.clean_model)
            
            epoch_time = time.time() - start_time
            
            print(f"[NAB_PSEUDO] Epoch {epoch+1}/{self.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Acc={val_acc:.2f}%, Time={epoch_time:.2f}s")
            
            # Adjust learning rate
            self._adjust_lr(optimizer, epoch)
        
        print(f"[NAB_PSEUDO] Clean model training completed")
        return self.clean_model
    
    def _train_clean_epoch(self, model, criterion, optimizer, agent_keys, isolation_masks, epoch):
        """
        Train one epoch using only non-isolated (clean) samples
        Following original train function in pseudo_label_vd.py
        """
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for agent_key in agent_keys:
            _, data_loader = self.helper.train_data[agent_key]
            isolation_mask = isolation_masks.get(agent_key, [])
            
            sample_idx = 0
            for batch_id, batch in enumerate(data_loader):
                data, targets = self.helper.get_batch(data_loader, batch, evaluation=False)
                
                # Filter out isolated samples from this batch
                clean_indices = []
                for i in range(len(data)):
                    if sample_idx < len(isolation_mask) and not isolation_mask[sample_idx]:
                        clean_indices.append(i)
                    elif sample_idx >= len(isolation_mask):
                        clean_indices.append(i)  # No isolation info, assume clean
                    sample_idx += 1
                
                if len(clean_indices) == 0:
                    continue  # Skip if no clean samples in batch
                
                # Extract clean samples
                clean_data = data[clean_indices]
                clean_targets = targets[clean_indices]
                
                optimizer.zero_grad()
                outputs = model(clean_data)
                loss = criterion(outputs, clean_targets)
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item() * len(clean_data)
                pred = outputs.max(1)[1]
                correct += pred.eq(clean_targets).sum().item()
                total_samples += len(clean_data)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0
        
        return accuracy, avg_loss
    
    def _validate_clean_model(self, model):
        """
        Validate clean model on test data
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_id, batch in enumerate(self.helper.test_data):
                data, targets = self.helper.get_batch(self.helper.test_data, batch, evaluation=True)
                outputs = model(data)
                pred = outputs.max(1)[1]
                correct += pred.eq(targets).sum().item()
                total += len(targets)
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        model.train()
        return accuracy
    
    def generate_pseudo_labels(self, agent_key, isolation_mask):
        """
        Generate pseudo labels for isolated samples using clean model with stamp
        Following original update_pseudo_label function
        """
        print(f"[NAB_PSEUDO] === GENERATING PSEUDO LABELS ===")
        print(f"[NAB_PSEUDO] Generating pseudo labels for agent {agent_key}")
        
        if self.clean_model is None:
            print(f"[NAB_PSEUDO] ERROR: Clean model not trained yet")
            return []
        
        self.clean_model.eval()
        pseudo_labels = []
        true_labels = []
        sample_count = 0
        
        _, data_loader = self.helper.train_data[agent_key]
        
        with torch.no_grad():
            sample_idx = 0
            for batch_id, batch in enumerate(data_loader):
                data, targets = self.helper.get_batch(data_loader, batch, evaluation=True)
                
                for i in range(len(data)):
                    if sample_idx < len(isolation_mask):
                        # Add defensive stamp and get pseudo label
                        # Following original NAB: set top-left pixels to 0
                        stamped_image = data[i:i+1].clone()
                        stamped_image[:, :, :config.NAB_STAMP_SIZE, :config.NAB_STAMP_SIZE] = 0.0
                        
                        # Get pseudo label from clean model
                        output = self.clean_model(stamped_image)
                        pseudo_label = output.max(1)[1].item()
                        
                        pseudo_labels.append(pseudo_label)
                        true_labels.append(targets[i].item())
                        sample_count += 1
                    
                    sample_idx += 1
        
        # Calculate pseudo label accuracy
        if len(pseudo_labels) > 0 and len(true_labels) > 0:
            correct_pseudo = sum(1 for p, t in zip(pseudo_labels, true_labels) if p == t)
            pseudo_acc = 100.0 * correct_pseudo / len(pseudo_labels)
            print(f"[NAB_PSEUDO] Agent {agent_key}: Generated {len(pseudo_labels)} pseudo labels")
            print(f"[NAB_PSEUDO] Pseudo label accuracy: {correct_pseudo}/{len(pseudo_labels)} ({pseudo_acc:.2f}%)")
        else:
            print(f"[NAB_PSEUDO] Agent {agent_key}: No pseudo labels generated")
        
        self.clean_model.train()
        return pseudo_labels
    
    def update_clean_model(self, global_model, agent_keys, isolation_masks):
        """
        Update existing clean model with new data
        """
        print(f"[NAB_PSEUDO] === UPDATING CLEAN MODEL ===")
        
        if self.clean_model is None:
            print(f"[NAB_PSEUDO] No existing clean model, training new one")
            return self.train_clean_model(global_model, agent_keys, isolation_masks)
        
        print(f"[NAB_PSEUDO] Updating clean model with new global model information")
        
        # Fine-tune existing clean model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.clean_model.parameters(), lr=self.lr * 0.1,  # Lower LR for fine-tuning
                             momentum=self.momentum, weight_decay=self.weight_decay)
        
        # Few epochs of fine-tuning
        fine_tune_epochs = max(1, self.epochs // 10)
        print(f"[NAB_PSEUDO] Fine-tuning for {fine_tune_epochs} epochs")
        
        for epoch in range(fine_tune_epochs):
            train_acc, train_loss = self._train_clean_epoch(
                self.clean_model, criterion, optimizer, agent_keys, isolation_masks, epoch
            )
            
            if epoch % max(1, fine_tune_epochs // 2) == 0:
                val_acc = self._validate_clean_model(self.clean_model)
                print(f"[NAB_PSEUDO] Fine-tune epoch {epoch+1}: "
                      f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        print(f"[NAB_PSEUDO] Clean model update completed")
    
    def sync_with_global_model(self, global_model):
        """Synchronize with better layer name detection"""
        if self.clean_model is None:
            return
        
        global_state = global_model.state_dict()
        clean_state = self.clean_model.state_dict()
        
        # Detect classifier layer names dynamically
        classifier_keywords = ['linear', 'fc', 'classifier', 'head']
        
        updated_state = {}
        for name, param in clean_state.items():
            is_classifier = any(keyword in name.lower() for keyword in classifier_keywords)
            
            if not is_classifier and name in global_state:
                updated_state[name] = 0.1 * global_state[name] + 0.9 * param
            else:
                updated_state[name] = param
        
        self.clean_model.load_state_dict(updated_state)
        
    def _create_clean_model(self):
        """
        Create clean model based on dataset type using DBA's model structure
        """
        current_time = self.helper.current_time
        
        if self.dataset_type == config.TYPE_CIFAR:
            model = ResNet18(name='NAB_Clean', created_time=current_time)
            print(f"[NAB_PSEUDO] Created clean model: ResNet18 for CIFAR")
        elif self.dataset_type in [config.TYPE_MNIST, config.TYPE_FMNIST, config.TYPE_EMNIST]:
            model = MnistNet(name='NAB_Clean', created_time=current_time)
            print(f"[NAB_PSEUDO] Created clean model: MnistNet for {self.dataset_type}")
        elif self.dataset_type == config.TYPE_TINYIMAGENET:
            model = resnet18(name='NAB_Clean', created_time=current_time)
            print(f"[NAB_PSEUDO] Created clean model: ResNet18 for Tiny ImageNet")
        else:
            # Default fallback
            model = MnistNet(name='NAB_Clean', created_time=current_time)
            print(f"[NAB_PSEUDO] Created default clean model: MnistNet")
        
        return model
    
    def _adjust_lr(self, optimizer, epoch):
        """
        Adjust learning rate following original implementation
        """
        if self.dataset_type == config.TYPE_CIFAR:
            # CIFAR-10 learning rate schedule from original
            if epoch < 20:
                lr = 0.01
            elif epoch < 60:
                lr = 0.001
            else:
                lr = 0.0001
        else:
            # General cosine annealing schedule
            lr = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs)) * self.lr
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if epoch % 10 == 0:  # Log every 10 epochs
            print(f"[NAB_PSEUDO] Learning rate adjusted to: {lr:.6f}")
    
    def get_clean_model_state(self):
        """
        Get current state of clean model for debugging
        """
        if self.clean_model is None:
            return None
        
        # Test clean model performance
        val_acc = self._validate_clean_model(self.clean_model)
        
        return {
            'validation_accuracy': val_acc,
            'model_parameters': sum(p.numel() for p in self.clean_model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.clean_model.parameters() if p.requires_grad)
        }
    
    def save_clean_model(self, save_path):
        """
        Save clean model for later use
        """
        if self.clean_model is None:
            print(f"[NAB_PSEUDO] No clean model to save")
            return
        
        torch.save({
            'model_state_dict': self.clean_model.state_dict(),
            'model_config': {
                'dataset_type': self.dataset_type,
                'num_classes': self.num_classes,
                'architecture': self._get_model_architecture()
            }
        }, save_path)
        
        print(f"[NAB_PSEUDO] Clean model saved to: {save_path}")
    
    def load_clean_model(self, load_path):
        """
        Load pre-trained clean model
        """
        try:
            checkpoint = torch.load(load_path, map_location=config.device)
            
            # Create model if not exists
            if self.clean_model is None:
                self.clean_model = self._create_clean_model()
                self.clean_model = self.clean_model.to(config.device)
            
            # Load state dict
            self.clean_model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"[NAB_PSEUDO] Clean model loaded from: {load_path}")
            
            # Validate loaded model
            val_acc = self._validate_clean_model(self.clean_model)
            print(f"[NAB_PSEUDO] Loaded model validation accuracy: {val_acc:.2f}%")
            
        except Exception as e:
            print(f"[NAB_PSEUDO] Error loading clean model: {e}")
            self.clean_model = None
    
    def _get_model_architecture(self):
        """
        Get model architecture name for saving
        """
        if self.dataset_type == config.TYPE_CIFAR:
            return 'ResNet18'
        elif self.dataset_type in [config.TYPE_MNIST, config.TYPE_FMNIST, config.TYPE_EMNIST]:
            return 'MnistNet'
        elif self.dataset_type == config.TYPE_TINYIMAGENET:
            return 'ResNet18_TinyImageNet'
        else:
            return 'MnistNet_Default'