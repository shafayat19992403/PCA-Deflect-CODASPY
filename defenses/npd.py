import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
import config
from torch.autograd import Variable

class NPDPlugLayer(nn.Module):
    """
    Neural Polarizer Defense Plug Layer
    A lightweight layer inserted between feature extraction and classification
    """
    def __init__(self, in_channels):
        super(NPDPlugLayer, self).__init__()
        # 1x1 convolution + batch normalization
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        
        # Initialize as identity transformation
        self._init_as_identity()
        
    def _init_as_identity(self):
        """Initialize the plug layer as identity mapping"""
        with torch.no_grad():
            # Initialize conv layer as identity (weights and bias)
            self.conv.weight.zero_()
            for i in range(min(self.conv.in_channels, self.conv.out_channels)):
                self.conv.weight[i, i, 0, 0] = 1.0
            
            if self.conv.bias is not None:
                self.conv.bias.zero_()
            
            # Initialize batch norm as identity
            self.bn.weight.fill_(1.0)
            self.bn.bias.zero_()
        
    def forward(self, x):
        return self.bn(self.conv(x))

class NPDModel(nn.Module):
    """
    NPD Enhanced Model with plug layer inserted
    """
    def __init__(self, base_model, plug_layer_position='before_classifier'):
        super(NPDModel, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.plug_layer_position = plug_layer_position
        
        # Forward important attributes from the base model
        if hasattr(base_model, 'name'):
            self.name = base_model.name
        if hasattr(base_model, 'num_classes'):
            self.num_classes = base_model.num_classes
        
        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Determine where to insert plug layer and its input channels
        self.plug_layer = None
        self._insert_plug_layer()
        
    def _insert_plug_layer(self):
        """Insert plug layer at appropriate position in the model"""
        # For MnistNet models, insert before the final classifier (fc2)
        if hasattr(self.base_model, 'fc2') and hasattr(self.base_model, 'fc1'):
            # MnistNet structure: conv1 -> conv2 -> fc1 -> fc2
            in_features = self.base_model.fc2.in_features
            self.plug_layer = nn.Sequential(
                nn.Linear(in_features, in_features, bias=False),
                nn.BatchNorm1d(in_features)
            )
            self._init_linear_plug_layer()
            
        # For ResNet CIFAR models, insert before the final classifier (linear)
        elif hasattr(self.base_model, 'linear'):
            # ResNet CIFAR structure with linear layer
            in_features = self.base_model.linear.in_features
            self.plug_layer = nn.Sequential(
                nn.Linear(in_features, in_features, bias=False),
                nn.BatchNorm1d(in_features)
            )
            self._init_linear_plug_layer()
            
        # For ResNet-like models, insert before the final classifier
        elif hasattr(self.base_model, 'fc'):
            # Get feature dimension from final layer
            if hasattr(self.base_model, 'avgpool'):
                # Standard ResNet structure
                in_channels = self.base_model.fc.in_features
                # Convert to 2D for conv layer (assume 1x1 spatial size after avgpool)
                self.plug_layer = NPDPlugLayer(in_channels)
                self._modify_resnet_forward()
            else:
                # Other architectures - insert before classifier
                in_channels = self.base_model.fc.in_features
                self.plug_layer = nn.Sequential(
                    nn.Linear(in_channels, in_channels, bias=False),
                    nn.BatchNorm1d(in_channels)
                )
                self._init_linear_plug_layer()
                
        elif hasattr(self.base_model, 'classifier'):
            # VGG-like models
            if isinstance(self.base_model.classifier, nn.Sequential):
                in_features = self.base_model.classifier[0].in_features
            else:
                in_features = self.base_model.classifier.in_features
            self.plug_layer = nn.Sequential(
                nn.Linear(in_features, in_features, bias=False),
                nn.BatchNorm1d(in_features)
            )
            self._init_linear_plug_layer()
            
    def _init_linear_plug_layer(self):
        """Initialize linear plug layer as identity"""
        if isinstance(self.plug_layer, nn.Sequential):
            linear_layer = self.plug_layer[0]
            bn_layer = self.plug_layer[1]
            nn.init.eye_(linear_layer.weight)
            nn.init.ones_(bn_layer.weight)
            nn.init.zeros_(bn_layer.bias)
            
    def _modify_resnet_forward(self):
        """Modify ResNet forward pass to include plug layer"""
        self.original_forward = self.base_model.forward
        
    def forward(self, x):
        """Forward pass with plug layer"""
        if hasattr(self.base_model, 'fc2') and hasattr(self.base_model, 'fc1'):
            # MnistNet forward with plug layer
            # Apply convolutional layers
            x = F.relu(self.base_model.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.base_model.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            
            # Apply first fully connected layer
            x = F.relu(self.base_model.fc1(x))
            
            # Apply plug layer before final classification
            if self.plug_layer is not None:
                x = self.plug_layer(x)
                
            # Apply final classification layer
            x = self.base_model.fc2(x)
            return F.log_softmax(x, dim=1)
            
        elif hasattr(self.base_model, 'linear'):
            # ResNet CIFAR forward with plug layer
            # Apply all layers except the final linear layer
            x = F.relu(self.base_model.bn1(self.base_model.conv1(x)))
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            
            # Apply plug layer before final classification
            if self.plug_layer is not None:
                x = self.plug_layer(x)
                
            # Apply final classification layer
            x = self.base_model.linear(x)
            return x
            
        elif hasattr(self.base_model, 'fc') and hasattr(self.base_model, 'avgpool'):
            # ResNet-like forward with conv plug layer
            # Extract features up to avgpool
            for name, module in self.base_model.named_children():
                if name == 'fc':
                    break
                x = module(x)
                
            # Apply plug layer before final pooling and classification
            if self.plug_layer is not None:
                # For conv plug layer, apply before avgpool
                if isinstance(self.plug_layer, NPDPlugLayer):
                    x = self.plug_layer(x)
                    
            # Apply avgpool and final classification
            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.base_model.fc(x)
            
        else:
            # Other architectures - apply plug layer to flattened features
            features = self.base_model.features(x) if hasattr(self.base_model, 'features') else x
            features = torch.flatten(features, 1)
            
            if self.plug_layer is not None:
                features = self.plug_layer(features)
                
            x = self.base_model.classifier(features) if hasattr(self.base_model, 'classifier') else self.base_model.fc(features)
            
        return x
    
    def train(self, mode=True):
        """Override train mode to properly handle base model and plug layer"""
        super().train(mode)
        self.base_model.train(mode)
        if self.plug_layer is not None:
            self.plug_layer.train(mode)
        return self
    
    def eval(self):
        """Override eval mode to properly handle base model and plug layer"""
        super().eval()
        self.base_model.eval()
        if self.plug_layer is not None:
            self.plug_layer.eval()
        return self

class TPGDAttack:
    """
    Targeted Projected Gradient Descent Attack
    Used by NPD for generating adversarial examples during training
    """
    def __init__(self, model, eps=0.3, eps_iter=0.1, bounds=(-3.0, 3.0), steps=2, targeted=True):
        self.model = model
        self.eps = eps
        self.eps_iter = eps_iter  
        self.bounds = bounds
        self.steps = steps
        self.targeted = targeted
        
    def generate(self, inputs, labels):
        """Generate adversarial examples using TPGD"""
        self.model.eval()
        
        # Get target labels (second highest prediction for targeted attack)
        with torch.no_grad():
            logits = self.model(inputs)
            # Set ground truth label logits to very small value
            logits_copy = logits.clone()
            logits_copy[torch.arange(len(labels)), labels] = -1e10
            target_labels = logits_copy.argmax(1)
        
        adv_inputs = inputs.clone().detach()
        
        if self.targeted:
            for _ in range(self.steps):
                adv_inputs.requires_grad_(True)
                
                logits = self.model(adv_inputs)
                loss = F.cross_entropy(logits, target_labels)
                
                grad = torch.autograd.grad(loss, adv_inputs, 
                                         retain_graph=False, create_graph=False)[0]
                
                # Targeted attack: move towards target
                adv_inputs = adv_inputs.detach() - self.eps_iter * grad.sign()
                
                # Project to L2 ball
                delta = adv_inputs - inputs
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                delta_norm = torch.clamp(delta_norm, max=self.eps)
                delta = delta * (delta_norm / (torch.norm(delta.view(delta.size(0), -1), p=2, dim=1) + 1e-12)).view(-1, 1, 1, 1)
                
                adv_inputs = inputs + delta
                
                # Clip to valid range
                if self.bounds is not None:
                    adv_inputs = torch.clamp(adv_inputs, self.bounds[0], self.bounds[1])
                    
        return adv_inputs.detach()

class NPDDefense:
    """
    Neural Polarizer Defense Implementation
    A lightweight and effective backdoor defense via purifying poisoned features
    """
    
    def __init__(self, helper, current_time):
        print(f"[NPD_DEFENSE] Initializing Neural Polarizer Defense")
        self.helper = helper
        self.current_time = current_time
        self.dataset_type = helper.params['type']
        
        # NPD specific parameters
        self.clean_ratio = 0.05  # Ratio of clean data to use for training
        self.num_epochs = getattr(config, 'NPD_EPOCHS', 100)
        self.learning_rate = getattr(config, 'NPD_LR', 0.01)
        self.weight_decay = getattr(config, 'NPD_WEIGHT_DECAY', 0.0001)
        
        # NPD model and components
        self.npd_model = None
        self.tpgd_attacker = None
        self.clean_data = None
        
        # Training state
        self.defense_applied = False
        
        print(f"[NPD_DEFENSE] Configuration:")
        print(f"[NPD_DEFENSE]   - Clean ratio: {self.clean_ratio}")
        print(f"[NPD_DEFENSE]   - Epochs: {self.num_epochs}")
        print(f"[NPD_DEFENSE]   - Learning rate: {self.learning_rate}")
        print(f"[NPD_DEFENSE]   - Dataset type: {self.dataset_type}")
        
    def apply_defense(self, model, train_data=None):
        """
        Apply NPD defense to the backdoored model
        """
        print(f"[NPD_DEFENSE] *** Starting apply_defense method ***")
        print(f"[NPD_DEFENSE] Model type: {type(model)}")
        print(f"[NPD_DEFENSE] Train data provided: {train_data is not None}")
        
        # Create NPD enhanced model
        print(f"[NPD_DEFENSE] Creating NPD enhanced model...")
        try:
            self.npd_model = NPDModel(model)
            
            # Move model to the same device as the original model
            device = next(model.parameters()).device
            self.npd_model = self.npd_model.to(device)
            
            print(f"[NPD_DEFENSE] [SUCCESS] NPD model created successfully")
            print(f"[NPD_DEFENSE] Plug layer: {self.npd_model.plug_layer}")
        except Exception as e:
            print(f"[NPD_DEFENSE] [ERROR] Failed to create NPD model: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Setup TPGD attacker
        print(f"[NPD_DEFENSE] Setting up TPGD attacker...")
        try:
            self.tpgd_attacker = TPGDAttack(
                model=self.npd_model,
                eps_iter=0.1,
                bounds=(-3.0, 3.0),
                steps=2,
                targeted=True
            )
            print(f"[NPD_DEFENSE] [SUCCESS] TPGD attacker created successfully")
        except Exception as e:
            print(f"[NPD_DEFENSE] [ERROR] Failed to create TPGD attacker: {e}")
            return None
        
        # Prepare clean training data
        print(f"[NPD_DEFENSE] Preparing clean training data...")
        try:
            if train_data is not None:
                self.clean_data = self._get_small_clean_data(train_data)
            else:
                self.clean_data = self._get_small_clean_data()
            
            if self.clean_data is None:
                print(f"[NPD_DEFENSE] [ERROR] Failed to get clean training data")
                return None
            else:
                print(f"[NPD_DEFENSE] [SUCCESS] Clean training data prepared")
        except Exception as e:
            print(f"[NPD_DEFENSE] [ERROR] Failed to prepare clean data: {e}")
            return None
            
        # Train plug layer
        print(f"[NPD_DEFENSE] Starting plug layer training...")
        try:
            self._train_plug_layer()
            print(f"[NPD_DEFENSE] [SUCCESS] Plug layer training completed")
        except Exception as e:
            print(f"[NPD_DEFENSE] [ERROR] Plug layer training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        self.defense_applied = True
        print(f"[NPD_DEFENSE] *** Defense applied successfully ***")
        
        return self.npd_model
        
    def _get_small_clean_data(self, train_data=None):
        """Get small subset of clean training data"""
        if train_data is None:
            # Use helper's training data
            if hasattr(self.helper, 'train_data') and self.helper.train_data is not None:
                all_data = self.helper.train_data
            else:
                print("[NPD_DEFENSE] Warning: No training data available")
                return None
        else:
            all_data = train_data
        
        # Handle federated learning structure: [(participant_id, DataLoader), ...]
        if isinstance(all_data, list) and len(all_data) > 0 and isinstance(all_data[0], tuple):
            print(f"[NPD_DEFENSE] Detected federated learning structure with {len(all_data)} participants")
            # Combine datasets from all participants
            combined_datasets = []
            batch_size = 32  # Default
            
            for participant_id, participant_loader in all_data:
                if hasattr(participant_loader, 'dataset'):
                    combined_datasets.append(participant_loader.dataset)
                    batch_size = participant_loader.batch_size
            
            if combined_datasets:
                # Combine all datasets
                dataset = torch.utils.data.ConcatDataset(combined_datasets)
                total_size = len(dataset)
            else:
                print("[NPD_DEFENSE] Warning: No valid participant datasets found")
                return None
                
        # Handle standard DataLoader structure
        elif hasattr(all_data, 'dataset'):
            # DataLoader case - extract the underlying dataset
            dataset = all_data.dataset
            total_size = len(dataset)
            batch_size = all_data.batch_size
        elif hasattr(all_data, '__len__'):
            # Direct dataset case
            dataset = all_data
            total_size = len(dataset)
            batch_size = 32  # Default batch size
        else:
            print(f"[NPD_DEFENSE] Warning: Unsupported data type: {type(all_data)}")
            return None
            
        clean_size = int(total_size * self.clean_ratio)
        clean_size = max(1, clean_size)  # Ensure at least 1 sample
        
        # Create subset with random indices
        indices = np.random.choice(total_size, clean_size, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        
        # Create DataLoader for the subset
        clean_data = torch.utils.data.DataLoader(
            subset,
            batch_size=min(batch_size, clean_size),  # Don't exceed subset size
            shuffle=True
        )
        
        print(f"[NPD_DEFENSE] Selected {clean_size} clean samples from {total_size} total")
        return clean_data
        
    def _train_plug_layer(self):
        """Train the NPD plug layer"""
        print(f"[NPD_DEFENSE] *** Starting plug layer training ***")
        
        if self.clean_data is None:
            print("[NPD_DEFENSE] [ERROR] Error: No clean data available for training")
            return
            
        # Setup optimizer - only train plug layer parameters
        if self.npd_model.plug_layer is not None:
            print(f"[NPD_DEFENSE] Setting up optimizer for plug layer...")
            print(f"[NPD_DEFENSE] Plug layer type: {type(self.npd_model.plug_layer)}")
            optimizer = torch.optim.SGD(
                self.npd_model.plug_layer.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
            print(f"[NPD_DEFENSE] [SUCCESS] Optimizer created with lr={self.learning_rate}")
        else:
            print("[NPD_DEFENSE] [ERROR] Error: No plug layer found in model")
            return
            
        criterion = nn.CrossEntropyLoss()
        
        device = next(self.npd_model.parameters()).device
        print(f"[NPD_DEFENSE] Using device: {device}")
        
        print(f"[NPD_DEFENSE] Starting training loop for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            try:
                for batch_idx, (data, labels) in enumerate(self.clean_data):
                    # Ensure data and labels are tensors and on correct device
                    if not isinstance(data, torch.Tensor):
                        data = torch.tensor(data)
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels)
                        
                    data, labels = data.to(device), labels.to(device)
                    
                    # Generate adversarial examples
                    data_copy = data.clone()
                    data_pert = self.tpgd_attacker.generate(data, labels)
                    
                    # Forward pass
                    self.npd_model.plug_layer.train()  # Only plug layer in training mode
                    self.npd_model.base_model.eval()   # Base model in eval mode
                    
                    # Get predictions for both clean and perturbed data
                    logits_clean = self.npd_model(data_copy)
                    logits_pert = self.npd_model(data_pert)
                    
                    # NPD loss components
                    loss_acc = criterion(logits_clean, labels)  # Accuracy loss
                    
                    # Robustness loss - maintain predictions on clean data
                    orig_pred = logits_clean.argmax(1)
                    loss_ra = criterion(logits_pert, orig_pred)
                    
                    # BCE loss - reduce confidence on wrong predictions 
                    tmp_pred = torch.topk(logits_pert, 2, dim=1)[1]  # Top 2 predictions
                    new_targets = torch.where(tmp_pred[:, 0] == labels, tmp_pred[:, 1], tmp_pred[:, 0])
                    
                    # Use negative log likelihood for "inverse" predictions
                    log_probs = F.log_softmax(logits_pert, dim=1)
                    inv_log_probs = torch.log(1.0001 - F.softmax(logits_pert, dim=1) + 1e-12)
                    loss_bce = F.nll_loss(inv_log_probs, new_targets)
                    
                    # ASR loss - reduce attack success rate
                    target_pred = logits_clean.argmax(1)
                    loss_asr = F.nll_loss(inv_log_probs, target_pred)
                    
                    # Total loss
                    total_loss = loss_acc + loss_ra + loss_bce + loss_asr
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                    
                    if batch_idx == 0 and epoch % 20 == 0:  # Debug first batch every 20 epochs
                        print(f"[NPD_DEFENSE] Epoch {epoch}, Batch {batch_idx}: "
                              f"loss_acc={loss_acc.item():.4f}, loss_ra={loss_ra.item():.4f}, "
                              f"loss_bce={loss_bce.item():.4f}, loss_asr={loss_asr.item():.4f}, "
                              f"total={total_loss.item():.4f}")
                        
            except Exception as e:
                print(f"[NPD_DEFENSE] [ERROR] Error in training loop at epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                return
                
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            if epoch % 20 == 0:
                print(f"[NPD_DEFENSE] Epoch [{epoch+1}/{self.num_epochs}], Avg Loss: {avg_loss:.4f}")
                
        print(f"[NPD_DEFENSE] *** Plug layer training completed successfully ***")
        
    def get_model(self):
        """Get the defended model"""
        if self.defense_applied and self.npd_model is not None:
            return self.npd_model
        else:
            print("[NPD_DEFENSE] Warning: Defense not applied yet")
            return None
            
    def is_defense_applied(self):
        """Check if defense has been applied"""
        return self.defense_applied
        
    def reset_defense(self):
        """Reset defense state"""
        self.defense_applied = False
        self.npd_model = None
        self.tpgd_attacker = None
        self.clean_data = None
        print("[NPD_DEFENSE] Defense state reset")
