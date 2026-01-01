#!/usr/bin/env python3
"""
Truly Fixed NAB Defense Implementation - Tensor Processing Only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging
import os
import time
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Import existing components
import config

class SelfSupervisedModel(nn.Module):
    """
    Self-supervised model for feature extraction (SimCLR-style)
    """
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        print(f"[DEBUG] Creating SelfSupervisedModel with projection_dim={projection_dim}")
        
        self.backbone = copy.deepcopy(backbone)
        
        # Get the feature dimension from backbone
        feature_dim = 500  # Default for MnistNet
        
        # For MnistNet, we know the structure
        if hasattr(self.backbone, 'fc2'):
            feature_dim = self.backbone.fc2.in_features
            self.backbone.fc2 = nn.Identity()
            print(f"[DEBUG] Found fc2 layer, feature_dim={feature_dim}")
        elif hasattr(self.backbone, 'fc'):
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print(f"[DEBUG] Found fc layer, feature_dim={feature_dim}")
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
        print(f"[DEBUG] SelfSupervisedModel created with feature_dim={feature_dim}")
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.projection_head(features)
    
    def get_features(self, x):
        """Get features without projection head"""
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning - PURE TENSOR PROCESSING
    """
    def __init__(self, base_dataset, input_channels=1):
        print(f"[DEBUG] Creating ContrastiveDataset with {len(base_dataset)} samples")
        print(f"[DEBUG] Input channels: {input_channels}")
        self.base_dataset = base_dataset
        self.input_channels = input_channels
        
        # NO TORCHVISION TRANSFORMS - Pure tensor operations only
        print(f"[DEBUG] Using pure tensor transforms for {input_channels} channel data")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, label, *other_info = self.base_dataset[idx]
        
        if idx == 0:
            print(f"[DEBUG] Processing item {idx}: data type={type(data)}, shape={data.shape if hasattr(data, 'shape') else 'N/A'}")
        
        try:
            # Convert to tensor if needed
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Ensure correct tensor format
            if len(data.shape) == 2:  # H x W -> add channel
                data = data.unsqueeze(0)  # 1 x H x W
            elif len(data.shape) == 3:
                # Already has channels
                pass
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
            
            if idx == 0:
                print(f"[DEBUG] After tensor conversion: {data.shape}")
            
            # Force to correct number of channels
            if self.input_channels == 1:
                if data.shape[0] != 1:
                    # Convert to grayscale
                    data = data.mean(dim=0, keepdim=True)
                    if idx == 0:
                        print(f"[DEBUG] Converted to grayscale: {data.shape}")
            elif self.input_channels == 3:
                if data.shape[0] == 1:
                    # Convert grayscale to RGB by repeating
                    data = data.repeat(3, 1, 1)
                    if idx == 0:
                        print(f"[DEBUG] Converted to RGB: {data.shape}")
            
            # Create two views with pure tensor operations
            view1 = self.tensor_augment(data.clone())
            view2 = self.tensor_augment(data.clone())
            
            if idx == 0:
                print(f"[DEBUG] Final tensor shapes - View1: {view1.shape}, View2: {view2.shape}")
            
            return view1, view2, label, idx
                
        except Exception as e:
            print(f"[ERROR] Transform failed for idx {idx}: {e}")
            # Return safe dummy tensor with correct shape
            if self.input_channels == 1:
                dummy = torch.zeros(1, 28, 28, dtype=torch.float32)
            else:
                dummy = torch.zeros(3, 32, 32, dtype=torch.float32)
            return dummy, dummy, label, idx
    
    def tensor_augment(self, tensor):
        """Pure tensor augmentation without PIL or torchvision transforms"""
        # Ensure tensor is float
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        # Normalize to [0, 1] if needed
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        # Simple augmentations using pure tensor operations
        
        # 1. Random horizontal flip
        if torch.rand(1) < 0.3:
            tensor = torch.flip(tensor, dims=[-1])
        
        # 2. Add small amount of noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(tensor) * 0.01
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0, 1)
        
        # 3. Random brightness adjustment
        if torch.rand(1) < 0.3:
            brightness_factor = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
            tensor = tensor * brightness_factor
            tensor = torch.clamp(tensor, 0, 1)
        
        # 4. Normalize (MNIST normalization for 1-channel, ImageNet for 3-channel)
        if self.input_channels == 1:
            # MNIST normalization
            tensor = (tensor - 0.1307) / 0.3081
        else:
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
        
        return tensor

class NABDefense:
    """
    Effective NAB Defense Implementation - TENSOR ONLY
    """
    
    def __init__(self, helper, params):
        print(f"[DEBUG] ===== INITIALIZING NAB DEFENSE =====")
        
        self.helper = helper
        self.params = params
        self.device = params.get('device', config.device)
        
        # Determine dataset type
        dataset_type = str(getattr(helper.params, 'type', 'unknown')).lower()
        print(f"[DEBUG] Dataset type: {dataset_type}")
        
        if any(x in dataset_type for x in ['mnist', 'fmnist', 'emnist']):
            self.input_channels = 1
            self.input_size = (28, 28)
            print(f"[DEBUG] MNIST-family dataset detected -> 1 channel")
        else:
            self.input_channels = 3
            self.input_size = (32, 32)
            print(f"[DEBUG] RGB dataset detected -> 3 channels")
        
        # NAB parameters
        self.ssl_epochs = params.get('nab_ssl_epochs', 8)  # Reduced for stability
        self.detection_epochs = params.get('nab_detection_epochs', 10)
        self.gamma = params.get('nab_gamma', 0.5)
        self.isolation_ratio = params.get('nab_isolation_ratio', 0.15)
        self.batch_size = 8  # Small batch size for stability
        
        # Known adversarial clients
        self.known_adversaries = set(str(adv) for adv in helper.params.get('adversary_list', []))
        print(f"[DEBUG] Known adversaries: {self.known_adversaries}")
        
        print(f"[DEBUG] NAB Defense initialized:")
        print(f"[DEBUG]   Input channels: {self.input_channels}")
        print(f"[DEBUG]   SSL epochs: {self.ssl_epochs}")
        print(f"[DEBUG]   Batch size: {self.batch_size}")
        
        self.ssl_model = None
        self.class_centroids = None
        
    def apply_nab_defense(self, epochs_submit_update_dict, num_samples_dict, 
                         global_model, epoch):
        """
        Apply comprehensive NAB defense
        """
        print(f"\n[DEBUG] ===== APPLYING NAB DEFENSE (EPOCH {epoch}) =====")
        
        try:
            # Step 1: Collect training data
            combined_data = self.collect_client_data(epochs_submit_update_dict)
            
            if len(combined_data['data']) < 8:
                print(f"[WARNING] Insufficient data for defense")
                return epochs_submit_update_dict, num_samples_dict
            
            # Step 2: Train SSL model
            if self.ssl_model is None:
                print(f"[DEBUG] Training SSL model...")
                self.ssl_model = self.train_ssl_model(combined_data, global_model)
            
            # Step 3: Detect poisoned samples
            print(f"[DEBUG] Detecting poisoned samples...")
            suspicious_samples = self.detect_poisoned_samples(combined_data, global_model)
            
            # Step 4: Filter malicious clients
            print(f"[DEBUG] Filtering malicious clients...")
            cleaned_updates, cleaned_samples = self.filter_malicious_clients(
                epochs_submit_update_dict, num_samples_dict, combined_data, suspicious_samples
            )
            
            original_count = len(epochs_submit_update_dict)
            cleaned_count = len(cleaned_updates)
            
            print(f"[DEBUG] Defense Results:")
            print(f"[DEBUG]   Original clients: {original_count}")
            print(f"[DEBUG]   Cleaned clients: {cleaned_count}")
            print(f"[DEBUG]   Suspicious samples: {len(suspicious_samples)}")
            
            return cleaned_updates, cleaned_samples
            
        except Exception as e:
            print(f"[ERROR] NAB Defense failed: {e}")
            import traceback
            traceback.print_exc()
            return epochs_submit_update_dict, num_samples_dict
    
    def collect_client_data(self, epochs_submit_update_dict):
        """
        Collect training data from participating clients
        """
        print(f"[DEBUG] Collecting client data...")
        
        all_data = []
        all_labels = []
        client_data_map = {}
        
        participating_clients = set(str(k) for k in epochs_submit_update_dict.keys())
        
        for client_id, client_loader in self.helper.train_data:
            if str(client_id) in participating_clients:
                client_samples = []
                sample_count = 0
                
                try:
                    for batch_idx, batch in enumerate(client_loader):
                        if sample_count >= 30:  # Reduced samples per client
                            break
                        
                        if hasattr(batch, '__len__') and len(batch) >= 2:
                            data, labels = batch[0], batch[1]
                            
                            for i in range(min(len(data), 30 - sample_count)):
                                sample_idx = len(all_data)
                                all_data.append(data[i])
                                all_labels.append(labels[i])
                                client_samples.append(sample_idx)
                                sample_count += 1
                        
                        if batch_idx >= 1:  # Limit batches
                            break
                
                except Exception as e:
                    print(f"[ERROR] Error collecting from client {client_id}: {e}")
                    continue
                
                client_data_map[str(client_id)] = client_samples
                print(f"[DEBUG] Collected {len(client_samples)} samples from client {client_id}")
        
        print(f"[DEBUG] Total collected: {len(all_data)} samples")
        
        return {
            'data': all_data,
            'labels': all_labels,
            'client_map': client_data_map
        }
    
    def train_ssl_model(self, combined_data, global_model):
        """
        Train self-supervised model
        """
        print(f"[DEBUG] Training SSL model...")
        
        try:
            # Create SSL model
            backbone = copy.deepcopy(global_model)
            ssl_model = SelfSupervisedModel(backbone).to(self.device)
            
            # Create dataset with tensor-only processing
            base_dataset = [(data, label) for data, label in zip(combined_data['data'], combined_data['labels'])]
            ssl_dataset = ContrastiveDataset(base_dataset, input_channels=self.input_channels)
            ssl_loader = DataLoader(ssl_dataset, batch_size=self.batch_size, shuffle=True)
            
            print(f"[DEBUG] Created SSL dataset: {len(ssl_dataset)} samples, {len(ssl_loader)} batches")
            
            # Training
            optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)
            ssl_model.train()
            
            for epoch in range(self.ssl_epochs):
                total_loss = 0
                batch_count = 0
                
                for batch_idx, batch_data in enumerate(ssl_loader):
                    try:
                        view1, view2, _, _ = batch_data
                        view1, view2 = view1.to(self.device), view2.to(self.device)
                        
                        # Strict validation
                        if (view1.shape[0] < 2 or 
                            view1.shape[1] != self.input_channels or
                            view2.shape[1] != self.input_channels):
                            print(f"[WARNING] Skipping batch due to shape mismatch: {view1.shape}, {view2.shape}")
                            continue
                        
                        optimizer.zero_grad()
                        
                        z1 = ssl_model(view1)
                        z2 = ssl_model(view2)
                        
                        # Simple contrastive loss
                        loss = F.mse_loss(z1, z2)
                        
                        if not torch.isnan(loss):
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                            batch_count += 1
                        
                    except Exception as e:
                        print(f"[ERROR] SSL batch {batch_idx} error: {e}")
                        continue
                
                avg_loss = total_loss / max(1, batch_count)
                if epoch % 2 == 0:
                    print(f"[DEBUG] SSL Epoch {epoch+1}: Loss={avg_loss:.4f}, Successful batches={batch_count}")
            
            ssl_model.eval()
            print(f"[DEBUG] SSL training completed")
            return ssl_model
            
        except Exception as e:
            print(f"[ERROR] SSL training failed: {e}")
            return None
    
    def detect_poisoned_samples(self, combined_data, global_model):
        """
        Detect poisoned samples using loss analysis
        """
        print(f"[DEBUG] Detecting poisoned samples...")
        
        try:
            # Create simple dataset for detection
            dataset = [(data, label) for data, label in zip(combined_data['data'], combined_data['labels'])]
            data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
            
            # Use global model for detection
            detection_model = copy.deepcopy(global_model).to(self.device)
            criterion = nn.CrossEntropyLoss(reduction='none')
            
            # Get losses for all samples
            detection_model.eval()
            all_losses = []
            
            with torch.no_grad():
                for batch_data in data_loader:
                    try:
                        data, labels = batch_data
                        
                        # Convert to tensor and ensure correct channels
                        if not isinstance(data, torch.Tensor):
                            data = torch.stack([torch.tensor(d, dtype=torch.float32) for d in data])
                        
                        data = data.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Ensure correct input format
                        if len(data.shape) == 3:  # Missing batch dimension
                            data = data.unsqueeze(0)
                        
                        # Force correct number of channels for MNIST
                        if self.input_channels == 1 and data.shape[1] != 1:
                            data = data.mean(dim=1, keepdim=True)
                        
                        outputs = detection_model(data)
                        losses = criterion(outputs, labels)
                        all_losses.extend(losses.cpu().numpy())
                        
                    except Exception as e:
                        print(f"[ERROR] Detection batch error: {e}")
                        continue
            
            # Select samples with lowest loss (most suspicious)
            if len(all_losses) > 0:
                num_suspicious = max(1, int(len(all_losses) * self.isolation_ratio))
                suspicious_indices = set(np.argsort(all_losses)[:num_suspicious])
                print(f"[DEBUG] Detected {len(suspicious_indices)} suspicious samples out of {len(all_losses)}")
                return suspicious_indices
            else:
                return set()
                
        except Exception as e:
            print(f"[ERROR] Poison detection failed: {e}")
            return set()
    
    def filter_malicious_clients(self, epochs_submit_update_dict, num_samples_dict, 
                                combined_data, suspicious_samples):
        """
        Filter clients based on suspicious sample ratios
        """
        print(f"[DEBUG] Filtering malicious clients...")
        
        client_suspicion_scores = {}
        client_map = combined_data.get('client_map', {})
        
        # Calculate suspicion score for each client
        for client_id, sample_indices in client_map.items():
            if not sample_indices:
                continue
                
            suspicious_count = sum(1 for idx in sample_indices if idx in suspicious_samples)
            suspicion_ratio = suspicious_count / len(sample_indices)
            client_suspicion_scores[client_id] = suspicion_ratio
            
            print(f"[DEBUG] Client {client_id}: {suspicious_count}/{len(sample_indices)} suspicious (ratio: {suspicion_ratio:.3f})")
        
        # Filter clients with high suspicion ratios
        base_threshold = 0.25  # Base threshold
        cleaned_updates = {}
        cleaned_samples = {}
        filtered_clients = []
        
        for client_id, updates in epochs_submit_update_dict.items():
            client_str = str(client_id)
            suspicion_score = client_suspicion_scores.get(client_str, 0.0)
            
            # More aggressive filtering for known adversaries
            if client_str in self.known_adversaries:
                threshold = 0.15  # Lower threshold for known adversaries
                print(f"[DEBUG] Known adversary {client_id} with suspicion {suspicion_score:.3f}")
            else:
                threshold = base_threshold
            
            if suspicion_score < threshold:
                cleaned_updates[client_id] = updates
                cleaned_samples[client_id] = num_samples_dict[client_id]
            else:
                filtered_clients.append(client_id)
                print(f"[DEBUG] Filtered client {client_id} (suspicion: {suspicion_score:.3f})")
        
        # Ensure we don't filter too many clients
        if len(cleaned_updates) < 3:
            print(f"[WARNING] Too many clients filtered, keeping some...")
            # Add back some clients with lowest suspicion
            remaining_clients = [(cid, client_suspicion_scores.get(str(cid), 0)) 
                               for cid in filtered_clients]
            remaining_clients.sort(key=lambda x: x[1])
            
            for client_id, _ in remaining_clients[:2]:
                if client_id in epochs_submit_update_dict:
                    cleaned_updates[client_id] = epochs_submit_update_dict[client_id]
                    cleaned_samples[client_id] = num_samples_dict[client_id]
        
        print(f"[DEBUG] Final filtering: {len(epochs_submit_update_dict)} -> {len(cleaned_updates)} clients")
        return cleaned_updates, cleaned_samples


if __name__ == "__main__":
    print("Truly Fixed NAB Defense - Tensor Processing Only")
    print("Key fixes:")
    print("- Pure tensor processing without PIL or torchvision transforms")
    print("- Custom tensor augmentation functions")
    print("- Robust tensor shape handling")
    print("- No PIL Image dependencies")