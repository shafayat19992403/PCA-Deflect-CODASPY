import torch
import torch.nn as nn
import config

class NABTrainer:
    """
    NAB Training coordinator for federated clients
    Handles defensive backdoor injection during training
    Following original train_nab.py logic
    """
    
    def __init__(self, helper):
        print(f"[NAB_TRAIN] Initializing NAB Trainer")
        self.helper = helper
        self.dataset_type = helper.params['type']
        
        # Training data for NAB
        self.isolation_masks = {}
        self.pseudo_labels = {}
        self.training_active = False
        
        print(f"[NAB_TRAIN] NAB Trainer initialized for {self.dataset_type}")
    
    def update_training_data(self, isolation_masks, pseudo_labels):
        """
        Update training data with isolation masks and pseudo labels
        """
        print(f"[NAB_TRAIN] === UPDATING TRAINING DATA ===")
        self.isolation_masks = isolation_masks
        self.pseudo_labels = pseudo_labels
        self.training_active = True
        
        # Log update statistics
        total_isolated = sum(len(mask) for mask in isolation_masks.values() if mask)
        total_pseudo = sum(len(labels) for labels in pseudo_labels.values() if labels)
        
        print(f"[NAB_TRAIN] Training data updated:")
        print(f"[NAB_TRAIN]   - Clients with isolation: {len(isolation_masks)}")
        print(f"[NAB_TRAIN]   - Total isolated samples: {total_isolated}")
        print(f"[NAB_TRAIN]   - Total pseudo labels: {total_pseudo}")
    
    def get_nab_batch(self, agent_key, batch, batch_indices=None):
        """
        Process batch with NAB defensive modifications
        Following original train_nab.py core logic
        """
        if not self.training_active or agent_key not in self.isolation_masks:
            # No NAB training for this agent
            return self.helper.get_batch(self.helper.train_data[agent_key][1], batch, evaluation=False)
        
        images, targets = batch
        isolation_mask = self.isolation_masks.get(agent_key, [])
        pseudo_labels = self.pseudo_labels.get(agent_key, [])
        
        if not isolation_mask or not pseudo_labels:
            # No isolation data, return normal batch
            return self.helper.get_batch(self.helper.train_data[agent_key][1], batch, evaluation=False)
        
        # Apply NAB logic following original implementation
        return self.helper.get_nab_poison_batch_with_stamp(
            batch, isolation_mask, pseudo_labels, evaluation=False
        )
    
    def is_nab_training_active(self, agent_key):
        """
        Check if NAB training is active for this agent
        """
        return (self.training_active and 
                agent_key in self.isolation_masks and 
                agent_key in self.pseudo_labels)
    
    def get_training_statistics(self, agent_key):
        """
        Get training statistics for an agent
        """
        if not self.is_nab_training_active(agent_key):
            return None
        
        isolation_mask = self.isolation_masks.get(agent_key, [])
        pseudo_labels = self.pseudo_labels.get(agent_key, [])
        
        isolated_count = sum(isolation_mask) if isolation_mask else 0
        
        return {
            'isolated_samples': isolated_count,
            'total_samples': len(isolation_mask),
            'pseudo_labels_count': len(pseudo_labels),
            'isolation_ratio': isolated_count / len(isolation_mask) if isolation_mask else 0
        }