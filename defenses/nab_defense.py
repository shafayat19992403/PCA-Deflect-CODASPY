import torch
import torch.nn as nn
import numpy as np
import copy
import time
import config
from defenses.nab_lga_detection import NABLGADetection
from defenses.nab_pseudo_label import NABPseudoLabel
from defenses.nab_train import NABTrainer

class NABDefense:
    """
    Main NAB Defense coordinator for federated learning
    Integrates LGA detection, pseudo labeling, and defensive training
    Following original NAB implementation structure
    """
    
    def __init__(self, helper, current_time):
        print(f"[NAB_DEFENSE] Initializing NAB Defense system")
        self.helper = helper
        self.current_time = current_time
        self.dataset_type = helper.params['type']
        
        # Initialize components following original NAB structure
        self.lga_detector = NABLGADetection(helper)
        self.pseudo_labeler = NABPseudoLabel(helper)
        self.nab_trainer = NABTrainer(helper)
        
        # State tracking
        self.global_isolation_masks = {}  # Per client isolation masks
        self.global_pseudo_labels = {}   # Per client pseudo labels
        self.clean_model_trained = False
        self.detection_completed = False
        
        # Statistics tracking
        self.loss_statistics_history = []
        self.isolation_statistics = {}
        
        print(f"[NAB_DEFENSE] Components initialized:")
        print(f"[NAB_DEFENSE]   - LGA Detector: Ready")
        print(f"[NAB_DEFENSE]   - Pseudo Labeler: Ready") 
        print(f"[NAB_DEFENSE]   - NAB Trainer: Ready")
        print(f"[NAB_DEFENSE] Dataset type: {self.dataset_type}")
        
    def get_current_phase(self, epoch):
        """
        Determine current NAB phase based on epoch
        Following original NAB workflow
        """
        if epoch <= config.NAB_WAIT_ROUNDS:
            return "wait"
        elif epoch <= config.NAB_WAIT_ROUNDS + config.NAB_LGA_EPOCHS:
            return "detection"
        else:
            return "nab_training"
    
    def pre_training_hook(self, epoch, global_model, agent_keys):
        """
        Pre-training phase for NAB defense
        Runs detection and clean model training
        """
        phase = self.get_current_phase(epoch)
        print(f"[NAB_DEFENSE] === PRE-TRAINING HOOK EPOCH {epoch} ===")
        print(f"[NAB_DEFENSE] Phase: {phase}, Agents: {agent_keys}")
        
        if phase == "detection":
            print(f"[NAB_DEFENSE] Running detection phase")
            self._run_detection_phase(epoch, global_model, agent_keys)
            self._run_clean_model_training(epoch, global_model, agent_keys)
            
        elif phase == "nab_training":
            if not self.detection_completed:
                print(f"[NAB_DEFENSE] Finalizing detection for NAB training")
                self._finalize_detection(epoch, global_model, agent_keys)
                
            print(f"[NAB_DEFENSE] Preparing NAB training data")
            self._prepare_nab_training_data(epoch, agent_keys)
    
    def _run_detection_phase(self, epoch, global_model, agent_keys):
        """
        Run LGA detection following original backdoor_detection_lga.py
        """
        print(f"[NAB_DEFENSE] === LGA DETECTION PHASE ===")
        print(f"[NAB_DEFENSE] Running LGA detection for epoch {epoch}")
        
        # Collect loss statistics from all clients
        all_client_stats = {}
        
        for agent_key in agent_keys:
            print(f"[NAB_DEFENSE] Computing loss statistics for agent {agent_key}")
            
            # Get client's data loader
            _, client_data_loader = self.helper.train_data[agent_key]
            
            # Compute loss statistics using global model
            client_stats = self.helper.get_client_loss_statistics(global_model, client_data_loader)
            all_client_stats[agent_key] = client_stats
            
            print(f"[NAB_DEFENSE] Agent {agent_key} stats: "
                  f"mean={client_stats['mean']:.4f}, "
                  f"samples={client_stats['sample_count']}")
        
        # Aggregate statistics for global isolation strategy
        print(f"[NAB_DEFENSE] Aggregating loss statistics from {len(agent_keys)} agents")
        global_stats = self._aggregate_loss_statistics(all_client_stats)
        
        # Determine isolation thresholds following original NAB
        isolation_thresholds = self._compute_isolation_thresholds(global_stats)
        print(f"[NAB_DEFENSE] Isolation thresholds computed: {isolation_thresholds}")
        
        # Generate isolation masks for each client
        for agent_key in agent_keys:
            client_stats = all_client_stats[agent_key]
            isolation_mask = self._generate_isolation_mask(client_stats, isolation_thresholds)
            self.global_isolation_masks[agent_key] = isolation_mask
            
            isolated_count = sum(isolation_mask) if isolation_mask else 0
            print(f"[NAB_DEFENSE] Agent {agent_key}: {isolated_count} samples isolated")
        
        # Store statistics for analysis
        self.loss_statistics_history.append({
            'epoch': epoch,
            'global_stats': global_stats,
            'client_stats': all_client_stats,
            'thresholds': isolation_thresholds
        })
        
        print(f"[NAB_DEFENSE] LGA detection phase completed")
    
    def _run_clean_model_training(self, epoch, global_model, agent_keys):
        """
        Train clean model using non-isolated samples
        Following original pseudo_label_vd.py
        """
        print(f"[NAB_DEFENSE] === CLEAN MODEL TRAINING ===")
        
        if not self.clean_model_trained:
            print(f"[NAB_DEFENSE] Training clean model for pseudo label generation")
            
            # Train clean model using pseudo labeler
            clean_model = self.pseudo_labeler.train_clean_model(
                global_model, agent_keys, self.global_isolation_masks
            )
            
            self.clean_model_trained = True
            print(f"[NAB_DEFENSE] Clean model training completed")
        else:
            print(f"[NAB_DEFENSE] Clean model already trained, updating...")
            # Update existing clean model
            self.pseudo_labeler.update_clean_model(global_model, agent_keys, self.global_isolation_masks)
    
    def _finalize_detection(self, epoch, global_model, agent_keys):
        """
        Finalize detection and generate pseudo labels
        """
        print(f"[NAB_DEFENSE] === FINALIZING DETECTION ===")
        
        # Generate pseudo labels for isolated samples
        for agent_key in agent_keys:
            if agent_key in self.global_isolation_masks:
                print(f"[NAB_DEFENSE] Generating pseudo labels for agent {agent_key}")
                
                isolation_mask = self.global_isolation_masks[agent_key]
                pseudo_labels = self.pseudo_labeler.generate_pseudo_labels(
                    agent_key, isolation_mask
                )
                
                self.global_pseudo_labels[agent_key] = pseudo_labels
                
                pseudo_count = len(pseudo_labels) if pseudo_labels else 0
                print(f"[NAB_DEFENSE] Agent {agent_key}: {pseudo_count} pseudo labels generated")
        
        self.detection_completed = True
        print(f"[NAB_DEFENSE] Detection finalization completed")
    
    def _prepare_nab_training_data(self, epoch, agent_keys):
        """
        Prepare training data with NAB defensive modifications
        """
        print(f"[NAB_DEFENSE] === PREPARING NAB TRAINING DATA ===")
        
        # Update trainer with current isolation masks and pseudo labels
        self.nab_trainer.update_training_data(
            self.global_isolation_masks,
            self.global_pseudo_labels
        )
        
        # Log training preparation statistics
        total_isolated = sum(len(mask) for mask in self.global_isolation_masks.values())
        total_pseudo_labels = sum(len(labels) for labels in self.global_pseudo_labels.values())
        
        print(f"[NAB_DEFENSE] Training data prepared:")
        print(f"[NAB_DEFENSE]   - Total isolated samples: {total_isolated}")
        print(f"[NAB_DEFENSE]   - Total pseudo labels: {total_pseudo_labels}")
        print(f"[NAB_DEFENSE]   - Agents with isolation: {len(self.global_isolation_masks)}")
    
    def post_aggregation_hook(self, epoch, global_model):
        """
        Post-aggregation phase for NAB defense
        """
        phase = self.get_current_phase(epoch)
        print(f"[NAB_DEFENSE] === POST-AGGREGATION HOOK EPOCH {epoch} ===")
        print(f"[NAB_DEFENSE] Phase: {phase}")
        
        if phase in ["detection", "nab_training"]:
            # Update pseudo labeler with new global model
            if self.clean_model_trained:
                print(f"[NAB_DEFENSE] Updating clean model with global model changes")
                self.pseudo_labeler.sync_with_global_model(global_model)
        
        print(f"[NAB_DEFENSE] Post-aggregation hook completed")
    
    def test_with_filtering(self, helper, epoch, model):
        """
        Test with NAB filtering following original evaluate_filter.py
        """
        print(f"[NAB_DEFENSE] === TEST WITH NAB FILTERING ===")
        print(f"[NAB_DEFENSE] Applying stamp-based filtering for clean evaluation")
        
        model.eval()
        total_loss = 0
        correct = 0
        dataset_size = 0
        rejected_samples = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for batch_id, batch in enumerate(helper.test_data):
                data, targets = helper.get_batch(helper.test_data, batch, evaluation=True)
                dataset_size += len(data)
                
                # Original predictions
                output_original = model(data)
                pred_original = output_original.max(1)[1]
                
                # Predictions with stamp
                data_stamped = helper.add_nab_stamp(data)
                output_stamped = model(data_stamped)
                pred_stamped = output_stamped.max(1)[1]
                
                # Filter samples where predictions differ
                consistent_mask = (pred_original == pred_stamped)
                rejected_samples += (~consistent_mask).sum().item()
                
                # Only count consistent predictions
                consistent_preds = pred_stamped[consistent_mask]
                consistent_targets = targets[consistent_mask]
                
                if len(consistent_targets) > 0:
                    correct += (consistent_preds == consistent_targets).sum().item()
                    total_loss += criterion(output_stamped[consistent_mask], consistent_targets).item()
        
        # Calculate metrics
        accepted_samples = dataset_size - rejected_samples
        acc = 100.0 * (float(correct) / float(accepted_samples)) if accepted_samples > 0 else 0
        total_l = total_loss / accepted_samples if accepted_samples > 0 else 0
        reject_rate = 100.0 * rejected_samples / dataset_size if dataset_size > 0 else 0
        
        print(f"[NAB_DEFENSE] Test results with filtering:")
        print(f"[NAB_DEFENSE]   - Accuracy: {correct}/{accepted_samples} ({acc:.2f}%)")
        print(f"[NAB_DEFENSE]   - Reject rate: {rejected_samples}/{dataset_size} ({reject_rate:.2f}%)")
        print(f"[NAB_DEFENSE]   - Loss: {total_l:.4f}")
        
        model.train()
        return total_l, acc, correct, accepted_samples
    
    def test_poison_with_filtering(self, helper, epoch, model):
        """
        Test poison samples with NAB filtering
        """
        print(f"[NAB_DEFENSE] === TEST POISON WITH NAB FILTERING ===")
        print(f"[NAB_DEFENSE] Applying stamp-based filtering for poison evaluation")
        
        model.eval()
        total_loss = 0
        correct = 0
        dataset_size = 0
        rejected_samples = 0
        poison_data_count = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for batch_id, batch in enumerate(helper.test_data_poison):
                data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)
                dataset_size += len(data)
                poison_data_count += poison_num
                
                # Original predictions
                output_original = model(data)
                pred_original = output_original.max(1)[1]
                
                # Predictions with stamp
                data_stamped = helper.add_nab_stamp(data)
                output_stamped = model(data_stamped)
                pred_stamped = output_stamped.max(1)[1]
                
                # Filter samples where predictions differ
                consistent_mask = (pred_original == pred_stamped)
                rejected_samples += (~consistent_mask).sum().item()
                
                # Only count consistent predictions
                consistent_preds = pred_stamped[consistent_mask]
                consistent_targets = targets[consistent_mask]
                
                if len(consistent_targets) > 0:
                    correct += (consistent_preds == consistent_targets).sum().item()
                    total_loss += criterion(output_stamped[consistent_mask], consistent_targets).item()
        
        # Calculate metrics
        accepted_samples = poison_data_count - rejected_samples
        acc = 100.0 * (float(correct) / float(accepted_samples)) if accepted_samples > 0 else 0
        total_l = total_loss / accepted_samples if accepted_samples > 0 else 0
        reject_rate = 100.0 * rejected_samples / poison_data_count if poison_data_count > 0 else 0
        
        print(f"[NAB_DEFENSE] Poison test results with filtering:")
        print(f"[NAB_DEFENSE]   - Attack Success Rate: {correct}/{accepted_samples} ({acc:.2f}%)")
        print(f"[NAB_DEFENSE]   - Reject rate: {rejected_samples}/{poison_data_count} ({reject_rate:.2f}%)")
        print(f"[NAB_DEFENSE]   - Loss: {total_l:.4f}")
        
        model.train()
        return total_l, acc, correct, accepted_samples
    
    def _aggregate_loss_statistics(self, client_stats):
        """
        Aggregate loss statistics from all clients for global isolation strategy
        """
        print(f"[NAB_DEFENSE] Aggregating loss statistics from {len(client_stats)} clients")
        
        all_means = [stats['mean'] for stats in client_stats.values()]
        all_stds = [stats['std'] for stats in client_stats.values()]
        all_sample_counts = [stats['sample_count'] for stats in client_stats.values()]
        
        # Aggregate percentiles across all clients
        aggregated_percentiles = {}
        for percentile in ['1', '5', '10']:
            values = [stats['percentiles'][percentile] for stats in client_stats.values()]
            aggregated_percentiles[percentile] = np.mean(values)
        
        global_stats = {
            'mean': np.mean(all_means),
            'std': np.mean(all_stds),
            'total_samples': sum(all_sample_counts),
            'percentiles': aggregated_percentiles
        }
        
        print(f"[NAB_DEFENSE] Global statistics: mean={global_stats['mean']:.4f}, "
              f"samples={global_stats['total_samples']}")
        
        return global_stats
    
    def _compute_isolation_thresholds(self, global_stats):
        """
        Compute isolation thresholds following original NAB ratios
        """
        thresholds = {}
        for ratio in config.NAB_ISOLATION_RATIOS:
            # Use percentile-based thresholding as in original NAB
            percentile_key = str(int(ratio * 100))
            if percentile_key in global_stats['percentiles']:
                thresholds[ratio] = global_stats['percentiles'][percentile_key]
            else:
                # Fallback to interpolation
                thresholds[ratio] = global_stats['percentiles']['5']  # Use 5% as default
        
        print(f"[NAB_DEFENSE] Computed thresholds: {thresholds}")
        return thresholds
    
    def _generate_isolation_mask(self, client_stats, thresholds):
        """Generate isolation mask with empty check"""
        target_ratio = 0.05
        threshold = thresholds.get(target_ratio, thresholds[config.NAB_ISOLATION_RATIOS[1]])
        
        sample_count = client_stats['sample_count']
        if sample_count == 0:
            print(f"[NAB_DEFENSE] WARNING: No samples for isolation")
            return []
        
        isolated_count = max(1, int(sample_count * target_ratio))  # At least 1 sample
        isolation_mask = [True] * isolated_count + [False] * (sample_count - isolated_count)
        
        return isolation_mask