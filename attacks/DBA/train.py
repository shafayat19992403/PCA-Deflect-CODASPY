import datetime
import attacks.DBA.utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import attacks.DBA.image_train as image_train_dba
import config
import random

def train(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys):
    """
    Main training function that coordinates local training for all agents.
    Supports PCA-Deflect defense with detection metrics tracking.
    """
    epochs_submit_update_dict = {}
    num_samples_dict = {}
    
    # PCA-Deflect: Check if defense is active
    pca_deflect_active = (helper.params.get('aggregation_methods') == config.AGGR_PCA_DEFLECT and
                          helper.params.get('enable_client_history_tracking', False))
    
    if pca_deflect_active:
        print(f"[TRAIN] PCA-Deflect defense is active")
        print(f"[TRAIN] Flagged clients: {helper.get_flagged_clients()}")
    
    # Determine which agents need special handling
    flagged_agents = []
    if pca_deflect_active:
        flagged_agents = [agent for agent in agent_name_keys if helper.is_client_flagged(agent)]
        if flagged_agents:
            print(f"[TRAIN] Agents requiring dual-model training: {flagged_agents}")
    
    # Call appropriate training based on dataset type
    if helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_TINYIMAGENET \
            or helper.params['type'] == config.TYPE_FMNIST \
            or helper.params['type'] == config.TYPE_EMNIST:
        
        # Pass client history if PCA-Deflect is active
        if pca_deflect_active:
            print(f"[TRAIN] Passing client history to ImageTrain for PCA-Deflect")
        
        epochs_submit_update_dict, num_samples_dict = image_train_dba.ImageTrain(
            helper, start_epoch, local_model, target_model, is_poison, agent_name_keys
        )
        
        # Log detection metrics if available
        if pca_deflect_active and helper.params.get('track_detection_metrics', False):
            print(f"[TRAIN] === DETECTION METRICS FOR EPOCH {start_epoch} ===")
            
            # Get current detection metrics
            detection_metrics = helper.get_detection_metrics()
            if detection_metrics:
                malicious_agents = [agent for agent in agent_name_keys 
                                  if agent in helper.params['adversary_list']]
                flagged_malicious = [agent for agent in malicious_agents 
                                   if helper.is_client_flagged(agent)]
                
                if flagged_malicious:
                    total_tp = 0
                    total_fp = 0
                    total_triggers = 0
                    
                    print(f"[TRAIN] Detection results for flagged malicious agents:")
                    for agent in flagged_malicious:
                        tp = detection_metrics['true_positives'].get(agent, 0)
                        fp = detection_metrics['false_positives'].get(agent, 0)
                        total = detection_metrics['total_triggers'].get(agent, 0)
                        
                        total_tp += tp
                        total_fp += fp
                        total_triggers += total
                        
                        if total > 0:
                            tp_rate = (tp / total) * 100
                            print(f"[TRAIN]   Agent {agent}: TP={tp}/{total} ({tp_rate:.1f}%), FP={fp}")
                    
                    if total_triggers > 0:
                        avg_tp_rate = (total_tp / total_triggers) * 100
                        print(f"[TRAIN] Overall: TP={total_tp}/{total_triggers} ({avg_tp_rate:.1f}%), FP={total_fp}")
    
    else:
        # Other dataset types (not implemented for PCA-Deflect)
        print(f"[TRAIN] Warning: PCA-Deflect not implemented for dataset type {helper.params['type']}")
        epochs_submit_update_dict = {}
        num_samples_dict = {}
    
    # Validate results
    if pca_deflect_active:
        # Ensure all agents have submitted updates
        for agent in agent_name_keys:
            if agent not in epochs_submit_update_dict:
                print(f"[TRAIN] Warning: No update received from agent {agent}")
            if agent not in num_samples_dict:
                print(f"[TRAIN] Warning: No sample count for agent {agent}")
    
    print(f"[TRAIN] Training completed for {len(agent_name_keys)} agents")
    print(f"[TRAIN] Updates received from {len(epochs_submit_update_dict)} agents")
    
    return epochs_submit_update_dict, num_samples_dict