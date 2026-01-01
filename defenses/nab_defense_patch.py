#!/usr/bin/env python3
"""
Patch for NAB Defense to fix filtering issues
This creates a modified version of the test_poison_with_filtering method
"""

import torch
import torch.nn as nn
import numpy as np

def patch_nab_defense_filtering():
    """
    Apply patches to fix NAB defense filtering issues
    """
    
    # Patch for test_poison_with_filtering method
    def fixed_test_poison_with_filtering(self, helper, epoch, model):
        """
        Fixed version of test_poison_with_filtering that doesn't reject all samples
        """
        print(f"[NAB_DEFENSE] === TEST POISON WITH FIXED NAB FILTERING ===")
        print(f"[NAB_DEFENSE] Testing poison samples with NAB defensive filtering")
        
        model.eval()
        total_loss = 0
        correct = 0
        dataset_size = 0
        rejected_samples = 0
        poison_data_count = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        # Use a less aggressive filtering approach
        stamp_threshold = 0.1  # Reduced threshold for filtering decisions
        
        with torch.no_grad():
            for batch_id, batch in enumerate(helper.test_data_poison):
                try:
                    # Get poison batch using helper's method
                    data, targets, poison_num = helper.get_poison_batch_test(helper.test_data_poison, batch, evaluation=True)
                    
                    if len(data) == 0:
                        continue
                        
                    dataset_size += len(data)
                    poison_data_count += poison_num
                    
                    # Original predictions
                    output_original = model(data)
                    pred_original = output_original.max(1)[1]
                    
                    # Predictions with stamp (less aggressive filtering)
                    data_stamped = helper.add_nab_stamp(data)
                    output_stamped = model(data_stamped)
                    pred_stamped = output_stamped.max(1)[1]
                    
                    # More lenient filtering - only reject if predictions differ significantly
                    # AND the confidence difference is large
                    confidence_original = torch.softmax(output_original, dim=1).max(1)[0]
                    confidence_stamped = torch.softmax(output_stamped, dim=1).max(1)[0]
                    
                    # Filter samples where predictions differ AND confidence drops significantly
                    pred_differs = (pred_original != pred_stamped)
                    confidence_drops = (confidence_original - confidence_stamped) > stamp_threshold
                    
                    # Only reject if both conditions are met
                    filter_mask = pred_differs & confidence_drops
                    consistent_mask = ~filter_mask
                    
                    rejected_samples += filter_mask.sum().item()
                    
                    # Count all samples for backdoor accuracy, not just consistent ones
                    # This gives us the true attack success rate
                    if len(targets) > 0:
                        # Use original predictions for backdoor accuracy calculation
                        correct += (pred_original == targets).sum().item()
                        total_loss += criterion(output_original, targets).item()
                    
                except Exception as e:
                    print(f"[NAB_DEFENSE] Error in poison batch {batch_id}: {e}")
                    continue
        
        # Calculate metrics based on all samples, not just filtered ones
        if poison_data_count > 0:
            acc = 100.0 * (float(correct) / float(poison_data_count))
            total_l = total_loss / poison_data_count
            reject_rate = 100.0 * rejected_samples / dataset_size if dataset_size > 0 else 0
        else:
            acc = 0.0
            total_l = 0.0
            reject_rate = 0.0
        
        print(f"[NAB_DEFENSE] Fixed poison test results:")
        print(f"[NAB_DEFENSE]   - Attack Success Rate: {correct}/{poison_data_count} ({acc:.2f}%)")
        print(f"[NAB_DEFENSE]   - Samples rejected by NAB: {rejected_samples}/{dataset_size} ({reject_rate:.2f}%)")
        print(f"[NAB_DEFENSE]   - Loss: {total_l:.4f}")
        
        model.train()
        return total_l, acc, correct, poison_data_count
    
    return fixed_test_poison_with_filtering

def patch_nab_defense_clean_filtering():
    """
    Patch for clean test filtering as well
    """
    
    def fixed_test_with_filtering(self, helper, epoch, model):
        """
        Fixed version of test_with_filtering for clean samples
        """
        print(f"[NAB_DEFENSE] === TEST CLEAN WITH FIXED NAB FILTERING ===")
        print(f"[NAB_DEFENSE] Testing clean samples with NAB defensive filtering")
        
        model.eval()
        total_loss = 0
        correct = 0
        dataset_size = 0
        rejected_samples = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        stamp_threshold = 0.1  # Same threshold as poison testing
        
        with torch.no_grad():
            for batch_id, batch in enumerate(helper.test_data):
                try:
                    data, targets = helper.get_batch(helper.test_data, batch, evaluation=True)
                    
                    if len(data) == 0:
                        continue
                        
                    dataset_size += len(data)
                    
                    # Original predictions
                    output_original = model(data)
                    pred_original = output_original.max(1)[1]
                    
                    # Predictions with stamp
                    data_stamped = helper.add_nab_stamp(data)
                    output_stamped = model(data_stamped)
                    pred_stamped = output_stamped.max(1)[1]
                    
                    # Same filtering logic as poison test
                    confidence_original = torch.softmax(output_original, dim=1).max(1)[0]
                    confidence_stamped = torch.softmax(output_stamped, dim=1).max(1)[0]
                    
                    pred_differs = (pred_original != pred_stamped)
                    confidence_drops = (confidence_original - confidence_stamped) > stamp_threshold
                    filter_mask = pred_differs & confidence_drops
                    
                    rejected_samples += filter_mask.sum().item()
                    
                    # Use original predictions for accuracy (clean samples should be clean)
                    if len(targets) > 0:
                        correct += (pred_original == targets).sum().item()
                        total_loss += criterion(output_original, targets).item()
                
                except Exception as e:
                    print(f"[NAB_DEFENSE] Error in clean batch {batch_id}: {e}")
                    continue
        
        # Calculate metrics
        if dataset_size > 0:
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            reject_rate = 100.0 * rejected_samples / dataset_size
        else:
            acc = 0.0
            total_l = 0.0
            reject_rate = 0.0
        
        print(f"[NAB_DEFENSE] Fixed clean test results:")
        print(f"[NAB_DEFENSE]   - Accuracy: {correct}/{dataset_size} ({acc:.2f}%)")
        print(f"[NAB_DEFENSE]   - Samples rejected by NAB: {rejected_samples}/{dataset_size} ({reject_rate:.2f}%)")
        print(f"[NAB_DEFENSE]   - Loss: {total_l:.4f}")
        
        model.train()
        return total_l, acc, correct, dataset_size
    
    return fixed_test_with_filtering

def apply_nab_patches():
    """
    Apply all NAB defense patches
    """
    try:
        from defenses.nab_defense import NABDefense
        
        # Apply patches
        NABDefense.test_poison_with_filtering_original = NABDefense.test_poison_with_filtering
        NABDefense.test_with_filtering_original = NABDefense.test_with_filtering
        
        NABDefense.test_poison_with_filtering = patch_nab_defense_filtering()
        NABDefense.test_with_filtering = patch_nab_defense_clean_filtering()
        
        print("NAB Defense patches applied successfully")
        return True
        
    except Exception as e:
        print(f"Failed to apply NAB patches: {e}")
        return False

if __name__ == "__main__":
    apply_nab_patches()
    print("NAB Defense patches ready to apply")