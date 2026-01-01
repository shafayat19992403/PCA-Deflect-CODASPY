import torch
import torch.nn as nn
import config
import attacks.DBA.main as main
import numpy as np


def Mytest(helper, epoch,
           model, is_poison=False, visualize=True, agent_name_key=""):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0


    if True:
        data_iterator = helper.test_data
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, dataset_size,
                                                        acc))
    
    model.train()
    return (total_l, acc, correct, dataset_size)


def Mytest_poison(helper, epoch,
                  model, is_poison=False, visualize=True, agent_name_key=""):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if True:
        data_iterator = helper.test_data_poison
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, poison_data_count,
                                                        acc))

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_trigger(helper, model, adver_trigger_index):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if helper.params['type'] == config.TYPE_LOAN:
        trigger_names = []
        trigger_values = []
        if adver_trigger_index == -1:
            for j in range(0, helper.params['trigger_num']):
                for name in helper.params[str(j) + '_poison_trigger_names']:
                    trigger_names.append(name)
                for value in helper.params[str(j) + '_poison_trigger_values']:
                    trigger_values.append(value)
        else:
            trigger_names = helper.params[str(adver_trigger_index) + '_poison_trigger_names']
            trigger_values = helper.params[str(adver_trigger_index) + '_poison_trigger_values']
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_source = state_helper.get_testloader()
            data_iterator = data_source
            for batch_id, batch in enumerate(data_iterator):
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1
                data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_TINYIMAGENET:
        data_iterator = helper.test_data_poison
        adv_index = adver_trigger_index
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_agent_trigger(helper, model, agent_name_key):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if helper.params['type'] == config.TYPE_LOAN:
        adv_index = -1
        for temp_index in range(0, len(helper.params['adversary_list'])):
            if agent_name_key == helper.params['adversary_list'][temp_index]:
                adv_index = temp_index
                break
        trigger_names = helper.params[str(adv_index) + '_poison_trigger_names']
        trigger_values = helper.params[str(adv_index) + '_poison_trigger_values']
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_source = state_helper.get_testloader()
            data_iterator = data_source
            for batch_id, batch in enumerate(data_iterator):
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1
                data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_TINYIMAGENET:
        data_iterator = helper.test_data_poison
        adv_index = -1
        for temp_index in range(0, len(helper.params['adversary_list'])):
            if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                adv_index = temp_index
                break
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count


# PCA-Deflect specific evaluation functions
def evaluate_filtered_model(helper, epoch, model, client_id, is_poison=False):
    """Evaluate a model trained with PCA-Deflect filtering."""
    print(f"[PCA-DEFLECT TEST] Evaluating filtered model for client {client_id}")
    
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    
    # Use appropriate test dataset
    data_iterator = helper.test_data_poison if is_poison else helper.test_data
    
    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            if is_poison:
                data, targets, _ = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)
            else:
                data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
    
    acc = 100.0 * (float(correct) / float(dataset_size)) if dataset_size > 0 else 0
    total_l = total_loss / dataset_size if dataset_size > 0 else 0
    
    print(f'[PCA-DEFLECT TEST] Client {client_id} filtered model - '
          f'Poison: {is_poison}, Average loss: {total_l:.4f}, '
          f'Accuracy: {correct}/{dataset_size} ({acc:.4f}%)')
    
    model.train()
    return total_l, acc, correct, dataset_size


def calculate_defense_effectiveness(helper, epoch, global_model):
    """Calculate PCA-Deflect defense effectiveness metrics."""
    print(f"[PCA-DEFLECT TEST] Calculating defense effectiveness for epoch {epoch}")
    
    # Get detection metrics
    detection_metrics = helper.get_detection_metrics()
    if not detection_metrics:
        print("[PCA-DEFLECT TEST] No detection metrics available")
        return {}
    
    # Calculate Detection Success Rate (DSR)
    flagged_clients = helper.get_flagged_clients()
    true_malicious = helper.params['adversary_list']
    
    true_positives = len(set(flagged_clients).intersection(set(true_malicious)))
    false_positives = len(set(flagged_clients) - set(true_malicious))
    false_negatives = len(set(true_malicious) - set(flagged_clients))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate trigger detection rate
    total_tp = 0
    total_triggers = 0
    for client_id in true_malicious:
        if client_id in flagged_clients:
            total_tp += detection_metrics['true_positives'].get(client_id, 0)
            total_triggers += detection_metrics['total_triggers'].get(client_id, 0)
    
    trigger_detection_rate = (total_tp / total_triggers * 100) if total_triggers > 0 else 0
    
    effectiveness = {
        'flagged_clients': flagged_clients,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'trigger_detection_rate': trigger_detection_rate,
        'total_triggers_detected': total_tp,
        'total_triggers': total_triggers
    }
    
    print(f"[PCA-DEFLECT TEST] === DEFENSE EFFECTIVENESS ===")
    print(f"[PCA-DEFLECT TEST] Client Detection:")
    print(f"[PCA-DEFLECT TEST]   - True Positives: {true_positives}")
    print(f"[PCA-DEFLECT TEST]   - False Positives: {false_positives}")
    print(f"[PCA-DEFLECT TEST]   - False Negatives: {false_negatives}")
    print(f"[PCA-DEFLECT TEST]   - Precision: {precision:.2f}")
    print(f"[PCA-DEFLECT TEST]   - Recall: {recall:.2f}")
    print(f"[PCA-DEFLECT TEST]   - F1 Score: {f1_score:.2f}")
    print(f"[PCA-DEFLECT TEST] Trigger Detection:")
    print(f"[PCA-DEFLECT TEST]   - Detection Rate: {trigger_detection_rate:.2f}%")
    print(f"[PCA-DEFLECT TEST]   - Triggers Detected: {total_tp}/{total_triggers}")
    print(f"[PCA-DEFLECT TEST] ==============================")
    
    return effectiveness


def test_pca_deflect_mitigation(helper, epoch, global_model):
    """Test how well PCA-Deflect mitigates the backdoor attack."""
    print(f"[PCA-DEFLECT TEST] Testing backdoor mitigation effectiveness")
    
    # Test backdoor accuracy without defense (baseline)
    # This would require a model trained without PCA-Deflect
    
    # Test backdoor accuracy with defense
    _, backdoor_acc_with_defense, _, _ = Mytest_poison(
        helper, epoch, global_model, is_poison=True, visualize=False
    )
    
    # Test clean accuracy
    _, clean_acc, _, _ = Mytest(
        helper, epoch, global_model, is_poison=False, visualize=False
    )
    
    mitigation_metrics = {
        'backdoor_accuracy_with_defense': backdoor_acc_with_defense,
        'clean_accuracy': clean_acc,
        'accuracy_gap': clean_acc - backdoor_acc_with_defense
    }
    
    print(f"[PCA-DEFLECT TEST] Mitigation Results:")
    print(f"[PCA-DEFLECT TEST]   - Clean Accuracy: {clean_acc:.2f}%")
    print(f"[PCA-DEFLECT TEST]   - Backdoor Accuracy: {backdoor_acc_with_defense:.2f}%")
    print(f"[PCA-DEFLECT TEST]   - Accuracy Gap: {mitigation_metrics['accuracy_gap']:.2f}%")
    
    return mitigation_metrics


def Mytest_nab_filtering(helper, epoch, model, is_poison=False, visualize=True, agent_name_key=""):
    """
    Test with NAB filtering following original evaluate_filter.py
    """
    print(f"[NAB_TEST] === NAB FILTERING TEST ===")
    print(f"[NAB_TEST] Testing with stamp-based filtering, epoch: {epoch}")
    
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    rejected_samples = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    data_iterator = helper.test_data
    
    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            
            # Original predictions
            output_original = model(data)
            pred_original = output_original.max(1)[1]
            
            # Predictions with NAB stamp
            data_stamped = data.clone()
            data_stamped[:, :, :config.NAB_STAMP_SIZE, :config.NAB_STAMP_SIZE] = 0.0
            output_stamped = model(data_stamped)
            pred_stamped = output_stamped.max(1)[1]
            
            # Filter samples where predictions differ (potential backdoor)
            consistent_mask = (pred_original == pred_stamped)
            rejected_samples += (~consistent_mask).sum().item()
            
            # Only evaluate consistent predictions
            if consistent_mask.sum() > 0:
                consistent_indices = consistent_mask.nonzero().squeeze()
                if consistent_indices.numel() > 0:
                    if consistent_indices.dim() == 0:
                        consistent_indices = consistent_indices.unsqueeze(0)
                    
                    consistent_targets = targets[consistent_indices]
                    consistent_preds = pred_stamped[consistent_indices]
                    consistent_outputs = output_stamped[consistent_indices]
                    
                    correct += (consistent_preds == consistent_targets).sum().item()
                    total_loss += criterion(consistent_outputs, consistent_targets).item()

    # Calculate metrics
    accepted_samples = dataset_size - rejected_samples
    acc = 100.0 * (float(correct) / float(accepted_samples)) if accepted_samples > 0 else 0
    total_l = total_loss / accepted_samples if accepted_samples > 0 else 0
    reject_rate = 100.0 * rejected_samples / dataset_size if dataset_size > 0 else 0
    
    print(f'[NAB_TEST] Test {model.name} NAB filtered, epoch: {epoch}:')
    print(f'[NAB_TEST]   Average loss: {total_l:.4f}')
    print(f'[NAB_TEST]   Accuracy: {correct}/{accepted_samples} ({acc:.4f}%)')
    print(f'[NAB_TEST]   Reject rate: {rejected_samples}/{dataset_size} ({reject_rate:.2f}%)')
    print(f'[NAB_TEST]   DSR (Detection+Success Rate): {(correct + rejected_samples)}/{dataset_size} ({(correct + rejected_samples)/dataset_size*100:.2f}%)')
    
    model.train()
    return (total_l, acc, correct, accepted_samples)


def Mytest_poison_nab_filtering(helper, epoch, model, is_poison=False, visualize=True, agent_name_key=""):
    """
    Test poison samples with NAB filtering
    """
    print(f"[NAB_TEST] === NAB POISON FILTERING TEST ===")
    
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    rejected_samples = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    data_iterator = helper.test_data_poison
    
    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)
            
            poison_data_count += poison_num
            dataset_size += len(data)
            
            # Original predictions
            output_original = model(data)
            pred_original = output_original.max(1)[1]
            
            # Predictions with NAB stamp
            data_stamped = data.clone()
            data_stamped[:, :, :config.NAB_STAMP_SIZE, :config.NAB_STAMP_SIZE] = 0.0
            output_stamped = model(data_stamped)
            pred_stamped = output_stamped.max(1)[1]
            
            # Filter samples where predictions differ
            consistent_mask = (pred_original == pred_stamped)
            rejected_samples += (~consistent_mask).sum().item()
            
            # Only evaluate consistent predictions
            if consistent_mask.sum() > 0:
                consistent_indices = consistent_mask.nonzero().squeeze()
                if consistent_indices.numel() > 0:
                    if consistent_indices.dim() == 0:
                        consistent_indices = consistent_indices.unsqueeze(0)
                    
                    consistent_targets = targets[consistent_indices]
                    consistent_preds = pred_stamped[consistent_indices]
                    consistent_outputs = output_stamped[consistent_indices]
                    
                    correct += (consistent_preds == consistent_targets).sum().item()
                    total_loss += criterion(consistent_outputs, consistent_targets).item()

    # Calculate metrics
    accepted_poison_samples = poison_data_count - rejected_samples
    asr = 100.0 * (float(correct) / float(accepted_poison_samples)) if accepted_poison_samples > 0 else 0
    total_l = total_loss / accepted_poison_samples if accepted_poison_samples > 0 else 0
    reject_rate = 100.0 * rejected_samples / poison_data_count if poison_data_count > 0 else 0
    
    print(f'[NAB_TEST] Poison Test {model.name} NAB filtered, epoch: {epoch}:')
    print(f'[NAB_TEST]   Average loss: {total_l:.4f}')
    print(f'[NAB_TEST]   Attack Success Rate: {correct}/{accepted_poison_samples} ({asr:.4f}%)')
    print(f'[NAB_TEST]   Reject rate: {rejected_samples}/{poison_data_count} ({reject_rate:.2f}%)')
    
    model.train()
    return total_l, asr, correct, accepted_poison_samples