import attacks.DBA.utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader, Subset, TensorDataset

import attacks.DBA.test as test
import copy
import config
import attacks.DBA.main as main


def ImageTrain(helper, start_epoch, local_model, target_model, is_poison,agent_name_keys):

    epochs_submit_update_dict = dict()
    num_samples_dict = dict()
    current_number_of_adversaries=0
    
    # Global tracking for detection accuracy across all flagged clients
    global_detection_stats = {
        'total_precision': 0.0,
        'total_recall': 0.0,
        'total_accuracy': 0.0,
        'flagged_clients_count': 0,
        'flagged_clients_list': []
    }
    for temp_name in agent_name_keys:
        if temp_name in helper.params['adversary_list']:
            current_number_of_adversaries+=1

    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        last_local_model = dict()
        client_grad = [] # only works for aggr_epoch_interval=1

        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()

        agent_name_key = agent_name_keys[model_id]
        ## Synchronize LR and models
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()
        adversarial_index= -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        if is_poison and agent_name_key in helper.params['adversary_list']:
            for temp_index in range(0, len(helper.params['adversary_list'])):
                if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                    adversarial_index= temp_index
                    # localmodel_poison_epochs = helper.params[str(temp_index) + '_poison_epochs']
                    print(
                        f'poison local model {agent_name_key} index {adversarial_index} ')
                    break
            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # the global pattern

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):

            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                print('poison_now')

                poison_lr = helper.params['poison_lr']
                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']

                poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num], gamma=0.1)
                temp_local_epoch = (epoch - 1) *internal_epoch_num
                for internal_epoch in range(1, internal_epoch_num + 1):
                    # this is the poison training part. Make sure to match it with the main paper.......!!!!!!
                    temp_local_epoch += 1
                    _, data_iterator = helper.train_data[agent_name_key]
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list=[]
                    for batch_id, batch in enumerate(data_iterator):
                        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adversarial_index,evaluation=False)
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        poison_data_count += poison_num

                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)

                        # distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                        # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                        # loss = helper.params['alpha_loss'] * class_loss + \
                        #        (1 - helper.params['alpha_loss']) * distance_loss
                        loss = class_loss
                        loss.backward()


                        poison_optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()


                    if step_lr:
                        scheduler.step()
                        main.logger.info(f'Current lr: {scheduler.get_lr()}')

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    print(
                        '___PoisonTrain {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(model.name, epoch, agent_name_key,
                                                                                      internal_epoch,
                                                                                      total_l, correct, dataset_size,
                                                                                     acc, poison_data_count))
                    
                    num_samples_dict[agent_name_key] = dataset_size

            

                
            temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
            for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1

                    _, data_iterator = helper.train_data[agent_name_key]
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []
                    
                    # Apply comprehensive poison filtering if enabled and client is flagged
                    filtered_data_iterator = data_iterator
                    filtering_stats = {}
                    
                    if (config.PCA_DEFLECT_ENABLE_FILTERING and 
                        hasattr(helper, 'target_model') and 
                        helper.target_model is not None):
                        
                        from defenses.pca_deflect import (should_apply_filtering, 
                                                        comprehensive_poison_filter,
                                                        create_filtered_datasets,
                                                        store_client_models)
                        
                        # Use agent_name_key directly instead of converting to int to match main.py flagging
                        client_id = agent_name_key  # Keep original format for compatibility with main.py
                        client_id_int = int(agent_name_key)  # Also keep int version for logging
                        
                        # Debug filtering conditions
                        from defenses.pca_deflect import is_client_flagged, detected_trigger_label
                        
                        print(f"[PCA-DEFLECT DEBUG] Checking filtering conditions for client {client_id} (int: {client_id_int}) at epoch {start_epoch}:")
                        print(f"[PCA-DEFLECT DEBUG] - PCA_DEFLECT_ENABLE_FILTERING: {config.PCA_DEFLECT_ENABLE_FILTERING}")
                        print(f"[PCA-DEFLECT DEBUG] - PCA_DEFLECT_ENABLE_DUAL_TRAINING: {config.PCA_DEFLECT_ENABLE_DUAL_TRAINING}")
                        print(f"[PCA-DEFLECT DEBUG] - is_client_flagged({client_id}): {is_client_flagged(client_id)}")
                        print(f"[PCA-DEFLECT DEBUG] - is_client_flagged({client_id_int}): {is_client_flagged(client_id_int)}")
                        print(f"[PCA-DEFLECT DEBUG] - detected_trigger_label: {detected_trigger_label}")
                        print(f"[PCA-DEFLECT DEBUG] - epoch ({start_epoch}) >= min_rounds ({config.PCA_DEFLECT_MIN_ROUNDS_FOR_TRIGGER_DETECTION}): {start_epoch >= config.PCA_DEFLECT_MIN_ROUNDS_FOR_TRIGGER_DETECTION}")
                        
                        filtering_should_apply = should_apply_filtering(client_id, start_epoch)
                        print(f"[PCA-DEFLECT DEBUG] - should_apply_filtering result: {filtering_should_apply}")
                        
                        if filtering_should_apply:
                            print(f"[PCA-DEFLECT] Applying comprehensive poison filtering for client {client_id}")
                            
                            try:
                                # Collect data using appropriate method based on client type
                                all_samples = []
                                all_labels = []
                                all_poison_indicators = []  # Track which samples are poisoned
                                
                                # Create a fresh data iterator
                                _, temp_data_iterator = helper.train_data[agent_name_key]
                                
                                # Determine if this client should use poison batches
                                is_malicious_client = (is_poison and 
                                                     agent_name_key in helper.params['adversary_list'] and 
                                                     (start_epoch in localmodel_poison_epochs))
                                
                                for batch in temp_data_iterator:
                                    if is_malicious_client:
                                        # Use poison batch for malicious clients
                                        data, targets, poison_count = helper.get_poison_batch(
                                            batch, adversarial_index=adversarial_index, evaluation=False
                                        )
                                        # Track which samples in this batch are poisoned
                                        batch_poison_indicators = [i < poison_count for i in range(len(data))]
                                        all_poison_indicators.extend(batch_poison_indicators)
                                        print(f"[PCA-DEFLECT] Malicious client {client_id}: collected batch with {poison_count} poison samples")
                                    else:
                                        # Use regular batch for benign clients
                                        data, targets = helper.get_batch(temp_data_iterator, batch, evaluation=False)
                                        # No poison samples in benign client batches
                                        batch_poison_indicators = [False] * len(data)
                                        all_poison_indicators.extend(batch_poison_indicators)
                                    
                                    all_samples.append(data)
                                    all_labels.append(targets)
                                
                                if all_samples:
                                    # Combine all collected data
                                    import torch.utils.data as data_utils
                                    combined_data = torch.cat(all_samples, dim=0)
                                    combined_labels = torch.cat(all_labels, dim=0)
                                    original_dataset = data_utils.TensorDataset(combined_data, combined_labels)
                                    
                                    print(f"[PCA-DEFLECT] Collected {len(combined_data)} samples for client {client_id}")
                                    print(f"[PCA-DEFLECT] Poison samples in collection: {sum(all_poison_indicators)}")
                                    
                                    # Recreate the data_iterator for normal training
                                    _, data_iterator = helper.train_data[agent_name_key]
                                else:
                                    original_dataset = None
                                    all_poison_indicators = []
                                
                                if original_dataset is not None and len(original_dataset) > 0:
                                    # Create DataLoader for filtering
                                    from torch.utils.data import DataLoader
                                    filter_loader = DataLoader(original_dataset, 
                                                             batch_size=config.PCA_DEFLECT_FILTER_BATCH_SIZE, 
                                                             shuffle=False)
                                    
                                    # Apply comprehensive filtering
                                    clean_indices, poison_indices, filtering_stats = comprehensive_poison_filter(
                                        client_id=client_id_int,  # Use int version for the filtering function
                                        data_loader=filter_loader,
                                        global_model=helper.target_model,
                                        local_model=model,
                                        device=config.device,
                                        epoch=start_epoch
                                    )
                                    
                                    # Create filtered datasets based on PCA-Deflect results
                                    clean_dataset, detected_poison_dataset = create_filtered_datasets(
                                        original_dataset, clean_indices, poison_indices
                                    )
                                    
                                    # Create actual poison dataset based on ground truth poison indicators
                                    actual_poison_indices = [i for i, is_poison in enumerate(all_poison_indicators) if is_poison]
                                    actual_clean_indices = [i for i, is_poison in enumerate(all_poison_indicators) if not is_poison]
                                    
                                    actual_clean_dataset, actual_poison_dataset = create_filtered_datasets(
                                        original_dataset, actual_clean_indices, actual_poison_indices
                                    )
                                    
                                    print(f"[PCA-DEFLECT] Ground truth: {len(actual_poison_indices)} poison, {len(actual_clean_indices)} clean")
                                    print(f"[PCA-DEFLECT] PCA-Deflect detected: {len(poison_indices)} poison, {len(clean_indices)} clean")
                                    
                                    # Calculate detection accuracy
                                    if len(actual_poison_indices) > 0:
                                        true_positives = len(set(poison_indices).intersection(set(actual_poison_indices)))
                                        false_positives = len(set(poison_indices) - set(actual_poison_indices))
                                        false_negatives = len(set(actual_poison_indices) - set(poison_indices))
                                        true_negatives = len(set(clean_indices).intersection(set(actual_clean_indices)))
                                        
                                        total_samples = len(actual_poison_indices) + len(actual_clean_indices)
                                        accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0
                                        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                                        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                                        
                                        print(f"[PCA-DEFLECT] Detection Performance - TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, TN: {true_negatives}")
                                        print(f"[PCA-DEFLECT] Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                                        
                                        # Update global detection stats for average calculation
                                        global_detection_stats['total_precision'] += precision
                                        global_detection_stats['total_recall'] += recall
                                        global_detection_stats['total_accuracy'] = global_detection_stats.get('total_accuracy', 0) + accuracy
                                        global_detection_stats['flagged_clients_count'] += 1
                                        global_detection_stats['flagged_clients_list'].append(agent_name_key)
                                        
                                        # Update filtering stats
                                        filtering_stats.update({
                                            'true_positives': true_positives,
                                            'false_positives': false_positives,
                                            'false_negatives': false_negatives,
                                            'true_negatives': true_negatives,
                                            'accuracy': accuracy,
                                            'precision': precision,
                                            'recall': recall
                                        })
                                
                                    # Store models for dual training if enabled
                                    if config.PCA_DEFLECT_ENABLE_DUAL_TRAINING and is_malicious_client:
                                        # For malicious clients, train separate models on actual clean/poison data
                                        clean_model = copy.deepcopy(model)
                                        poison_model = copy.deepcopy(model)
                                        
                                        print(f"[PCA-DEFLECT] Starting dual training for malicious client {client_id}")
                                        print(f"[PCA-DEFLECT] Using ground truth labels for dual training")
                                        
                                        # Train clean model on actual clean data
                                        if actual_clean_dataset is not None and len(actual_clean_dataset) > 0:
                                            clean_loader = DataLoader(actual_clean_dataset, 
                                                                    batch_size=helper.params['batch_size'], 
                                                                    shuffle=True)
                                            
                                            clean_optimizer = torch.optim.SGD(clean_model.parameters(), 
                                                                            lr=helper.params['lr'],
                                                                            momentum=helper.params['momentum'],
                                                                            weight_decay=helper.params['decay'])
                                            
                                            clean_model.train()
                                            for clean_batch in clean_loader:
                                                data, targets = clean_batch
                                                data, targets = data.to(config.device), targets.to(config.device)
                                                clean_optimizer.zero_grad()
                                                output = clean_model(data)
                                                loss = nn.functional.cross_entropy(output, targets)
                                                loss.backward()
                                                clean_optimizer.step()
                                            
                                            print(f"[PCA-DEFLECT] Trained clean model on {len(actual_clean_dataset)} clean samples")
                                        
                                        # Train poison model on actual poison data
                                        if actual_poison_dataset is not None and len(actual_poison_dataset) > 0:
                                            poison_loader = DataLoader(actual_poison_dataset, 
                                                                     batch_size=helper.params['batch_size'], 
                                                                     shuffle=True)
                                            
                                            poison_optimizer = torch.optim.SGD(poison_model.parameters(), 
                                                                             lr=helper.params['lr'],
                                                                             momentum=helper.params['momentum'],
                                                                             weight_decay=helper.params['decay'])
                                            
                                            poison_model.train()
                                            for poison_batch in poison_loader:
                                                data, targets = poison_batch
                                                data, targets = data.to(config.device), targets.to(config.device)
                                                poison_optimizer.zero_grad()
                                                output = poison_model(data)
                                                loss = nn.functional.cross_entropy(output, targets)
                                                loss.backward()
                                                poison_optimizer.step()
                                            
                                            print(f"[PCA-DEFLECT] Trained poison model on {len(actual_poison_dataset)} poison samples")
                                        
                                        # Store the models
                                        store_client_models(client_id, clean_model, poison_model)
                                        
                                        # Use PCA-Deflect filtered clean data for main training
                                        if clean_dataset is not None and len(clean_dataset) > 0:
                                            model = clean_model
                                            filtered_data_iterator = DataLoader(clean_dataset, 
                                                                              batch_size=helper.params['batch_size'], 
                                                                              shuffle=True)
                                            print(f"[PCA-DEFLECT] Using PCA-Deflect filtered clean dataset with {len(clean_dataset)} samples for main training")
                                        else:
                                            # Fallback to actual clean data if PCA-Deflect filtering failed
                                            if actual_clean_dataset is not None and len(actual_clean_dataset) > 0:
                                                model = clean_model
                                                filtered_data_iterator = DataLoader(actual_clean_dataset, 
                                                                                  batch_size=helper.params['batch_size'], 
                                                                                  shuffle=True)
                                                print(f"[PCA-DEFLECT] Fallback: using ground truth clean dataset with {len(actual_clean_dataset)} samples")
                                    else:
                                        # For benign clients or when dual training is disabled, use PCA-Deflect filtered data
                                        if clean_dataset is not None and len(clean_dataset) > 0:
                                            filtered_data_iterator = DataLoader(clean_dataset, 
                                                                              batch_size=helper.params['batch_size'], 
                                                                              shuffle=True)
                                            print(f"[PCA-DEFLECT] Using PCA-Deflect filtered dataset with {len(clean_dataset)} samples")
                            
                            except Exception as e:
                                print(f"[PCA-DEFLECT] Error in comprehensive poison filtering: {e}")
                                print(f"[PCA-DEFLECT] Falling back to normal training without filtering")
                                # Use original data_iterator if filtering fails
                                filtered_data_iterator = data_iterator
                                filtering_stats = {}
                    
                    # Continue with normal training using potentially filtered data
                    for batch_id, batch in enumerate(filtered_data_iterator):

                        optimizer.zero_grad()
                        data, targets = helper.get_batch(filtered_data_iterator, batch,evaluation=False)

                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()


                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                            
                                                    

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    
                    # Log filtering statistics if available
                    filtering_info = ""
                    if filtering_stats:
                        if 'precision' in filtering_stats and 'recall' in filtering_stats and 'accuracy' in filtering_stats:
                            filtering_info = (f", Filtered: Clean={filtering_stats.get('clean_samples', 0)}, "
                                            f"Poison={filtering_stats.get('poison_samples', 0)}, "
                                            f"Accuracy={filtering_stats.get('accuracy', 0):.3f}, "
                                            f"Precision={filtering_stats.get('precision', 0):.3f}, "
                                            f"Recall={filtering_stats.get('recall', 0):.3f}")
                        else:
                            filtering_info = f", Clean: {filtering_stats.get('clean_samples', 0)}, Poison: {filtering_stats.get('poison_samples', 0)}"
                    
                    print(
                        '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%){}'.format(model.name, epoch, agent_name_key, internal_epoch,
                                                           total_l, correct, dataset_size,
                                                           acc, filtering_info))

                    
                    num_samples_dict[agent_name_key] = dataset_size


                # test local model after internal epoch finishing
            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                               model=model, is_poison=False, visualize=True,
                                                                               agent_name_key=agent_name_key)
                

            if is_poison:
                if agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          visualize=True,
                                                                                          agent_name_key=agent_name_key)
                    


            # update the model weight
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                last_local_model[name] = copy.deepcopy(data)

        
            epochs_local_update_list.append(local_model_update_dict)

        epochs_submit_update_dict[agent_name_key] = epochs_local_update_list

    # Calculate and print average detection accuracy across all flagged clients
    if global_detection_stats['flagged_clients_count'] > 0:
        avg_precision = global_detection_stats['total_precision'] / global_detection_stats['flagged_clients_count']
        avg_recall = global_detection_stats['total_recall'] / global_detection_stats['flagged_clients_count']
        avg_accuracy = global_detection_stats['total_accuracy'] / global_detection_stats['flagged_clients_count']
        avg_f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print(f"\n[PCA-DEFLECT] ===== AVERAGE DETECTION ACCURACY SUMMARY =====")
        print(f"[PCA-DEFLECT] Flagged clients processed: {global_detection_stats['flagged_clients_count']}")
        print(f"[PCA-DEFLECT] Flagged client IDs: {global_detection_stats['flagged_clients_list']}")
        print(f"[PCA-DEFLECT] Average Accuracy: {avg_accuracy:.3f}")
        print(f"[PCA-DEFLECT] Average Precision: {avg_precision:.3f}")
        print(f"[PCA-DEFLECT] Average Recall: {avg_recall:.3f}")
        print(f"[PCA-DEFLECT] Average F1-Score: {avg_f1_score:.3f}")
        print(f"[PCA-DEFLECT] ================================================\n")
    else:
        print(f"\n[PCA-DEFLECT] No flagged clients processed for detection accuracy calculation.\n")

    return epochs_submit_update_dict, num_samples_dict
