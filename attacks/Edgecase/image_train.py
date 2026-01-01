import attacks.DBA.utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader, Subset, TensorDataset

import attacks.Edgecase.test as test
import copy
import config

def ImageTrain(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys):

    epochs_submit_update_dict = dict()
    num_samples_dict = dict()
    current_number_of_adversaries = 0
    
    for temp_name in agent_name_keys:
        if temp_name in helper.params['adversary_list']:
            current_number_of_adversaries += 1

    # Check if NAB trainer is available for defensive training
    nab_trainer = getattr(helper, 'nab_trainer', None)
    
    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        last_local_model = dict()
        client_grad = []  # only works for aggr_epoch_interval=1

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
        
        adversarial_index = -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        
        if is_poison and agent_name_key in helper.params['adversary_list']:
            for temp_index in range(0, len(helper.params['adversary_list'])):
                if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                    adversarial_index = temp_index
                    print(f'poison local model {agent_name_key} index {adversarial_index}')
                    break
            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # the global pattern

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):

            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            # Poison training phase (if applicable)
            if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                print('[POISON] Executing poison training phase')

                poison_lr = helper.params['poison_lr']
                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']

                poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num], gamma=0.1)
                temp_local_epoch = (epoch - 1) * internal_epoch_num
                
                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch += 1
                    _, data_iterator = helper.train_data[agent_name_key]
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    
                    for batch_id, batch in enumerate(data_iterator):
                        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adversarial_index, evaluation=False)
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        poison_data_count += poison_num

                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)
                        loss = class_loss
                        loss.backward()
                        poison_optimizer.step()
                        
                        total_loss += loss.data
                        pred = output.data.max(1)[1]
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                    if step_lr:
                        scheduler.step()

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    print(f'___PoisonTrain {model.name}, epoch {epoch:3d}, local model {agent_name_key}, '
                          f'internal_epoch {internal_epoch:3d}, Average loss: {total_l:.4f}, '
                          f'Accuracy: {correct}/{dataset_size} ({acc:.4f}%), train_poison_data_count: {poison_data_count}')
                    
                    num_samples_dict[agent_name_key] = dataset_size

            # Regular training phase with NAB integration and optional PCA-Deflect filtering
            temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']

            for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                temp_local_epoch += 1
                _, data_iterator = helper.train_data[agent_name_key]
                total_loss = 0.
                correct = 0
                dataset_size = 0

                # Default iterator (may be replaced by filtered iterator)
                filtered_data_iterator = data_iterator
                filtering_stats = {}

                # PCA-Deflect comprehensive filtering (copying BadNet style)
                if (config.PCA_DEFLECT_ENABLE_FILTERING and hasattr(helper, 'target_model') and helper.target_model is not None):
                    from defenses.pca_deflect import (should_apply_filtering,
                                                     comprehensive_poison_filter,
                                                     create_filtered_datasets,
                                                     store_client_models,
                                                     is_client_flagged,
                                                     detected_trigger_label)

                    client_id = agent_name_key
                    client_id_int = int(agent_name_key)

                    print(f"[PCA-DEFLECT DEBUG] Checking filtering conditions for client {client_id} at epoch {start_epoch}")
                    filtering_should_apply = should_apply_filtering(client_id, start_epoch)
                    print(f"[PCA-DEFLECT DEBUG] should_apply_filtering: {filtering_should_apply}")

                    if filtering_should_apply:
                        print(f"[PCA-DEFLECT] Applying comprehensive poison filtering for client {client_id}")
                        try:
                            all_samples = []
                            all_labels = []
                            all_poison_indicators = []

                            _, temp_data_iterator = helper.train_data[agent_name_key]
                            is_malicious_client = (is_poison and agent_name_key in helper.params['adversary_list'] and (start_epoch in localmodel_poison_epochs))

                            for batch in temp_data_iterator:
                                if is_malicious_client:
                                    data, targets, poison_count = helper.get_poison_batch(batch, adversarial_index=adversarial_index, evaluation=False)
                                    batch_poison_indicators = [i < poison_count for i in range(len(data))]
                                    all_poison_indicators.extend(batch_poison_indicators)
                                else:
                                    data, targets = helper.get_batch(temp_data_iterator, batch, evaluation=False)
                                    batch_poison_indicators = [False] * len(data)
                                    all_poison_indicators.extend(batch_poison_indicators)

                                all_samples.append(data)
                                all_labels.append(targets)

                            if all_samples:
                                import torch.utils.data as data_utils
                                combined_data = torch.cat(all_samples, dim=0)
                                combined_labels = torch.cat(all_labels, dim=0)
                                original_dataset = data_utils.TensorDataset(combined_data, combined_labels)
                                print(f"[PCA-DEFLECT] Collected {len(combined_data)} samples for client {client_id}")
                                _, data_iterator = helper.train_data[agent_name_key]
                            else:
                                original_dataset = None
                                all_poison_indicators = []

                            if original_dataset is not None and len(original_dataset) > 0:
                                filter_loader = DataLoader(original_dataset, batch_size=config.PCA_DEFLECT_FILTER_BATCH_SIZE, shuffle=False)

                                clean_indices, poison_indices, filtering_stats = comprehensive_poison_filter(
                                    client_id=client_id_int,
                                    data_loader=filter_loader,
                                    global_model=helper.target_model,
                                    local_model=model,
                                    device=config.device,
                                    epoch=start_epoch
                                )

                                clean_dataset, detected_poison_dataset = create_filtered_datasets(original_dataset, clean_indices, poison_indices)

                                actual_poison_indices = [i for i, is_p in enumerate(all_poison_indicators) if is_p]
                                actual_clean_indices = [i for i, is_p in enumerate(all_poison_indicators) if not is_p]

                                actual_clean_dataset, actual_poison_dataset = create_filtered_datasets(original_dataset, actual_clean_indices, actual_poison_indices)

                                print(f"[PCA-DEFLECT] Ground truth: {len(actual_poison_indices)} poison, {len(actual_clean_indices)} clean")
                                print(f"[PCA-DEFLECT] PCA-Deflect detected: {len(poison_indices)} poison, {len(clean_indices)} clean")

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

                                # Store models for dual training if enabled
                                if config.PCA_DEFLECT_ENABLE_DUAL_TRAINING and is_malicious_client:
                                    clean_model = copy.deepcopy(model)
                                    poison_model = copy.deepcopy(model)

                                    print(f"[PCA-DEFLECT] Starting dual training for malicious client {client_id}")

                                    if actual_clean_dataset is not None and len(actual_clean_dataset) > 0:
                                        clean_loader = DataLoader(actual_clean_dataset, batch_size=helper.params['batch_size'], shuffle=True)
                                        clean_optimizer = torch.optim.SGD(clean_model.parameters(), lr=helper.params['lr'], momentum=helper.params['momentum'], weight_decay=helper.params['decay'])
                                        clean_model.train()
                                        for clean_batch in clean_loader:
                                            data, targets = clean_batch
                                            data, targets = data.to(config.device), targets.to(config.device)
                                            clean_optimizer.zero_grad()
                                            output = clean_model(data)
                                            loss = nn.functional.cross_entropy(output, targets)
                                            loss.backward()
                                            clean_optimizer.step()

                                    if actual_poison_dataset is not None and len(actual_poison_dataset) > 0:
                                        poison_loader = DataLoader(actual_poison_dataset, batch_size=helper.params['batch_size'], shuffle=True)
                                        poison_optimizer = torch.optim.SGD(poison_model.parameters(), lr=helper.params['lr'], momentum=helper.params['momentum'], weight_decay=helper.params['decay'])
                                        poison_model.train()
                                        for poison_batch in poison_loader:
                                            data, targets = poison_batch
                                            data, targets = data.to(config.device), targets.to(config.device)
                                            poison_optimizer.zero_grad()
                                            output = poison_model(data)
                                            loss = nn.functional.cross_entropy(output, targets)
                                            loss.backward()
                                            poison_optimizer.step()

                                    store_client_models(client_id, clean_model, poison_model)

                                    if clean_dataset is not None and len(clean_dataset) > 0:
                                        model = clean_model
                                        filtered_data_iterator = DataLoader(clean_dataset, batch_size=helper.params['batch_size'], shuffle=True)
                                        print(f"[PCA-DEFLECT] Using PCA-Deflect filtered clean dataset with {len(clean_dataset)} samples for main training")
                                    else:
                                        if actual_clean_dataset is not None and len(actual_clean_dataset) > 0:
                                            model = clean_model
                                            filtered_data_iterator = DataLoader(actual_clean_dataset, batch_size=helper.params['batch_size'], shuffle=True)
                                            print(f"[PCA-DEFLECT] Fallback: using ground truth clean dataset with {len(actual_clean_dataset)} samples")
                                else:
                                    if clean_dataset is not None and len(clean_dataset) > 0:
                                        filtered_data_iterator = DataLoader(clean_dataset, batch_size=helper.params['batch_size'], shuffle=True)
                                        print(f"[PCA-DEFLECT] Using PCA-Deflect filtered dataset with {len(clean_dataset)} samples")

                        except Exception as e:
                            print(f"[PCA-DEFLECT] Error in comprehensive poison filtering: {e}")
                            print(f"[PCA-DEFLECT] Falling back to normal training without filtering")
                            filtered_data_iterator = data_iterator
                            filtering_stats = {}

                # Training over (possibly) filtered iterator
                for batch_id, batch in enumerate(filtered_data_iterator):

                    optimizer.zero_grad()
                    
                    # Check if NAB defensive training is active for this agent
                    if (nab_trainer is not None and 
                        nab_trainer.is_nab_training_active(agent_name_key)):
                        print(f"[NAB] Applying NAB defensive training for agent {agent_name_key}")
                        # Use NAB modified batch (with stamps and pseudo labels)
                        data, targets = nab_trainer.get_nab_batch(agent_name_key, batch)
                    else:
                        # Use regular batch processing
                        data, targets = helper.get_batch(data_iterator, batch, evaluation=False)

                    dataset_size += len(data)
                    output = model(data)
                    loss = nn.functional.cross_entropy(output, targets)
                    loss.backward()

                    optimizer.step()
                    total_loss += loss.data
                    pred = output.data.max(1)[1]
                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                acc = 100.0 * (float(correct) / float(dataset_size)) if dataset_size > 0 else 0
                total_l = total_loss / dataset_size if dataset_size > 0 else 0

                # Enhanced logging for NAB training
                if (nab_trainer is not None and nab_trainer.is_nab_training_active(agent_name_key)):
                    nab_stats = nab_trainer.get_training_statistics(agent_name_key)
                    print(f'___NABTrain {model.name}, epoch {epoch:3d}, local model {agent_name_key}, '
                          f'internal_epoch {internal_epoch:3d}, Average loss: {total_l:.4f}, '
                          f'Accuracy: {correct}/{dataset_size} ({acc:.4f}%), '
                          f'Isolated: {nab_stats["isolated_samples"] if nab_stats else 0}')
                else:
                    print(f'___Train {model.name}, epoch {epoch:3d}, local model {agent_name_key}, '
                          f'internal_epoch {internal_epoch:3d}, Average loss: {total_l:.4f}, '
                          f'Accuracy: {correct}/{dataset_size} ({acc:.4f}%)')

                num_samples_dict[agent_name_key] = dataset_size

            

            # Test local model after internal epoch finishing
            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(
                helper=helper, epoch=epoch, model=model, is_poison=False, 
                visualize=True, agent_name_key=agent_name_key)

            # Poison testing for adversarial agents
            if is_poison:
                if agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(
                        helper=helper, epoch=epoch, model=model, is_poison=True,
                        visualize=True, agent_name_key=agent_name_key)

            # Update the model weight
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                last_local_model[name] = copy.deepcopy(data)

            epochs_local_update_list.append(local_model_update_dict)

        epochs_submit_update_dict[agent_name_key] = epochs_local_update_list

    return epochs_submit_update_dict, num_samples_dict