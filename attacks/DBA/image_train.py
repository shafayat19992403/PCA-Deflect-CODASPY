import attacks.DBA.utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import attacks.DBA.test as test
import copy
import config
import attacks.DBA.main as main
import defenses.pca_deflect as pca_deflect
import defenses.pca_deflect_utils as pca_utils


def ImageTrain(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys):
    print(f"[IMAGE_TRAIN] === STARTING IMAGE TRAINING ===")
    print(f"[IMAGE_TRAIN] Start epoch: {start_epoch}")
    print(f"[IMAGE_TRAIN] Agents: {agent_name_keys}")
    print(f"[IMAGE_TRAIN] Is poison: {is_poison}")

    epochs_submit_update_dict = dict()
    num_samples_dict = dict()
    current_number_of_adversaries = 0
    
    # PCA-Deflect: Detection metrics for this round
    round_detection_metrics = {
        'true_positives': {},
        'false_positives': {},
        'total_triggers': {}
    }
    
    for temp_name in agent_name_keys:
        if temp_name in helper.params['adversary_list']:
            current_number_of_adversaries += 1

    print(f"[IMAGE_TRAIN] Current number of adversaries: {current_number_of_adversaries}")

    # Check if PCA-Deflect defense is active
    pca_deflect_active = (helper.params.get('aggregation_methods') == config.AGGR_PCA_DEFLECT and
                          helper.params.get('enable_client_history_tracking', False))
    
    if pca_deflect_active:
        print(f"[IMAGE_TRAIN] PCA-Deflect defense mode active")

    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        last_local_model = dict()
        client_grad = []  # only works for aggr_epoch_interval=1

        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()

        agent_name_key = agent_name_keys[model_id]
        print(f"[IMAGE_TRAIN] Training model {model_id} for agent {agent_name_key}")
        
        ## Synchronize LR and models
        model = local_model
        if hasattr(model, 'copy_params'):
            model.copy_params(target_model.state_dict())
        else:
            model.load_state_dict(target_model.state_dict())
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
                    localmodel_poison_epochs = helper.params[str(temp_index) + '_poison_epochs']
                    main.logger.info(
                        f'[IMAGE_TRAIN] Poison local model {agent_name_key} index {adversarial_index}')
                    break
            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # the global pattern

        # PCA-Deflect: Check if this client is flagged
        is_flagged = False
        client_history_model = None
        if pca_deflect_active:
            print(f"[PCA-DEFLECT] Checking flagging status for client {agent_name_key}")
            is_flagged = helper.is_client_flagged(agent_name_key)
            print(f"[PCA-DEFLECT] Client {agent_name_key} flagged status: {is_flagged}")
            
            if is_flagged:
                print(f"[PCA-DEFLECT] Client {agent_name_key} is flagged - will use contaminated model for trigger detection")
                print(f"[PCA-DEFLECT] Will compare contaminated client model vs pure global model")
                print(f"[PCA-DEFLECT] Will run evaluate_globalvslocal and filter_poisoned_data")
                
                # For flagged clients, we'll create the contaminated model during training
                # The client_history_model will be set up later with the contaminated weights

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):
            print(f"[IMAGE_TRAIN] Agent {agent_name_key}, Epoch {epoch}")

            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            # Check if this is a poison epoch for adversarial agents
            is_poison_epoch = (is_poison and 
                             agent_name_key in helper.params['adversary_list'] and 
                             (epoch in localmodel_poison_epochs))

            # PCA-Deflect: Dual model training for flagged clients
            if is_flagged and not is_poison_epoch:
                print(f"[PCA-DEFLECT] === FILTERED TRAINING FOR FLAGGED CLIENT {agent_name_key} ===")
                
                # First, train the client normally to get the contaminated model
                print(f"[PCA-DEFLECT] Step 1: Training client normally to obtain contaminated model")
                
                # Train client with current data to get contaminated weights
                contaminated_model = copy.deepcopy(model)  # Start with current global model
                contaminated_model.train()
                
                # Get client's optimizer
                optimizer_contaminated = optim.SGD(contaminated_model.parameters(), lr=helper.params['lr'],
                                                 momentum=helper.params['momentum'], 
                                                 weight_decay=helper.params['decay'])
                
                # Train for one epoch to get contaminated model
                _, data_iterator = helper.train_data[agent_name_key]
                for batch_id, batch in enumerate(data_iterator):
                    data, targets = helper.get_batch(data_iterator, batch, evaluation=False)
                    
                    optimizer_contaminated.zero_grad()
                    output = contaminated_model(data)
                    loss = nn.functional.cross_entropy(output, targets)
                    loss.backward()
                    optimizer_contaminated.step()
                
                print(f"[PCA-DEFLECT] Step 2: Using contaminated model for trigger detection")
                
                # Get data iterator for trigger detection
                _, data_iterator = helper.train_data[agent_name_key]
                
                # Collect all data for this client
                all_data = []
                all_labels = []
                all_indices = []
                dataset_size = 0
                
                for batch_id, batch in enumerate(data_iterator):
                    data, targets = helper.get_batch(data_iterator, batch, evaluation=False)
                    all_data.append(data)
                    all_labels.append(targets)
                    dataset_size += len(data)
                    # Track original indices
                    batch_start = batch_id * data_iterator.batch_size
                    all_indices.extend(range(batch_start, batch_start + len(data)))
                
                # Create temporary dataset
                all_data = torch.cat(all_data, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                temp_dataset = torch.utils.data.TensorDataset(all_data, all_labels)
                
                # Detect trigger label using contaminated model vs pure global model
                print(f"[PCA-DEFLECT] === CALLING evaluate_globalvslocal for client {agent_name_key} ===")
                print(f"[PCA-DEFLECT] Comparing contaminated client model vs pure global model")
                trigger_label = pca_deflect.evaluate_globalvslocal(
                    model, contaminated_model, temp_dataset, config.device
                )
                print(f"[PCA-DEFLECT] Detected trigger label: {trigger_label}")
                
                # Get triggered indices for this client
                triggered_indices = set()
                if agent_name_key in helper.params['adversary_list']:
                    print(f"[PCA-DEFLECT] Client {agent_name_key} is in adversary list - identifying triggered samples")
                    # This is a malicious client - identify which samples are triggered
                    _, client_data_iterator = helper.train_data[agent_name_key]
                    sample_count = 0
                    for batch_id, batch in enumerate(client_data_iterator):
                        data, targets = helper.get_batch(client_data_iterator, batch, evaluation=False)
                        # Check first poisoning_per_batch samples in each batch
                        for i in range(min(helper.params['poisoning_per_batch'], len(data))):
                            if targets[i] == helper.params['poison_label_swap']:
                                triggered_indices.add(sample_count + i)
                        sample_count += len(data)
                    print(f"[PCA-DEFLECT] Found {len(triggered_indices)} triggered samples for client {agent_name_key}")
                
                # Filter poisoned data using contaminated model vs pure global model
                print(f"[PCA-DEFLECT] === CALLING filter_poisoned_data for client {agent_name_key} ===")
                print(f"[PCA-DEFLECT] Using contaminated client model vs pure global model for filtering")
                filtered_data, poisoned_data, tp, fp = pca_deflect.filter_poisoned_data(
                    agent_name_key, temp_dataset, all_indices, 
                    model, contaminated_model, trigger_label, 
                    triggered_indices, config.device
                )
                print(f"[PCA-DEFLECT] Filtering results - TP: {tp}, FP: {fp}")
                
                # Update detection metrics
                round_detection_metrics['true_positives'][agent_name_key] = tp
                round_detection_metrics['false_positives'][agent_name_key] = fp
                round_detection_metrics['total_triggers'][agent_name_key] = len(triggered_indices)
                
                # Train clean model on filtered data
                print(f"[PCA-DEFLECT] Training clean model on {len(filtered_data)} filtered samples")
                filtered_loader = torch.utils.data.DataLoader(
                    filtered_data, batch_size=helper.params['batch_size'], shuffle=True
                )
                
                total_loss = 0.
                correct = 0
                filtered_dataset_size = 0
                
                for batch_id, batch in enumerate(filtered_loader):
                    optimizer.zero_grad()
                    data, targets = batch[0].to(config.device), batch[1].to(config.device)
                    filtered_dataset_size += len(data)
                    
                    output = model(data)
                    loss = nn.functional.cross_entropy(output, targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.data
                    pred = output.data.max(1)[1]
                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                
                acc = 100.0 * (float(correct) / float(filtered_dataset_size)) if filtered_dataset_size > 0 else 0
                total_l = total_loss / filtered_dataset_size if filtered_dataset_size > 0 else 0
                
                print(f'[PCA-DEFLECT] Clean training - Accuracy: {correct}/{filtered_dataset_size} ({acc:.4f}%)')
                
                # Train history model on poisoned data (if any)
                if len(poisoned_data) > 0 and client_history_model is not None:
                    print(f"[PCA-DEFLECT] Training history model on {len(poisoned_data)} poisoned samples")
                    poisoned_loader = torch.utils.data.DataLoader(
                        poisoned_data, batch_size=helper.params['batch_size'], shuffle=True
                    )
                    
                    history_optimizer = torch.optim.SGD(
                        client_history_model.parameters(), 
                        lr=helper.params['lr'],
                        momentum=helper.params['momentum'],
                        weight_decay=helper.params['decay']
                    )
                    
                    for batch_id, batch in enumerate(poisoned_loader):
                        history_optimizer.zero_grad()
                        data, targets = batch[0].to(config.device), batch[1].to(config.device)
                        output = client_history_model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()
                        history_optimizer.step()
                
                num_samples_dict[agent_name_key] = dataset_size
                
            elif is_poison_epoch:
                print(f"[IMAGE_TRAIN] === POISON TRAINING FOR AGENT {agent_name_key} ===")
                main.logger.info('poison_now')

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
                    dis2global_list = []
                    
                    for batch_id, batch in enumerate(data_iterator):
                        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adversarial_index, evaluation=False)
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        poison_data_count += poison_num

                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)

                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                        # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                        loss = helper.params['alpha_loss'] * class_loss + \
                               (1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward()

                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        poison_optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        if helper.params["batch_track_distance"]:
                            # we can calculate distance to this model now.
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)

                    if step_lr:
                        scheduler.step()
                        main.logger.info(f'Current lr: {scheduler.get_lr()}')

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        f'[IMAGE_TRAIN] ___PoisonTrain {model.name}, epoch {epoch}, local model {agent_name_key}, internal_epoch {internal_epoch}, Average loss: {total_l:.4f}, '
                        f'Accuracy: {correct}/{dataset_size} ({acc:.4f}%), train_poison_data_count: {poison_data_count}')
                    csv_record.train_result.append(
                        [agent_name_key, temp_local_epoch,
                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])
                    
                    num_samples_dict[agent_name_key] = dataset_size
                    if helper.params["batch_track_distance"]:
                        main.logger.info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}.')

                # internal epoch finish
                main.logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
                main.logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                                 f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

                if not helper.params['baseline']:
                    main.logger.info(f'will scale.')
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                                   model=model, is_poison=False,
                                                                                   visualize=False,
                                                                                   agent_name_key=agent_name_key)
                    csv_record.test_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          visualize=False,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    clip_rate = helper.params['scale_weights_poison']
                    main.logger.info(f"Scaling by {clip_rate}")
                    for key, value in model.state_dict().items():
                        target_value = last_local_model[key]
                        new_value = target_value + (value - target_value) * clip_rate
                        model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(model, target_params_variables)
                    main.logger.info(
                        f'Scaled Norm after poisoning: '
                        f'{helper.model_global_norm(model)}, distance: {distance}')
                    csv_record.scale_temp_one_row.append(epoch)
                    csv_record.scale_temp_one_row.append(round(distance, 4))
                    if helper.params["batch_track_distance"]:
                        temp_data_len = len(helper.train_data[agent_name_key][1])
                        
                distance = helper.model_dist_norm(model, target_params_variables)
                main.logger.info(f"Total norm for {current_number_of_adversaries} "
                                 f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

            else:
                # Regular training (benign agents or non-poison epochs)
                print(f"[IMAGE_TRAIN] === REGULAR TRAINING FOR AGENT {agent_name_key} ===")
                
                temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1

                    _, data_iterator = helper.train_data[agent_name_key]
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []
            
                    for batch_id, batch in enumerate(data_iterator):
                        optimizer.zero_grad()
                        data, targets = helper.get_batch(data_iterator, batch, evaluation=False)
                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()

                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        if helper.params["vis_train_batch_loss"]:
                            cur_loss = loss.data
                            temp_data_len = len(data_iterator)
                            
                        if helper.params["batch_track_distance"]:
                            # we can calculate distance to this model now
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    
                    print(f'[IMAGE_TRAIN] ___Train {model.name}, epoch {epoch}, local model {agent_name_key}, internal_epoch {internal_epoch}, Average loss: {total_l:.4f}, '
                          f'Accuracy: {correct}/{dataset_size} ({acc:.4f}%)')
                        
                    csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                                    epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    num_samples_dict[agent_name_key] = dataset_size

                    if helper.params["batch_track_distance"]:
                        main.logger.info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}.')

                # test local model after internal epoch finishing
                epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                               model=model, is_poison=False, visualize=True,
                                                                               agent_name_key=agent_name_key)
                csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

            # Poison testing for adversarial agents
            if is_poison:
                if agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          visualize=True,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                #  test on local triggers
                if agent_name_key in helper.params['adversary_list']:
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
                        test.Mytest_poison_agent_trigger(helper=helper, model=model, agent_name_key=agent_name_key)
                    csv_record.poisontriggertest_result.append(
                        [agent_name_key, str(agent_name_key) + "_trigger", "", epoch, epoch_loss,
                         epoch_acc, epoch_corret, epoch_total])

            # PCA-Deflect: Update client history
            if pca_deflect_active:
                helper.update_client_history(agent_name_key, model.state_dict())
                if client_history_model is not None:
                    # Store the poisoned model state for flagged clients
                    helper.update_client_history(f"{agent_name_key}_poison", client_history_model.state_dict())

            # update the model weight
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                last_local_model[name] = copy.deepcopy(data)

            if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                epochs_local_update_list.append(client_grad)
            else:
                epochs_local_update_list.append(local_model_update_dict)

        epochs_submit_update_dict[agent_name_key] = epochs_local_update_list
        print(f"[IMAGE_TRAIN] Completed training for agent {agent_name_key}")

    # PCA-Deflect: Update global detection metrics
    if pca_deflect_active:
        pca_deflect.detection_metrics['true_positives'].update(round_detection_metrics['true_positives'])
        pca_deflect.detection_metrics['false_positives'].update(round_detection_metrics['false_positives'])
        pca_deflect.detection_metrics['total_triggers'].update(round_detection_metrics['total_triggers'])

    print(f"[IMAGE_TRAIN] === IMAGE TRAINING COMPLETED ===")
    print(f"[IMAGE_TRAIN] Trained {len(agent_name_keys)} agents")
    print(f"[IMAGE_TRAIN] PCA-Deflect defense active: {pca_deflect_active}")
    
    return epochs_submit_update_dict, num_samples_dict