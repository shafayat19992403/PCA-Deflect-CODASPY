import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import attacks.DBA.train as train_dba
import attacks.DBA.test as test_dba
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import csv
from torchvision import transforms

from attacks.DBA.image_helper import ImageHelper
from attacks.DBA.utils.utils import dict_html
import attacks.DBA.utils.csv_record as csv_record
import yaml
import time
import visdom
import numpy as np
import random
import config  # New config file
import copy
import defenses.pca_deflect
import defenses.pca_deflect_utils
import defenses.nab_defense as nab_defense  # NAB defense import
from defenses.npd import NPDDefense  # NPD Defense import


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")

criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


def trigger_test_byindex(helper, index, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test_dba.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])

def trigger_test_byname(helper, agent_name_key, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test_dba.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    

def main_dba(defense_name, dataset_name):
    print('Start training')
    print(f"[MAIN] Defense: {defense_name}, Dataset: {dataset_name}")
    np.random.seed(1)
    time_start_load_everything = time.time()
    
    # Load parameters
    with open('./attacks/DBA/utils/cifar_params.yaml') as f:
        params_loaded = yaml.safe_load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    
    params_loaded['aggregation_methods'] = defense_name
    params_loaded['type'] = dataset_name
    
    # Create helper based on dataset type
    if params_loaded['type'] == config.TYPE_CIFAR:
        print(f"[MAIN] Loading CIFAR-10 dataset")
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        print(f"[MAIN] Loading MNIST dataset")
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        print(f"[MAIN] Loading Tiny ImageNet dataset")
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_FMNIST or params_loaded['type'] == config.TYPE_EMNIST:
        print(f"[MAIN] Loading Fashion-MNIST/EMNIST dataset")
        helper = ImageHelper(current_time = current_time, params = params_loaded, name = params_loaded.get('name','fmnist'))
        helper.load_data()
    else:
        print(f"[MAIN] ERROR: Unknown dataset type: {dataset_name}")
        helper = None
        return

    print(f'[MAIN] Data loading completed')
    helper.create_model()
    logger.info(f'[MAIN] Model creation completed')
    results_all = {'MA':list(), 'BA': list()}
    
    ### Create models
    if helper.params['is_poison']:
        logger.info(f"[MAIN] Poisoned following participants: {(helper.params['adversary_list'])}")
        print(f"[MAIN] Attack configuration: Poison ratio per batch = {helper.params['poisoning_per_batch']}")

    best_loss = float('inf')

    logger.info(f"[MAIN] Environment for graphs: {helper.params['environment_name']}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # PCA-Deflect Initialization
    pca_deflect_active = (defense_name == config.AGGR_PCA_DEFLECT)
    if pca_deflect_active:
        print(f"[PCA-DEFLECT] === PCA-DEFLECT DEFENSE INITIALIZATION ===")
        print(f"[PCA-DEFLECT] Initializing PCA-Deflect defense for {dataset_name}")
        print(f"[PCA-DEFLECT] Trust factor: {config.PCA_DEFLECT_TRUST_FACTOR}")
        print(f"[PCA-DEFLECT] Confidence threshold: {config.PCA_DEFLECT_CONFIDENCE_THRESHOLD}")
        print(f"[PCA-DEFLECT] DBSCAN eps: normal={config.PCA_DEFLECT_DBSCAN_EPS_NORMAL}, flagged={config.PCA_DEFLECT_DBSCAN_EPS_FLAGGED}")
        print(f"[PCA-DEFLECT] Min rounds before flagging: {config.PCA_DEFLECT_MIN_ROUNDS_BEFORE_DETECTION}")
        print(f"[PCA-DEFLECT] === PCA-DEFLECT DEFENSE READY ===")

    # NAB Defense Initialization
    nab_handler = None
    if defense_name == config.AGGR_NAB:
        print(f"[NAB] === NAB DEFENSE INITIALIZATION ===")
        print(f"[NAB] Initializing NAB defense for {dataset_name}")
        print(f"[NAB] Wait period: {config.NAB_WAIT_ROUNDS} rounds")
        print(f"[NAB] Isolation ratios: {config.NAB_ISOLATION_RATIOS}")
        print(f"[NAB] LGA gamma: {config.NAB_LGA_GAMMA}")
        print(f"[NAB] Stamp size: {config.NAB_STAMP_SIZE}x{config.NAB_STAMP_SIZE}")
        
        nab_handler = nab_defense.NABDefense(helper, current_time)
        print(f"[NAB] NAB defense initialized successfully")
        print(f"[NAB] === NAB DEFENSE READY ===")

        helper.nab_trainer = nab_handler.nab_trainer

    # NPD Defense Initialization
    npd_handler = None
    if defense_name == config.AGGR_NPD:
        print(f"[NPD] === NPD DEFENSE INITIALIZATION ===")
        print(f"[NPD] Initializing NPD defense for {dataset_name}")
        
        try:
            npd_handler = NPDDefense(helper, current_time)
            print(f"[NPD] NPD defense initialized successfully")
            print(f"[NPD] === NPD DEFENSE READY ===")
        except Exception as e:
            print(f"[NPD] NPD defense initialization failed: {e}")
            npd_handler = None

    submit_update_dict = None
    num_no_progress = 0

    print(f"[MAIN] Starting federated training loop")
    print(f"[MAIN] Total epochs: {helper.params['epochs']}, Aggregation interval: {helper.params['aggr_epoch_interval']}")

    # PCA-Deflect: Initialize client histories at the start
    if helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT:
        print(f"[PCA-DEFLECT] Initializing client histories at start of training")
        # Initialize all participants with current global model as clean baseline
        for participant in helper.participants_list:
            helper.update_client_history(participant, helper.target_model.state_dict())

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()
        
        print(f"\n[MAIN] ========== EPOCH {epoch} ==========")

        # NAB Phase Check
        nab_phase = "normal"
        if nab_handler:
            nab_phase = nab_handler.get_current_phase(epoch)
            print(f"[NAB] Epoch {epoch}: Current phase = {nab_phase}")
            
            if nab_phase == "wait":
                print(f"[NAB] Wait phase: {epoch}/{config.NAB_WAIT_ROUNDS} - Building baseline")
            elif nab_phase == "detection":
                print(f"[NAB] Detection phase: Running LGA detection and clean model training")
            elif nab_phase == "nab_training":
                print(f"[NAB] NAB training phase: Applying defensive backdoor")

        # Agent selection logic (original DBA)
        agent_name_keys = []
        adversarial_name_keys = []
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                for _name_keys in agent_name_keys:
                    if _name_keys in helper.params['adversary_list']:
                        adversarial_name_keys.append(_name_keys)
            else:  
                ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                for idx in range(0, len(helper.params['adversary_list'])):
                    for ongoing_epoch in ongoing_epochs:
                        if ongoing_epoch in helper.params[str(idx) + '_poison_epochs']:
                            if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                                adversarial_name_keys.append(helper.params['adversary_list'][idx])

                nonattacker=[]
                for adv in helper.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist+nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary']==False:
                adversarial_name_keys=copy.deepcopy(helper.params['adversary_list'])
        
        logger.info(f'[MAIN] Server Epoch:{epoch} choose agents : {agent_name_keys}.')
        if adversarial_name_keys:
            print(f"[MAIN] Adversarial agents in this round: {adversarial_name_keys}")

        # PCA-Deflect: Inform flagged clients about their status
        if pca_deflect_active and len(helper.get_flagged_clients()) > 0:
            flagged_in_round = [client for client in agent_name_keys if helper.is_client_flagged(client)]
            if flagged_in_round:
                print(f"[PCA-DEFLECT] Informing flagged clients to perform dual-model training: {flagged_in_round}")

        # NAB Pre-training Hook
        if nab_handler and nab_phase in ['detection', 'nab_training']:
            print(f"[NAB] === PRE-TRAINING PHASE ===")
            print(f"[NAB] Running detection and clean model update")
            nab_handler.pre_training_hook(epoch, helper.target_model, agent_name_keys)
            print(f"[NAB] Pre-training phase completed")

        # PCA-Deflect: Reset detection metrics for this round
        if pca_deflect_active:
            defenses.pca_deflect.reset_detection_metrics()

        # PCA-Deflect: Save client parameter histories BEFORE training starts
        if helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT:
            print(f"[PCA-DEFLECT] Saving client baseline histories before training")
            for agent_name in agent_name_keys:
                # Save current target model state as baseline for each participating client
                helper.update_client_history(agent_name, helper.target_model.state_dict())

        # Local training phase
        print(f"[MAIN] Starting local training for {len(agent_name_keys)} agents")
        epochs_submit_update_dict, num_samples_dict = train_dba.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys)
        
        logger.info(f'[MAIN] Time spent on local training: {time.time() - t}')
        
        # PCA-Deflect: Apply outlier detection and flag clients
        if helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT:
            print(f"[PCA-DEFLECT] Applying PCA-Deflect outlier detection")
            client_names = list(epochs_submit_update_dict.keys())
            client_weights = [updates[-1] for updates in epochs_submit_update_dict.values()]

            client_weights_flat = defenses.pca_deflect.extract_client_weights(client_weights)
            
            # Apply PCA-based outlier detection
            outliers, _ = defenses.pca_deflect.apply_pca_to_weights(
                client_weights_flat, client_names, epoch, helper.get_flagged_clients()
            )
            print(f"[PCA-DEFLECT] Detected outliers in this round: {outliers}")
            print(f"[PCA-DEFLECT] Previously flagged clients: {helper.get_flagged_clients()}")

            # Update outlier history and flag clients (immediate or persistent based on config)
            newly_flagged = []
            for outlier in outliers:
                was_flagged_before = helper.is_client_flagged(outlier)
                print(f"[PCA-DEFLECT] Processing outlier {outlier}, was_flagged_before: {was_flagged_before}")
                helper.update_outlier_history(outlier, epoch)
                
                # Check if client is now flagged (either immediately or after persistence)
                is_now_flagged = helper.is_client_flagged(outlier)
                print(f"[PCA-DEFLECT] Client {outlier} flagged status after update: {is_now_flagged}")
                
                if is_now_flagged:
                    if not was_flagged_before:
                        newly_flagged.append(outlier)
                        print(f"[PCA-DEFLECT] Client {outlier} newly flagged - will use filtered training next round")
                    print(f"[PCA-DEFLECT] Client {outlier} is flagged as malicious")
                    
            if newly_flagged:
                print(f"[PCA-DEFLECT] Newly flagged clients in this round: {newly_flagged}")
                print(f"[PCA-DEFLECT] These clients will undergo evaluate_globalvslocal and filter_poisoned_data in NEXT round")
            
            # Show current status of all flagged clients
            all_flagged = helper.get_flagged_clients()
            if all_flagged:
                print(f"[PCA-DEFLECT] All currently flagged clients: {all_flagged}")

            # Apply trust factor to flagged clients' updates (not remove them)
            flagged_clients_in_round = [client for client in client_names if helper.is_client_flagged(client)]
            if flagged_clients_in_round:
                print(f"[PCA-DEFLECT] Applying trust factor to flagged clients: {flagged_clients_in_round}")
                trust_factor = helper.params.get('trust_factor', config.PCA_DEFLECT_TRUST_FACTOR)
                
                # Apply trust factor to updates
                for client_idx, client_name in enumerate(client_names):
                    if client_name in flagged_clients_in_round:
                        # Scale down the update
                        for update_dict in epochs_submit_update_dict[client_name]:
                            for param_name in update_dict:
                                update_dict[param_name] *= trust_factor
                        print(f"[PCA-DEFLECT] Applied trust factor {trust_factor} to client {client_name}")

            # Calculate and print detection metrics if we have flagged clients
            flagged_malicious = set(helper.get_flagged_clients()).intersection(set(helper.params['adversary_list']))
            if len(flagged_malicious) > 0:
                total_tp = 0
                total_triggers = 0
                
                for client_id in flagged_malicious:
                    tp = defenses.pca_deflect.detection_metrics['true_positives'].get(client_id, 0)
                    total = defenses.pca_deflect.detection_metrics['total_triggers'].get(client_id, 0)
                    total_tp += tp
                    total_triggers += total
                
                if total_triggers > 0:
                    avg_tp_percentage = (total_tp / total_triggers) * 100
                    print(f"\n[PCA-DEFLECT] Average trigger detection rate: {avg_tp_percentage:.2f}%")
                    print(f"[PCA-DEFLECT] (Detected {total_tp}/{total_triggers} poisoned samples across {len(flagged_malicious)} malicious clients)")
                    
                    # Save detection report
                    if helper.params.get('track_detection_metrics', False):
                        stats = defenses.pca_deflect_utils.calculate_detection_statistics(
                            defenses.pca_deflect.detection_metrics,
                            agent_name_keys,
                            helper.params['adversary_list']
                        )
                        defenses.pca_deflect_utils.save_detection_report(stats, epoch)

        agents_num = len(agent_name_keys)
        print(f"[MAIN] Aggregating updates from {agents_num} agents")
       
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)

        # Model aggregation
        is_updated = True
        print(f"[MAIN] Performing model aggregation using: {helper.params['aggregation_methods']}")
        
        if helper.params['aggregation_methods'] == config.AGGR_MEAN or helper.params['aggregation_methods'] == config.AGGR_PCA_DEFLECT or helper.params['aggregation_methods'] == config.AGGR_NPD:
            # Note: Trust factor already applied to flagged clients' updates
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'], 
                                                      agents_num=agents_num)
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            print(f"[DEFENSE] Applying Geometric Median aggregation")
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)
        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            print(f"[DEFENSE] Applying FoolsGold aggregation")
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_NAB:
            # NAB Aggregation (uses mean aggregation as base)
            print(f"[NAB] Performing NAB aggregation at epoch {epoch}")
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'], agents_num=agents_num)
            num_oracle_calls = 1
            print(f"[NAB] NAB aggregation completed")

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        # NAB Post-aggregation Hook
        if nab_handler:
            print(f"[NAB] === POST-AGGREGATION PHASE ===")
            print(f"[NAB] Updating global model information")
            nab_handler.post_aggregation_hook(epoch, helper.target_model)
            print(f"[NAB] Post-aggregation phase completed")

        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        # Evaluation phase
        print(f"[MAIN] === EVALUATION PHASE ===")
        
        # Clean accuracy evaluation
        if nab_handler and nab_phase == 'nab_training':
            print(f"[NAB] Applying test-time filtering for clean evaluation")
            epoch_loss, epoch_acc, epoch_corret, epoch_total = nab_handler.test_with_filtering(helper, temp_global_epoch, helper.target_model)
        else:
            epoch_loss, epoch_acc, epoch_corret, epoch_total = test_dba.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                           model=helper.target_model, is_poison=False,
                                                                           visualize=True, agent_name_key="global")
        
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        results_all['MA'].append(epoch_acc)
        
        print(f"[EVAL] Clean Accuracy: {epoch_corret}/{epoch_total} ({epoch_acc:.2f}%)")

        if len(csv_record.scale_temp_one_row)>0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        # Poison evaluation
        if helper.params['is_poison']:
            print(f"[EVAL] === BACKDOOR EVALUATION ===")
            
            if nab_handler and nab_phase == 'nab_training':
                print(f"[NAB] Applying test-time filtering for poison evaluation")
                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = nab_handler.test_poison_with_filtering(helper, temp_global_epoch, helper.target_model)
            else:
                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test_dba.Mytest_poison(helper=helper,
                                                                                        epoch=temp_global_epoch,
                                                                                        model=helper.target_model,
                                                                                        is_poison=True,
                                                                                        visualize=True,
                                                                                        agent_name_key="global")

            csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])

            results_all['BA'].append(epoch_acc_p)

            csv_record.poisontriggertest_result.append(
                ["global", "combine", "", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
            
            print(f"[EVAL] Backdoor Accuracy (combine): {epoch_corret}/{epoch_total} ({epoch_acc_p:.2f}%)")
            
            # Individual trigger testing
            if len(helper.params['adversary_list']) == 1:  
                if helper.params['centralized_test_trigger'] == True:  
                    print(f"[EVAL] Testing individual triggers for centralized attack")
                    for j in range(0, helper.params['trigger_num']):
                        trigger_test_byindex(helper, j, epoch)
            else:  
                print(f"[EVAL] Testing individual triggers for distributed attack")
                for agent_name_key in helper.params['adversary_list']:
                    trigger_test_byname(helper, agent_name_key, epoch)

        # Save model
        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        
        # Epoch summary
        epoch_time = time.time() - start_time
        logger.info(f'[MAIN] Epoch {epoch} completed in {epoch_time:.2f} seconds')
        print(f"[MAIN] Epoch {epoch} Summary:")
        print(f"[MAIN]   - Clean Accuracy: {epoch_acc:.2f}%")
        if helper.params['is_poison']:
            print(f"[MAIN]   - Backdoor Accuracy: {epoch_acc_p:.2f}%")
        print(f"[MAIN]   - Participants: {len(agent_name_keys)}")
        if nab_handler:
            print(f"[MAIN]   - NAB Phase: {nab_phase}")
        if pca_deflect_active:
            print(f"[MAIN]   - Currently flagged clients: {helper.get_flagged_clients()}")
            print(f"[MAIN]   - Outliers detected this round: {outliers if 'outliers' in locals() else []}")

    # Final results
    print(f"\n[MAIN] === TRAINING COMPLETED ===")
    print(f"[MAIN] Final Results:")
    print(f"[MAIN] Clean Accuracy History: {results_all['MA']}")
    if helper.params['is_poison']:
        print(f"[MAIN] Backdoor Accuracy History: {results_all['BA']}")
    
    # Apply NPD Defense after training is complete (only after minimum rounds)
    final_epoch = helper.params['epochs']
    if npd_handler and final_epoch >= config.NPD_WAIT_ROUNDS:
        print(f"\n[NPD] === APPLYING NPD DEFENSE ===")
        print(f"[NPD] Applying NPD defense after {final_epoch} epochs...")
        try:
            defended_model = npd_handler.apply_defense(helper.target_model, helper.train_data)
            if defended_model is not None:
                print(f"[NPD] Testing defended model...")
                
                # Test clean accuracy
                epoch_loss, npd_clean_acc, epoch_corret, epoch_total = test_dba.Mytest(
                    helper=helper, epoch=helper.params['epochs'],
                    model=defended_model, is_poison=False,
                    visualize=False, agent_name_key="npd_global"
                )
                print(f"[NPD] Clean Accuracy: {npd_clean_acc:.2f}%")
                
                # Test attack success rate
                npd_attack_acc = 0.0
                if helper.params['is_poison']:
                    epoch_loss, npd_attack_acc, epoch_corret, epoch_total = test_dba.Mytest_poison(
                        helper=helper,
                        epoch=helper.params['epochs'],
                        model=defended_model,
                        is_poison=True,
                        visualize=False,
                        agent_name_key="npd_global"
                    )
                    print(f"[NPD] Attack Success Rate: {npd_attack_acc:.2f}%")
                    
                npd_effectiveness = npd_clean_acc - npd_attack_acc
                print(f"[NPD] Defense Effectiveness: {npd_effectiveness:.2f}%")
                print(f"[NPD] NPD Defense successfully applied")
                
        except Exception as e:
            print(f"[NPD] Defense application failed: {e}")
    elif npd_handler and final_epoch < config.NPD_WAIT_ROUNDS:
        print(f"[NPD] Skipping NPD Defense - insufficient rounds ({final_epoch} < {config.NPD_WAIT_ROUNDS})")
    
    if pca_deflect_active:
        print(f"[PCA-DEFLECT] PCA-Deflect Defense Summary:")
        print(f"[PCA-DEFLECT] Total training epochs: {helper.params['epochs']}")
        print(f"[PCA-DEFLECT] Final flagged clients: {helper.get_flagged_clients()}")
        
        # Final detection report
        if helper.params.get('track_detection_metrics', False) and len(helper.get_flagged_clients()) > 0:
            final_stats = defenses.pca_deflect_utils.calculate_detection_statistics(
                defenses.pca_deflect.detection_metrics,
                helper.get_flagged_clients(),
                helper.params['adversary_list']
            )
            print(f"[PCA-DEFLECT] Final average trigger detection rate: {final_stats['overall']['avg_tp_rate']:.2f}%")
            print(f"[PCA-DEFLECT] Defense successfully applied")

    if nab_handler:
        print(f"[NAB] NAB Defense Summary:")
        print(f"[NAB] Total training epochs: {helper.params['epochs']}")
        print(f"[NAB] Wait period: {config.NAB_WAIT_ROUNDS} epochs")
        print(f"[NAB] Defense successfully applied")

    print(results_all)
    logger.info('[MAIN] Saving all the graphs.')
    logger.info(f"[MAIN] This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

if __name__ == "__main__":
    # Example usage
    main_dba("pca-deflect", "mnist")