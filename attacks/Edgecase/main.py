import datetime
import json
import os
import logging
import torch
import torch.nn as nn
import attacks.Edgecase.train as train_edge
import attacks.Edgecase.test as test_edge
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms

from attacks.Edgecase.image_helper import ImageHelper


import yaml
import time

import numpy as np
import random
import config 
import copy
import defenses.pca_deflect

# Import NAB Defense
from defenses.nab_defense import NABDefense
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
        test_edge.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    
def trigger_test_byname(helper, agent_name_key, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test_edge.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)
    


def main_edge(defense_name, dataset_name):
    print('Start training')
    np.random.seed(1)
    with open('./attacks/Edgecase/utils/params.yaml') as f:
        params_loaded = yaml.safe_load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    
    params_loaded['aggregation_methods'] = defense_name
    params_loaded['type'] = dataset_name
    if params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_FMNIST or params_loaded['type'] == config.TYPE_EMNIST:
        helper = ImageHelper(current_time = current_time, params = params_loaded, name = params_loaded.get('name','fmnist'))
        helper.load_data()
    else:
        helper = None
    
    print(f'load data done')
    helper.create_model()
    print(f'create model done')
    
    # Initialize NAB Defense if specified
    nab_defense = None
    if defense_name == "nab" or defense_name == "nab_pca":
        print(f"[NAB] Initializing NAB Defense for {dataset_name}")
        nab_defense = NABDefense(helper, current_time)
        print(f"[NAB] NAB Defense initialized successfully")
    
    # Initialize NPD Defense if specified
    npd_defense = None
    print(f"[DEBUG] Checking defense_name: '{defense_name}' vs config.AGGR_NPD: '{config.AGGR_NPD}'")
    if defense_name == config.AGGR_NPD:
        print(f"[NPD] *** INITIALIZING NPD DEFENSE for {dataset_name} ***")
        try:
            npd_defense = NPDDefense(helper, current_time)
            print(f"[NPD] *** NPD Defense initialized successfully ***")
            print(f"[NPD] NPD parameters: epochs={npd_defense.num_epochs}, lr={npd_defense.learning_rate}")
        except Exception as e:
            print(f"[NPD] *** NPD Defense initialization FAILED: {e} ***")
            import traceback
            traceback.print_exc()
            npd_defense = None
    else:
        print(f"[NPD] NPD Defense not selected (defense_name='{defense_name}')")
    
    results_all = {'MA':list(), 'BA': list(), 'NAB_MA': list(), 'NAB_BA': list(), 'NPD_MA': list(), 'NPD_BA': list()}
    ### Create models
    if helper.params['is_poison']:
        print(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    best_loss = float('inf')

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()

        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  # random choose , maybe don't have advasarial
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                for _name_keys in agent_name_keys:
                    if _name_keys in helper.params['adversary_list']:
                        adversarial_name_keys.append(_name_keys)
            else:  # must have advasarial if this epoch is in their poison epoch
                ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                for idx in range(0, len(helper.params['adversary_list'])):
                    for ongoing_epoch in ongoing_epochs:
                        if ongoing_epoch in helper.params['poison_epochs']:
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
        
        print(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
        
        # NAB Defense Pre-training Hook
        if nab_defense is not None:
            print(f"[NAB] Running pre-training hook for epoch {epoch}")
            nab_defense.pre_training_hook(epoch, helper.target_model, agent_name_keys)

        epochs_submit_update_dict, num_samples_dict = train_edge.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys)
        
        print(f'time spent on training: {time.time() - t}')
        
        # Apply defenses in order: NAB -> PCA-Deflect -> Aggregation
        if helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT or defense_name == "nab_pca":
            client_names   = list(epochs_submit_update_dict.keys())
            client_weights = [updates[-1] for updates in epochs_submit_update_dict.values()]

            client_weights_flat = defenses.pca_deflect.extract_client_weights(client_weights)
            outliers, _ = defenses.pca_deflect.apply_pca_to_weights(client_weights_flat, client_names, epoch, [])
            print(f"[PCA-Deflect] Detected outliers: {outliers}")
            outliers = list(outliers) + list(defenses.pca_deflect.flagged_malicious_clients)
            for bad in outliers:
                epochs_submit_update_dict.pop(bad, None)
                num_samples_dict.pop(bad, None)
                if bad in agent_name_keys:
                    agent_name_keys.remove(bad)  
                    print(f"[PCA-Deflect] Removed outlier's influence: {bad}")

            # Optional global-vs-local trigger detection (same behavior as BadNet)
            try:
                should_run_trigger_detection = (
                    config.PCA_DEFLECT_RANDOM_TRIGGER_DETECTION_ON and
                    helper.params.get("aggregation_methods") == config.AGGR_PCA_DEFLECT and
                    epoch >= config.PCA_DEFLECT_MIN_ROUNDS_FOR_TRIGGER_DETECTION and
                    (not config.PCA_DEFLECT_TRIGGER_DETECTION_ONCE or not defenses.pca_deflect.global_vs_local_executed)
                )

                if should_run_trigger_detection:
                    print(f"[PCA-DEFLECT] Starting trigger detection analysis at epoch {epoch}")
                    current_client_names = list(epochs_submit_update_dict.keys())
                    selected_client = defenses.pca_deflect.select_random_client_for_analysis(current_client_names, defenses.pca_deflect.flagged_malicious_clients)
                    print(f"[PCA-DEFLECT] Selected client {selected_client} for global vs local analysis")
                    client_model = copy.deepcopy(helper.local_model)
                    detected_trigger = defenses.pca_deflect.evaluate_globalvslocal(helper.target_model, client_model, helper.test_data, config.device, epoch=epoch, client_id=selected_client)
                    defenses.pca_deflect.detected_trigger_label = detected_trigger
                    defenses.pca_deflect.global_vs_local_executed = True
                    print(f"[PCA-DEFLECT] Global vs local analysis completed. Trigger label: {detected_trigger}")
            except Exception as e:
                print(f"[PCA-DEFLECT] Global vs local analysis failed: {e}")
                import traceback
                traceback.print_exc()

        agents_num = len(agent_name_keys)
        
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        
        # if helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT or defense_name == "nab_pca":
        #     for bad in outliers:
        #         if bad not in agent_name_keys:
        #             agent_name_keys.append(bad)
        # change here
        # Model Aggregation
        is_updated = True
        if helper.params['aggregation_methods'] == config.AGGR_MEAN or helper.params['aggregation_methods'] == config.AGGR_PCA_DEFLECT or defense_name in ["nab", "nab_pca"] or helper.params['aggregation_methods'] == config.AGGR_NPD:
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'], 
                                                      agents_num=agents_num)
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)
        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)
            num_oracle_calls = 1

        # NAB Defense Post-aggregation Hook
        if nab_defense is not None:
            nab_defense.post_aggregation_hook(epoch, helper.target_model)

        # Clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        # Standard Testing
        epoch_loss, epoch_acc, epoch_corret, epoch_total = test_edge.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=True, agent_name_key="global")
        results_all['MA'].append(epoch_acc)
        
        # Initialize poison accuracy for summary
        epoch_acc_p = 0.0

        # NAB Filtered Testing (if NAB is active)
        if nab_defense is not None:
            print(f"[NAB] Testing with NAB filtering...")
            nab_loss, nab_acc, nab_correct, nab_total = nab_defense.test_with_filtering(
                helper, temp_global_epoch, helper.target_model)
            results_all['NAB_MA'].append(nab_acc)
            print(f"[NAB] Clean Accuracy with filtering: {nab_correct}/{nab_total} ({nab_acc:.2f}%)")

        # Poison Testing
        if helper.params['is_poison']:
            print(f"[DEBUG] Testing poison accuracy at epoch {temp_global_epoch}")
            print(f"[DEBUG] Using poison test dataset with ARDIS images")
            
            # Standard poison testing
            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test_edge.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    is_poison=True,
                                                                                    visualize=True,
                                                                                    agent_name_key="global")
            results_all['BA'].append(epoch_acc_p)
            print(f"[Epoch {temp_global_epoch}] Backdoor Accuracy: {epoch_corret}/{epoch_total} ({epoch_acc_p:.2f}%)")
            print(f"[DEBUG] Poison accuracy is consistent because it's testing on separate ARDIS dataset, not actual backdoor triggers")

            # NAB Filtered poison testing (if NAB is active)
            if nab_defense is not None:
                print(f"[NAB] Testing poison samples with NAB filtering...")
                nab_poison_loss, nab_poison_acc, nab_poison_correct, nab_poison_total = nab_defense.test_poison_with_filtering(
                    helper, temp_global_epoch, helper.target_model)
                results_all['NAB_BA'].append(nab_poison_acc)
                print(f"[NAB] Backdoor Accuracy with filtering: {nab_poison_correct}/{nab_poison_total} ({nab_poison_acc:.2f}%)")

            # Individual trigger testing
            if len(helper.params['adversary_list']) == 1:  # centralized attack
                if helper.params.get('centralized_test_trigger', False) == True:
                    for j in range(0, helper.params.get('trigger_num', 1)):
                        trigger_test_byindex(helper, j, epoch)
            else:  # distributed attack
                for agent_name_key in helper.params['adversary_list']:
                    trigger_test_byname(helper, agent_name_key, epoch)
                    
            # Save intermediate results
            if epoch % 100 == 0 or epoch == 100:
                filename = f"results_{defense_name}_{dataset_name}_edgecase_nab_100.txt"
                with open(filename, "w") as f:
                    json.dump(results_all, f, indent=2)
                print(f"[Results] Saved intermediate results to {filename}")

        # Add epoch summary
        print(f"[MAIN] Epoch {temp_global_epoch} Summary:")
        print(f"[MAIN]   - Clean Accuracy: {epoch_acc:.2f}%")
        if helper.params['is_poison']:
            print(f"[MAIN]   - Backdoor Accuracy: {epoch_acc_p:.2f}%")
        else:
            print(f"[MAIN]   - Backdoor Accuracy: 0.00%")
        print(f"[MAIN]   - Participants: {len(agent_name_keys)}")
        
        print(f'Epoch {temp_global_epoch} completed in {time.time() - start_time:.2f} sec.')

    # Final results
    print("=== FINAL RESULTS ===")
    print(f"Standard - MA: {results_all['MA'][-1]:.2f}%, BA: {results_all['BA'][-1]:.2f}%" if results_all['MA'] and results_all['BA'] else "No results")
    if nab_defense is not None and results_all['NAB_MA'] and results_all['NAB_BA']:
        print(f"NAB Filtered - MA: {results_all['NAB_MA'][-1]:.2f}%, BA: {results_all['NAB_BA'][-1]:.2f}%")
    
    # Apply NPD Defense after training is complete (only after minimum rounds)
    print(f"[DEBUG] Checking NPD application:")
    print(f"[DEBUG] npd_defense is not None: {npd_defense is not None}")
    print(f"[DEBUG] helper.params['epochs']: {helper.params['epochs']}")
    print(f"[DEBUG] config.NPD_WAIT_ROUNDS: {config.NPD_WAIT_ROUNDS}")
    print(f"[DEBUG] Condition met: {npd_defense is not None and helper.params['epochs'] >= config.NPD_WAIT_ROUNDS}")
    
    final_epoch = helper.params['epochs']
    if npd_defense and final_epoch >= config.NPD_WAIT_ROUNDS:
        print(f"\n[NPD] *** APPLYING NPD DEFENSE - STARTING ***")
        print(f"[NPD] Applying NPD defense after {final_epoch} epochs...")
        try:
            print(f"[NPD] Calling apply_defense with model type: {type(helper.target_model)}")
            defended_model = npd_defense.apply_defense(helper.target_model, helper.train_data)
            
            if defended_model is not None:
                print(f"[NPD] *** DEFENSE APPLICATION SUCCESSFUL ***")
                print(f"[NPD] Defended model type: {type(defended_model)}")
                print(f"[NPD] Testing defended model...")
                
                # Test clean accuracy
                print(f"[NPD] Testing clean accuracy...")
                epoch_loss, npd_clean_acc, epoch_corret, epoch_total = test_edge.Mytest(
                    helper=helper, epoch=helper.params['epochs'],
                    model=defended_model, is_poison=False,
                    visualize=False, agent_name_key="npd_global"
                )
                results_all['NPD_MA'].append(npd_clean_acc)
                print(f"[NPD] *** Clean Accuracy: {npd_clean_acc:.2f}% ({epoch_corret}/{epoch_total}) ***")
                
                # Test attack success rate
                npd_attack_acc = 0.0
                if helper.params['is_poison']:
                    print(f"[NPD] Testing attack success rate...")
                    epoch_loss, npd_attack_acc, epoch_corret, epoch_total = test_edge.Mytest_poison(
                        helper=helper,
                        epoch=helper.params['epochs'],
                        model=defended_model,
                        is_poison=True,
                        visualize=False,
                        agent_name_key="npd_global"
                    )
                    results_all['NPD_BA'].append(npd_attack_acc)
                    print(f"[NPD] *** Attack Success Rate: {npd_attack_acc:.2f}% ({epoch_corret}/{epoch_total}) ***")
                else:
                    results_all['NPD_BA'].append(0.0)
                    
                npd_effectiveness = npd_clean_acc - npd_attack_acc
                print(f"[NPD] *** Defense Effectiveness: {npd_effectiveness:.2f}% ***")
                print(f"[NPD] *** NPD Results - Clean: {npd_clean_acc:.2f}%, Attack: {npd_attack_acc:.2f}% ***")
            else:
                print(f"[NPD] *** DEFENSE APPLICATION FAILED - defended_model is None ***")
                results_all['NPD_MA'].append(0.0)
                results_all['NPD_BA'].append(0.0)
                
        except Exception as e:
            print(f"[NPD] *** DEFENSE APPLICATION FAILED WITH EXCEPTION: {e} ***")
            import traceback
            traceback.print_exc()
            results_all['NPD_MA'].append(0.0)
            results_all['NPD_BA'].append(0.0)
    elif npd_defense and final_epoch < config.NPD_WAIT_ROUNDS:
        print(f"[NPD] *** SKIPPING NPD DEFENSE - insufficient rounds ({final_epoch} < {config.NPD_WAIT_ROUNDS}) ***")
        results_all['NPD_MA'].append(0.0)
        results_all['NPD_BA'].append(0.0)
    else:
        print(f"[NPD] NPD Defense not applied:")
        if npd_defense is None:
            print(f"[NPD] - npd_defense is None (not initialized)")
        results_all['NPD_MA'].append(0.0)
        results_all['NPD_BA'].append(0.0)
    
    # Save final results
    final_filename = f"results_{defense_name}_{dataset_name}_edgecase_nab_final.txt"
    with open(final_filename, "w") as f:
        json.dump(results_all, f, indent=2)
    
    print(f"[Results] Final results saved to {final_filename}")
    return results_all