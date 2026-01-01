import datetime
import json
import os
import logging
import torch
import torch.nn as nn
import attacks.Badnet.train as train_badnet
import attacks.Badnet.test as test_badnet
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from defenses.nab_defense import NABDefense  # NAB Defense import
from defenses.npd import NPDDefense  # NPD Defense import
from torchvision import transforms

from attacks.Badnet.image_helper import ImageHelper

import yaml
import time

import numpy as np
import random
import config 
import copy
import defenses.pca_deflect

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")

criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


def trigger_test_byindex(helper, index, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test_badnet.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    
def trigger_test_byname(helper, agent_name_key, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test_badnet.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)


def main_badnet(defense_name, dataset_name):
    print('Start training with defense:', defense_name)
    np.random.seed(1)
    with open('./attacks/Badnet/utils/params.yaml') as f:
        params_loaded = yaml.safe_load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    
    params_loaded['aggregation_methods'] = defense_name
    params_loaded['type'] = dataset_name
    
    # Enable NAB defense if selected
    if defense_name == config.AGGR_NAB:
        params_loaded['use_nab_defense'] = True
        print('[NAB] NAB Defense enabled')
    else:
        params_loaded['use_nab_defense'] = False

    # Enable NPD defense if selected
    if defense_name == config.AGGR_NPD:
        params_loaded['use_npd_defense'] = True
        print('[NPD] NPD Defense enabled')
    else:
        params_loaded['use_npd_defense'] = False

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
        helper = ImageHelper(current_time=current_time, params=params_loaded, 
                            name=params_loaded.get('name','fmnist'))
        helper.load_data()
    else:
        helper = None

    print(f'load data done')
    helper.create_model()
    print(f'create model done')

    nab_defense = None
    if params_loaded.get('use_nab_defense', False):
        print('[NAB] Initializing NAB Defense...')
        try:
            nab_defense = NABDefense(helper, helper.current_time)
            print('[NAB] NAB Defense initialized successfully')
        except Exception as e:
            print(f'[NAB] NAB Defense initialization failed: {e}')
            nab_defense = None

    npd_defense = None
    if params_loaded.get('use_npd_defense', False):
        print('[NPD] Initializing NPD Defense...')
        try:
            npd_defense = NPDDefense(helper, helper.current_time)
            print('[NPD] NPD Defense initialized successfully')
        except Exception as e:
            print(f'[NPD] NPD Defense initialization failed: {e}')
            npd_defense = None
    
    results_all = {
        'MA': list(),        # Main Accuracy (Clean)
        'BA': list(),        # Backdoor Attack Success Rate
        'NAB_MA': list(),    # NAB Clean Accuracy
        'NAB_BA': list(),    # NAB Backdoor Accuracy
        'NPD_MA': list(),    # NPD Clean Accuracy
        'NPD_BA': list(),    # NPD Backdoor Accuracy
    }
    
    if helper.params['is_poison']:
        print(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    best_loss = float('inf')
    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # Main training loop
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()

        # Client selection logic
        agent_name_keys = helper.participants_list
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
                        if ongoing_epoch in helper.params['poison_epochs']:
                            if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                                adversarial_name_keys.append(helper.params['adversary_list'][idx])

                nonattacker = []
                for adv in helper.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist + nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary'] == False:
                adversarial_name_keys = copy.deepcopy(helper.params['adversary_list'])
        
        print(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
        print(f'Adversarial agents this round: {adversarial_name_keys}')
        
        # Apply NAB Defense Pre-Training Hook
        if nab_defense and epoch >= config.NAB_WAIT_ROUNDS:
            print(f'[NAB] Applying NAB pre-training hook for epoch {epoch}')
            try:
                nab_defense.pre_training_hook(epoch, helper.target_model, agent_name_keys)
            except Exception as e:
                print(f'[NAB] Pre-training hook failed: {e}')
        
        epochs_submit_update_dict, num_samples_dict = train_badnet.train(
            helper=helper, start_epoch=epoch,
            local_model=helper.local_model,
            target_model=helper.target_model,
            is_poison=helper.params['is_poison'],
            agent_name_keys=agent_name_keys
        )
        
        print(f'time spent on training: {time.time() - t}')

        outliers = []
        if helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT and epoch >= config.PCA_DEFLECT_MIN_ROUNDS_BEFORE_DETECTION:
            print(f"[PCA] Applying PCA deflect defense...")
            client_names = list(epochs_submit_update_dict.keys())
            client_weights = [updates[-1] for updates in epochs_submit_update_dict.values()]
            client_weights_flat = defenses.pca_deflect.extract_client_weights(client_weights)
            outliers, _ = defenses.pca_deflect.apply_pca_to_weights(client_weights_flat, client_names, epoch, [])
            # outliers= [0, 1, 2, 3]
            print(f"[PCA] PCA outliers: {outliers}")
            print(f"[PCA] Available client_names: {client_names} (types: {[type(name) for name in client_names]})")
            print(f"[PCA] Current flagged_malicious_clients BEFORE flagging: {list(defenses.pca_deflect.flagged_malicious_clients)}")
            
            # Flag the outlier clients as malicious for filtering
            for outlier_id in outliers:
                # Try both integer and string versions for matching
                found = False
                for client_name in client_names:
                    # Check if outlier_id matches either as int or string
                    try:
                        if (client_name == outlier_id or 
                            client_name == str(outlier_id) or 
                            str(client_name) == str(outlier_id) or
                            int(client_name) == outlier_id):
                            defenses.pca_deflect.flagged_malicious_clients.add(client_name)  # Use the actual client_name format
                            print(f"[PCA-DEFLECT] Flagged client {client_name} (outlier {outlier_id}) as malicious")
                            found = True
                            break
                    except (ValueError, TypeError):
                        # Handle case where client_name can't be converted to int
                        if (client_name == outlier_id or 
                            client_name == str(outlier_id) or 
                            str(client_name) == str(outlier_id)):
                            defenses.pca_deflect.flagged_malicious_clients.add(client_name)  # Use the actual client_name format
                            print(f"[PCA-DEFLECT] Flagged client {client_name} (outlier {outlier_id}) as malicious")
                            found = True
                            break
                
                if not found:
                    print(f"[PCA-DEFLECT] Warning: Outlier {outlier_id} not found in client_names {client_names}")
        
            
            # Apply trust factor to outlier updates instead of removing them
            epochs_submit_update_dict = defenses.pca_deflect.apply_trust_factor_to_updates(
                epochs_submit_update_dict, client_names, defenses.pca_deflect.flagged_malicious_clients
            )
        else:
            print(f"[DEBUG] PCA deflect defense NOT applied at epoch {epoch}")
            print(f"[DEBUG] Current flagged clients (should persist): {list(defenses.pca_deflect.flagged_malicious_clients)}")
        
        # Global vs Local Analysis for Trigger Detection (independent of PCA defense)
        # Check if trigger detection should be performed
        should_run_trigger_detection = (
            config.PCA_DEFLECT_RANDOM_TRIGGER_DETECTION_ON and  # Feature enabled
            helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT and  # PCA defense active
            epoch >= config.PCA_DEFLECT_MIN_ROUNDS_FOR_TRIGGER_DETECTION and  # Minimum rounds reached
            (not config.PCA_DEFLECT_TRIGGER_DETECTION_ONCE or not defenses.pca_deflect.global_vs_local_executed)  # Run once or multiple times
        )
        
        if should_run_trigger_detection:
            try:
                print(f"[PCA-DEFLECT] Starting trigger detection analysis at epoch {epoch}")
                
                # Get current client list
                current_client_names = list(epochs_submit_update_dict.keys())
                
                # Select a random client (preferably flagged) for analysis
                selected_client = defenses.pca_deflect.select_random_client_for_analysis(
                    current_client_names, defenses.pca_deflect.flagged_malicious_clients
                )
                print(f"[PCA-DEFLECT] Selected client {selected_client} for global vs local analysis")
                
                # Get the client's local model (we'll use the helper's local_model as a proxy)
                client_model = copy.deepcopy(helper.local_model)
                
                # Run global vs local analysis
                detected_trigger = defenses.pca_deflect.evaluate_globalvslocal(
                    helper.target_model, client_model, helper.test_data, config.device, 
                    epoch=epoch, client_id=selected_client
                )
                
                defenses.pca_deflect.detected_trigger_label = detected_trigger
                defenses.pca_deflect.global_vs_local_executed = True
                
                print(f"[PCA-DEFLECT] Global vs local analysis completed. Trigger label: {detected_trigger}")
                
            except Exception as e:
                print(f"[PCA-DEFLECT] Global vs local analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Set dummy trigger label for testing if not already detected
        if (helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT and 
            defenses.pca_deflect.detected_trigger_label is None):
            defenses.pca_deflect.set_dummy_trigger_label_for_testing(trigger_label=1)
            print(f"[PCA-DEFLECT] Set dummy trigger label for testing")    


        agents_num = len(agent_name_keys)
        print(f"[MAIN] Aggregating updates from {agents_num} agents")
       
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)


        # Apply aggregation method
        is_updated = True
        if helper.params['aggregation_methods'] == config.AGGR_MEAN or helper.params['aggregation_methods'] == config.AGGR_PCA_DEFLECT or helper.params['aggregation_methods'] == config.AGGR_NAB or helper.params['aggregation_methods'] == config.AGGR_NPD:
            is_updated = helper.average_shrink_models(
                weight_accumulator=weight_accumulator,
                target_model=helper.target_model,
                epoch_interval=helper.params['aggr_epoch_interval'], 
                agents_num=agents_num
            )
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(
                helper.target_model, updates, maxiter=maxiter
            )
        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)

        # NAB post-aggregation hook
        if nab_defense:
            try:
                nab_defense.post_aggregation_hook(epoch, helper.target_model)
            except Exception as e:
                print(f'[NAB] Post-aggregation hook failed: {e}')

        # Clear weight accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        # Testing phase
        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        # Test clean accuracy
        epoch_loss, epoch_acc, epoch_corret, epoch_total = test_badnet.Mytest(
            helper=helper, epoch=temp_global_epoch,
            model=helper.target_model, is_poison=False,
            visualize=True, agent_name_key="global"
        )
        
        results_all['MA'].append(epoch_acc)

        # Test backdoor attack if poisoning is enabled
        epoch_acc_p = 0.0
        if helper.params['is_poison']:
            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test_badnet.Mytest_poison(
                helper=helper,
                epoch=temp_global_epoch,
                model=helper.target_model,
                is_poison=True,
                visualize=True,
                agent_name_key="global"
            )

            results_all['BA'].append(epoch_acc_p)

            print(f"[Epoch {temp_global_epoch}] Backdoor Accuracy: "
                  f"{epoch_corret}/{epoch_total} ({epoch_acc_p:.2f}%)")
            
            # Additional trigger testing
            if len(helper.params['adversary_list']) == 1:
                if helper.params.get('centralized_test_trigger', True):
                    for j in range(0, helper.params.get('trigger_num', 1)):
                        trigger_test_byindex(helper, j, epoch)
            else:
                for agent_name_key in helper.params['adversary_list']:
                    trigger_test_byname(helper, agent_name_key, epoch)

        # NAB filtered test results
        if nab_defense and epoch >= config.NAB_WAIT_ROUNDS:
            try:
                nab_clean_loss, nab_clean_acc, nab_clean_correct, nab_clean_total = \
                    nab_defense.test_with_filtering(helper, temp_global_epoch, helper.target_model)
                print(f'[NAB Clean Test] Epoch {temp_global_epoch}: {nab_clean_correct}/{nab_clean_total} ({nab_clean_acc:.2f}%)')
                results_all['NAB_MA'].append(nab_clean_acc)
                
                if helper.params['is_poison']:
                    nab_poison_loss, nab_poison_acc, nab_poison_correct, nab_poison_total = \
                        nab_defense.test_poison_with_filtering(helper, temp_global_epoch, helper.target_model)
                    print(f'[NAB Poison Test] Epoch {temp_global_epoch}: {nab_poison_correct}/{nab_poison_total} ({nab_poison_acc:.2f}%)')
                    results_all['NAB_BA'].append(nab_poison_acc)
                else:
                    results_all['NAB_BA'].append(0.0)
            except Exception as e:
                print(f'[NAB] Test with filtering failed: {e}')
                results_all['NAB_MA'].append(0.0)
                results_all['NAB_BA'].append(0.0)
        else:
            results_all['NAB_MA'].append(0.0)
            results_all['NAB_BA'].append(0.0)

        # Print epoch summary
        print(f"\n[EPOCH {epoch} SUMMARY]")
        print(f"  Clean Accuracy: {epoch_acc:.2f}%")
        print(f"  Attack Success Rate: {epoch_acc_p:.2f}%")
        print(f"  Time: {time.time() - start_time:.2f}s")

    # Apply NPD Defense after training is complete (only after minimum rounds)
    final_epoch = helper.params['epochs']
    if npd_defense and final_epoch >= config.NPD_WAIT_ROUNDS:
        print(f"\n[NPD] Applying NPD Defense to final model after {final_epoch} epochs...")
        try:
            defended_model = npd_defense.apply_defense(helper.target_model, helper.train_data)
            if defended_model is not None:
                # Test NPD defended model
                print(f"[NPD] Testing defended model...")
                
                # Test clean accuracy
                epoch_loss, npd_clean_acc, epoch_corret, epoch_total = test_badnet.Mytest(
                    helper=helper, epoch=temp_global_epoch,
                    model=defended_model, is_poison=False,
                    visualize=False, agent_name_key="npd_global"
                )
                results_all['NPD_MA'].append(npd_clean_acc)
                print(f"[NPD] Clean Accuracy: {npd_clean_acc:.2f}%")
                
                # Test attack success rate
                npd_attack_acc = 0.0
                if helper.params['is_poison']:
                    epoch_loss, npd_attack_acc, epoch_corret, epoch_total = test_badnet.Mytest_poison(
                        helper=helper,
                        epoch=temp_global_epoch,
                        model=defended_model,
                        is_poison=True,
                        visualize=False,
                        agent_name_key="npd_global"
                    )
                    results_all['NPD_BA'].append(npd_attack_acc)
                    print(f"[NPD] Attack Success Rate: {npd_attack_acc:.2f}%")
                    
                npd_effectiveness = npd_clean_acc - npd_attack_acc
                print(f"[NPD] Defense Effectiveness: {npd_effectiveness:.2f}%")
                
        except Exception as e:
            print(f"[NPD] Defense application failed: {e}")
            results_all['NPD_MA'].append(0.0)
            results_all['NPD_BA'].append(0.0)
    elif npd_defense and final_epoch < config.NPD_WAIT_ROUNDS:
        print(f"[NPD] Skipping NPD Defense - insufficient rounds ({final_epoch} < {config.NPD_WAIT_ROUNDS})")
        results_all['NPD_MA'].append(0.0)
        results_all['NPD_BA'].append(0.0)
    else:
        results_all['NPD_MA'].append(0.0)
        results_all['NPD_BA'].append(0.0)

    # Final results summary
    print(f"\n===== FINAL RESULTS SUMMARY =====")
    if results_all['MA']:
        final_clean_acc = results_all['MA'][-1]
        avg_clean_acc = sum(results_all['MA']) / len(results_all['MA'])
        print(f"Final Clean Accuracy: {final_clean_acc:.2f}%")
        print(f"Average Clean Accuracy: {avg_clean_acc:.2f}%")
    
    if results_all['BA']:
        final_attack_acc = results_all['BA'][-1]
        avg_attack_acc = sum(results_all['BA']) / len(results_all['BA'])
        print(f"Final Attack Success Rate: {final_attack_acc:.2f}%")
        print(f"Average Attack Success Rate: {avg_attack_acc:.2f}%")
        
        if results_all['MA']:
            final_effectiveness = final_clean_acc - final_attack_acc
            print(f"Final Defense Effectiveness: {final_effectiveness:.2f}%")



    return results_all


if __name__ == '__main__':
    import sys
    
    # Default parameters
    defense_name = 'mean'
    dataset_name = 'mnist'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        defense_name = sys.argv[1]
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]
    
    print(f"Running with defense: {defense_name}, dataset: {dataset_name}")
    main_badnet(defense_name, dataset_name)