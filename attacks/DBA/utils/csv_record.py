import csv
import copy
train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy", "correct_data",
                    "total_data"]
test_fileHeader = ["model", "epoch", "average_loss", "accuracy", "correct_data", "total_data"]
train_result = []  # train_fileHeader
test_result = []  # test_fileHeader
posiontest_result = []  # test_fileHeader

triggertest_fileHeader = ["model", "trigger_name", "trigger_value", "epoch", "average_loss", "accuracy", "correct_data",
                          "total_data"]
poisontriggertest_result = []  # triggertest_fileHeader

posion_test_result = []  # train_fileHeader
posion_posiontest_result = []  # train_fileHeader
weight_result=[]
scale_result=[]
scale_temp_one_row=[]

# PCA-Deflect specific tracking
pca_deflect_detection = []  # Track detection metrics per epoch
pca_deflect_fileHeader = ["epoch", "client_id", "true_positives", "false_positives", 
                          "total_triggers", "tp_rate", "is_malicious", "is_flagged"]
pca_deflect_detection_result = []

pca_deflect_summary_fileHeader = ["epoch", "avg_tp_rate", "total_tp", "total_triggers", 
                                  "num_flagged_malicious", "num_flagged_benign"]
pca_deflect_summary = []

def save_result_csv(epoch, is_posion, folder_path):
    train_csvFile = open(f'{folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csvFile)
    train_writer.writerow(train_fileHeader)
    train_writer.writerows(train_result)
    train_csvFile.close()

    test_csvFile = open(f'{folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_result)
    test_csvFile.close()

    if len(weight_result)>0:
        weight_csvFile=  open(f'{folder_path}/weight_result.csv', "w")
        weight_writer = csv.writer(weight_csvFile)
        weight_writer.writerows(weight_result)
        weight_csvFile.close()

    if len(scale_temp_one_row)>0:
        _csvFile=  open(f'{folder_path}/scale_result.csv', "w")
        _writer = csv.writer(_csvFile)
        scale_result.append(copy.deepcopy(scale_temp_one_row))
        scale_temp_one_row.clear()
        _writer.writerows(scale_result)
        _csvFile.close()

    if is_posion:
        test_csvFile = open(f'{folder_path}/posiontest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()

        test_csvFile = open(f'{folder_path}/poisontriggertest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(triggertest_fileHeader)
        test_writer.writerows(poisontriggertest_result)
        test_csvFile.close()
    
    # Save PCA-Deflect detection results
    if len(pca_deflect_detection_result) > 0:
        pca_csvFile = open(f'{folder_path}/pca_deflect_detection.csv', "w")
        pca_writer = csv.writer(pca_csvFile)
        pca_writer.writerow(pca_deflect_fileHeader)
        pca_writer.writerows(pca_deflect_detection_result)
        pca_csvFile.close()
    
    if len(pca_deflect_summary) > 0:
        summary_csvFile = open(f'{folder_path}/pca_deflect_summary.csv', "w")
        summary_writer = csv.writer(summary_csvFile)
        summary_writer.writerow(pca_deflect_summary_fileHeader)
        summary_writer.writerows(pca_deflect_summary)
        summary_csvFile.close()

def add_weight_result(name, weight, alpha):
    weight_result.append(name)
    weight_result.append(weight)
    weight_result.append(alpha)

def add_pca_deflect_detection(epoch, client_id, true_positives, false_positives, 
                              total_triggers, is_malicious, is_flagged):
    """Add PCA-Deflect detection result for a client."""
    tp_rate = (true_positives / total_triggers * 100) if total_triggers > 0 else 0.0
    pca_deflect_detection_result.append([
        epoch, client_id, true_positives, false_positives, 
        total_triggers, round(tp_rate, 2), is_malicious, is_flagged
    ])

def add_pca_deflect_summary(epoch, detection_metrics, flagged_clients, adversary_list):
    """Add PCA-Deflect summary for an epoch."""
    # Calculate overall statistics
    total_tp = 0
    total_triggers = 0
    num_flagged_malicious = 0
    num_flagged_benign = 0
    
    for client_id in flagged_clients:
        if client_id in adversary_list:
            num_flagged_malicious += 1
            total_tp += detection_metrics.get('true_positives', {}).get(client_id, 0)
            total_triggers += detection_metrics.get('total_triggers', {}).get(client_id, 0)
        else:
            num_flagged_benign += 1
    
    avg_tp_rate = (total_tp / total_triggers * 100) if total_triggers > 0 else 0.0
    
    pca_deflect_summary.append([
        epoch, round(avg_tp_rate, 2), total_tp, total_triggers,
        num_flagged_malicious, num_flagged_benign
    ])

def clear_pca_deflect_epoch_data():
    """Clear PCA-Deflect data for current epoch (if needed)."""
    global pca_deflect_detection
    pca_deflect_detection.clear()