import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from collections import Counter
from torch.utils.data import DataLoader, Subset
import config
import copy
import random

# Global detection metrics tracking
detection_metrics = {
    'true_positives': {},   # client_id: count
    'false_positives': {},  # client_id: count
    'total_triggers': {}    # client_id: count
}

# Global state for PCA-Deflect defense
flagged_malicious_clients = set()  # Track clients flagged as outliers
detected_trigger_label = None      # Store detected trigger label
global_vs_local_executed = False  # Track if global vs local analysis has been done
client_clean_models = {}           # Store clean models for each client
client_poison_models = {}          # Store poison models for each client

def manually_flag_clients_for_testing(client_ids):
    """Manually flag specific clients for testing purposes."""
    global flagged_malicious_clients
    for client_id in client_ids:
        flagged_malicious_clients.add(client_id)
        print(f"[PCA-DEFLECT TEST] Manually flagged client {client_id} as malicious")

def set_dummy_trigger_label_for_testing(trigger_label=1):
    """Set a dummy trigger label for testing purposes."""
    global detected_trigger_label
    detected_trigger_label = trigger_label
    print(f"[PCA-DEFLECT TEST] Set dummy trigger label to {trigger_label}")

def extract_client_weights(client_weights):
    """Extract and flatten client model weights."""
    client_weights_flat = []
    for weights in client_weights:
        if isinstance(weights, dict):
            # If weights is a dict of tensors
            flat_weights = torch.cat([w.flatten() for w in weights.values()])
        else:
            # If weights is already a tensor
            flat_weights = weights.flatten() if len(weights.shape) > 1 else weights
        client_weights_flat.append(flat_weights)
    return client_weights_flat

def apply_pca_to_weights(client_weights, client_ids, rnd, flagged_malicious_clients_arg):
    """Apply PCA-based outlier detection on client weights."""
    global flagged_malicious_clients
    
    # Convert to numpy if needed
    client_weights_np = []
    for w in client_weights:
        if torch.is_tensor(w):
            client_weights_np.append(w.cpu().numpy())
        else:
            client_weights_np.append(w)
    
    # Apply PCA
    pca = PCA(n_components=2)
    reduced_weights = pca.fit_transform(client_weights_np)
    
    # Extract PC1 values and normalize
    pc1_values = reduced_weights[:, 0]
    # pc1_values = (pc1_values - np.min(pc1_values)) / (np.max(pc1_values) - np.min(pc1_values))
    pc1_values = pc1_values.reshape(-1, 1)
    
    # Apply DBSCAN with adaptive epsilon
    if len(flagged_malicious_clients) > 0:
        eps_value = config.PCA_DEFLECT_DBSCAN_EPS_FLAGGED
    else:
        eps_value = config.PCA_DEFLECT_DBSCAN_EPS_NORMAL
    
    dbscan = DBSCAN(eps=eps_value, min_samples=config.PCA_DEFLECT_MIN_SAMPLES)
    cluster_labels = dbscan.fit_predict(pc1_values)
    
    # Identify outliers (smallest clusters)
    label_counts = Counter(cluster_labels)
    outliers = []
    
    if len(label_counts) > 1:
        smallest_cluster_size = min(label_counts.values())
        outlier_labels = [label for label, count in label_counts.items() 
                         if count == smallest_cluster_size]
        outliers = [client_ids[i] for i, label in enumerate(cluster_labels) 
                   if label in outlier_labels]
    
    # Add outliers to flagged clients set
    for outlier in outliers:
        flagged_malicious_clients.add(outlier)
    
    print(f"[PCA-DEFLECT] Round {rnd}: Detected outliers: {outliers}")
    print(f"[PCA-DEFLECT] Round {rnd}: Total flagged clients: {list(flagged_malicious_clients)}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], c=cluster_labels, cmap='viridis')
    
    if outliers:
        outlier_indices = [client_ids.index(client_id) for client_id in outliers]
        plt.scatter(reduced_weights[outlier_indices, 0], 
                   reduced_weights[outlier_indices, 1], 
                   color='red', marker='x', s=100)
    
    plt.title(f"Round {rnd}: PCA of Client Weights (Outliers: {outliers})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster')
    plt.savefig(f"Figures/pca_round_{rnd}.png")
    plt.close()
    
    # Extract most important weights (for future use)
    most_important_weights = []
    
    return outliers, most_important_weights

def evaluate_globalvslocal(global_model, contaminated_client_model, test_dataset, device, epoch=None, client_id=None):
    """
    Detect trigger label by comparing pure global model vs contaminated client model.
    
    The logic: 
    - global_model: Pure/clean aggregated model (through trust factor application)
    - contaminated_client_model: Client model that has learned backdoor triggers
    - Differences in False Positive Rates reveal which label is the backdoor target
    """
    print(f"[PCA-DEFLECT] Evaluating pure global model vs contaminated client model")
    print(f"[PCA-DEFLECT] Epoch {epoch}, Client {client_id}")
    print(f"[PCA-DEFLECT] This will reveal backdoor trigger through FPR differences")
    
    # Check if test_dataset is already a DataLoader or needs to be wrapped
    if isinstance(test_dataset, DataLoader):
        test_loader = test_dataset
    else:
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    global_model.eval()
    contaminated_client_model.eval()
    
    # Get predictions
    global_predictions = []
    contaminated_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Global model predictions (pure/clean)
            global_outputs = global_model(images)
            global_preds = global_outputs.max(1)[1]
            global_predictions.extend(global_preds.cpu().numpy())
            
            # Contaminated client model predictions (has learned backdoor)
            contaminated_outputs = contaminated_client_model(images)
            contaminated_preds = contaminated_outputs.max(1)[1]
            contaminated_predictions.extend(contaminated_preds.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())
    
    global_predictions = np.array(global_predictions)
    contaminated_predictions = np.array(contaminated_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate FPR and FNR for each class
    num_classes = len(np.unique(all_labels))
    global_fpr = []
    contaminated_fpr = []
    global_fnr = []
    contaminated_fnr = []
    
    for label in range(num_classes):
        # Global FPR (pure model)
        global_fp = ((global_predictions == label) & (all_labels != label)).sum()
        global_tn = ((global_predictions != label) & (all_labels != label)).sum()
        global_fpr.append(global_fp / (global_fp + global_tn) if (global_fp + global_tn) > 0 else 0)
        
        # Global FNR (pure model)
        global_fn = ((global_predictions != label) & (all_labels == label)).sum()
        global_tp = ((global_predictions == label) & (all_labels == label)).sum()
        global_fnr.append(global_fn / (global_fn + global_tp) if (global_fn + global_tp) > 0 else 0)
        
        # Contaminated FPR (backdoored model)
        contaminated_fp = ((contaminated_predictions == label) & (all_labels != label)).sum()
        contaminated_tn = ((contaminated_predictions != label) & (all_labels != label)).sum()
        contaminated_fpr.append(contaminated_fp / (contaminated_fp + contaminated_tn) if (contaminated_fp + contaminated_tn) > 0 else 0)
        
        # Contaminated FNR (backdoored model)
        contaminated_fn = ((contaminated_predictions != label) & (all_labels == label)).sum()
        contaminated_tp = ((contaminated_predictions == label) & (all_labels == label)).sum()
        contaminated_fnr.append(contaminated_fn / (contaminated_fn + contaminated_tp) if (contaminated_fn + contaminated_tp) > 0 else 0)
    
    # Find label with maximum FPR difference
    # High difference means contaminated model has much higher FPR for that label (backdoor target)
    fpr_diff = abs(np.array(contaminated_fpr) - np.array(global_fpr))
    fnr_diff = np.array(contaminated_fnr) - np.array(global_fnr)

    # Identify trigger label as the one with highest FPR difference
    max_fpr_label = np.argmax(fpr_diff)
    max_fnr_label = np.argmax(fnr_diff)
    print(f"[PCA-DEFLECT] Max FPR diff label: {max_fpr_label}, FPR diff: {fpr_diff[max_fpr_label]:.4f}")
    print(f"[PCA-DEFLECT] Max FNR diff label: {max_fnr_label}, FNR diff: {fnr_diff[max_fnr_label]:.4f}")
    print(f"[PCA-DEFLECT] Contaminated model shows higher FPR for label {max_fpr_label} (likely trigger)")
    
    # Create point plots for FPR and FNR comparisons
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    labels_range = list(range(num_classes))
    plt.plot(labels_range, global_fpr, 'bo-', label='Global FPR (Pure)', markersize=8, linewidth=2)
    plt.plot(labels_range, contaminated_fpr, 'ro-', label='Malicious FPR (Backdoored)', markersize=8, linewidth=2)
    plt.xlabel('Class Label')
    plt.ylabel('False Positive Rate')
    plt.title(f'FPR Comparison: Pure vs Malicious Models\nEpoch {epoch}, Client {client_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(labels_range)
    
    plt.subplot(1, 2, 2)
    plt.plot(labels_range, global_fnr, 'bo-', label='Global FNR (Pure)', markersize=8, linewidth=2)
    plt.plot(labels_range, contaminated_fnr, 'ro-', label='Malicious FNR (Backdoored)', markersize=8, linewidth=2)
    plt.xlabel('Class Label')
    plt.ylabel('False Negative Rate')
    plt.title(f'FNR Comparison: Pure vs Malicious Models\nEpoch {epoch}, Client {client_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(labels_range)
    
    plt.tight_layout()
    
    # Save with epoch and client information
    save_filename = f"Figures/contaminated_vs_pure_FPR_FNR_epoch_{epoch}_client_{client_id}.png"
    plt.savefig(save_filename)
    plt.close()
    
    print(f"[PCA-DEFLECT] Pure Global Model FPR: {global_fpr}")  
    print(f"[PCA-DEFLECT] Contaminated Model FPR: {contaminated_fpr}")
    print(f"[PCA-DEFLECT] Pure Global Model FNR: {global_fnr}")
    print(f"[PCA-DEFLECT] Contaminated Model FNR: {contaminated_fnr}")

    # Find label with maximum FPR difference
    trigger_label = np.argmax(fpr_diff)
    
    print(f"[PCA-DEFLECT] Detected trigger label: {trigger_label} (FPR diff: {fpr_diff[trigger_label]:.4f})")
    
    return trigger_label

def apply_trust_factor_to_updates(epochs_submit_update_dict, client_ids, flagged_clients):
    """
    Apply trust factor to flagged clients' updates instead of removing them.
    """
    trust_factor = config.PCA_DEFLECT_TRUST_FACTOR
    modified_clients = []
    
    for client_id in client_ids:
        if client_id in flagged_clients:
            print(f"[PCA-DEFLECT] Applying trust factor {trust_factor} to client {client_id}")
            # Apply trust factor to all updates from this client
            client_updates = epochs_submit_update_dict[client_id]
            for update_dict in client_updates:
                for param_name in update_dict:
                    if torch.is_tensor(update_dict[param_name]):
                        update_dict[param_name] = update_dict[param_name] * trust_factor
                    elif isinstance(update_dict[param_name], (list, np.ndarray)):
                        # Handle numpy arrays or lists
                        if isinstance(update_dict[param_name], list):
                            # Convert list to numpy for easier manipulation
                            arr = np.array(update_dict[param_name])
                            update_dict[param_name] = (arr * trust_factor).tolist()
                        else:
                            update_dict[param_name] = update_dict[param_name] * trust_factor
                    else:
                        # Handle scalar values
                        update_dict[param_name] = update_dict[param_name] * trust_factor
            modified_clients.append(client_id)
    
    if modified_clients:
        print(f"[PCA-DEFLECT] Applied trust factor to clients: {modified_clients}")
    
    return epochs_submit_update_dict

def select_random_client_for_analysis(client_ids, flagged_clients):
    """
    Select a random flagged client for global vs local analysis.
    If no flagged clients, select any random client.
    """
    import random
    
    if flagged_clients:
        # Prefer flagged clients for analysis
        selected_client = random.choice(list(flagged_clients))
        print(f"[PCA-DEFLECT] Selected flagged client {selected_client} for global vs local analysis")
    else:
        # Fallback to any random client
        selected_client = random.choice(client_ids)
        print(f"[PCA-DEFLECT] Selected random client {selected_client} for global vs local analysis")
    
    return selected_client

def filter_poisoned_data(client_id, sampled_data, sampled_indices, global_model, 
                        contaminated_client_model, trigger_label, triggered_indices, device):
    """Filter potentially poisoned data using PCA and confidence analysis."""
    print(f"[PCA-DEFLECT] Filtering data for client {client_id}")
    
    # Create data loader
    sampled_loader = DataLoader(sampled_data, batch_size=config.PCA_DEFLECT_FILTER_BATCH_SIZE, 
                               shuffle=False)
    
    # Find samples with target label
    targeted_images = []
    targeted_labels = []
    original_indices_local = []
    
    current_idx = 0
    for batch_idx, (images, labels) in enumerate(sampled_loader):
        mask = labels == trigger_label
        if mask.any():
            targeted_images.append(images[mask])
            targeted_labels.extend(labels[mask].tolist())
            
            # Track indices
            batch_size = len(labels)
            for i in range(batch_size):
                if mask[i]:
                    original_indices_local.append(current_idx + i)
        current_idx += len(labels)
    
    if len(targeted_images) == 0:
        print(f"[PCA-DEFLECT] No samples with trigger label {trigger_label} found")
        return sampled_data, [], 0, 0
    
    # Combine targeted images
    targeted_images = torch.cat(targeted_images, dim=0).to(device)
    print(f"[PCA-DEFLECT] Found {len(targeted_images)} samples with trigger label")
    
    # Extract features using hook
    features_list = []
    
    def hook_fn(module, input, output):
        features_list.append(input[0].detach())
    
    # Register hook based on model architecture
    if hasattr(contaminated_client_model, 'fc2'):
        handle = contaminated_client_model.fc2.register_forward_hook(hook_fn)
    elif hasattr(contaminated_client_model, 'classifier'):
        # For models with classifier module
        handle = contaminated_client_model.classifier[-2].register_forward_hook(hook_fn)
    else:
        # Default to second-to-last layer
        modules = list(contaminated_client_model.modules())
        fc_layers = [m for m in modules if isinstance(m, nn.Linear)]
        if len(fc_layers) >= 2:
            handle = fc_layers[-2].register_forward_hook(hook_fn)
        else:
            print(f"[PCA-DEFLECT] Warning: Could not find suitable layer for feature extraction")
            return sampled_data, [], 0, 0
    
    # Extract features
    contaminated_client_model.eval()
    with torch.no_grad():
        for i in range(0, len(targeted_images), config.PCA_DEFLECT_FILTER_BATCH_SIZE):
            batch = targeted_images[i:i + config.PCA_DEFLECT_FILTER_BATCH_SIZE]
            _ = contaminated_client_model(batch)
    
    handle.remove()
    
    # Process features
    features = torch.cat(features_list, dim=0)
    flattened_features = features.view(features.size(0), -1).cpu().numpy()
    
    # Apply PCA and clustering
    pca = PCA(n_components=config.PCA_DEFLECT_PCA_COMPONENTS)
    pca_result = pca.fit_transform(flattened_features)
    pc1_values = pca_result[:, 0].reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=config.PCA_DEFLECT_KMEANS_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(pc1_values)
    
    # Find larger cluster (assumed clean)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    
    # Get indices to exclude (smaller cluster)
    smallest_cluster_indices = np.where(cluster_labels != largest_cluster_label)[0]
    indices_to_exclude = set([original_indices_local[i] for i in smallest_cluster_indices])
    
    # Add confidence-based filtering
    contaminated_client_model.eval()
    global_model.eval()
    
    with torch.no_grad():
        for i in range(0, len(targeted_images), config.PCA_DEFLECT_FILTER_BATCH_SIZE):
            batch = targeted_images[i:i + config.PCA_DEFLECT_FILTER_BATCH_SIZE]
            
            # Get predictions
            contaminated_outputs = contaminated_client_model(batch)
            contaminated_confidences = F.softmax(contaminated_outputs, dim=1)
            contaminated_max_conf, contaminated_preds = torch.max(contaminated_confidences, dim=1)
            
            global_outputs = global_model(batch)
            global_confidences = F.softmax(global_outputs, dim=1)
            global_max_conf, global_preds = torch.max(global_confidences, dim=1)
            
            # Filter based on prediction differences
            for j in range(len(batch)):
                idx = i + j
                if idx < len(original_indices_local):
                    local_idx = original_indices_local[idx]
                    
                    # Different predictions or confidence gap
                    if (contaminated_preds[j] != global_preds[j] or
                        torch.abs(global_max_conf[j] - contaminated_max_conf[j]) > config.PCA_DEFLECT_CONFIDENCE_THRESHOLD):
                        indices_to_exclude.add(local_idx)
    
    # Calculate detection metrics
    true_positives = 0
    false_positives = 0
    
    for idx in indices_to_exclude:
        global_idx = sampled_indices[idx]
        if global_idx in triggered_indices:
            true_positives += 1
        else:
            false_positives += 1
    
    total_triggers = len(set(sampled_indices).intersection(triggered_indices))
    
    # Update global metrics
    detection_metrics['true_positives'][client_id] = true_positives
    detection_metrics['false_positives'][client_id] = false_positives
    detection_metrics['total_triggers'][client_id] = total_triggers
    
    print(f"[PCA-DEFLECT] Client {client_id}: TP={true_positives}, FP={false_positives}, Total triggers={total_triggers}")
    
    # Create filtered dataset
    keep_indices = [i for i in range(len(sampled_data)) if i not in indices_to_exclude]
    filtered_data = Subset(sampled_data, keep_indices)
    poisoned_data = Subset(sampled_data, list(indices_to_exclude))
    
    return filtered_data, poisoned_data, true_positives, false_positives

def comprehensive_poison_filter(client_id, data_loader, global_model, local_model, device, epoch):
    """
    Comprehensive poison data filtering using PCA and confidence analysis.
    
    Args:
        client_id: ID of the client
        data_loader: DataLoader with client's data
        global_model: Clean global model
        local_model: Client's local model (potentially contaminated)
        device: torch device
        epoch: current epoch
        
    Returns:
        filtered_indices: indices of clean samples
        poison_indices: indices of detected poison samples
        detection_stats: detection statistics
    """
    global detected_trigger_label
    
    if detected_trigger_label is None:
        print(f"[PCA-DEFLECT] No trigger label detected yet, skipping filtering for client {client_id}")
        return list(range(len(data_loader.dataset))), [], {}
    
    print(f"[PCA-DEFLECT] Starting comprehensive poison filtering for client {client_id}")
    print(f"[PCA-DEFLECT] Using trigger label: {detected_trigger_label}")
    
    # Extract features and collect data with trigger label
    global_model.eval()
    local_model.eval()
    
    target_samples = []
    target_indices = []
    all_samples = []
    all_indices = []
    
    # Collect samples with trigger label
    current_idx = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        for i, label in enumerate(labels):
            if label.item() == detected_trigger_label:
                target_samples.append(images[i])
                target_indices.append(current_idx + i)
            all_samples.append(images[i])
            all_indices.append(current_idx + i)
        current_idx += len(labels)
    
    if len(target_samples) == 0:
        print(f"[PCA-DEFLECT] No samples with trigger label {detected_trigger_label} found")
        return all_indices, [], {}
    
    print(f"[PCA-DEFLECT] Found {len(target_samples)} samples with trigger label {detected_trigger_label}")
    
    # Convert to tensors
    target_images = torch.stack(target_samples).to(device)
    
    # Extract features from FC2 layer
    features_list = []
    
    def hook_fn(module, input, output):
        features_list.append(input[0].detach())
    
    # Register hook on FC2 layer
    handle = None
    if hasattr(local_model, 'fc2'):
        handle = local_model.fc2.register_forward_hook(hook_fn)
    elif hasattr(local_model, 'classifier'):
        # For models with classifier module
        modules = list(local_model.classifier.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        if len(linear_layers) >= 2:
            handle = linear_layers[-2].register_forward_hook(hook_fn)
    else:
        # Default to second-to-last linear layer
        modules = list(local_model.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        if len(linear_layers) >= 2:
            handle = linear_layers[-2].register_forward_hook(hook_fn)
    
    if handle is None:
        print(f"[PCA-DEFLECT] Warning: Could not find suitable layer for feature extraction")
        return all_indices, [], {}
    
    # Extract features
    with torch.no_grad():
        for i in range(0, len(target_images), config.PCA_DEFLECT_FILTER_BATCH_SIZE):
            batch = target_images[i:i + config.PCA_DEFLECT_FILTER_BATCH_SIZE]
            _ = local_model(batch)
    
    handle.remove()
    
    if len(features_list) == 0:
        print(f"[PCA-DEFLECT] No features extracted")
        return all_indices, [], {}
    
    # Process features
    features = torch.cat(features_list, dim=0)
    flattened_features = features.view(features.size(0), -1).cpu().numpy()
    
    # Apply PCA and clustering
    pca = PCA(n_components=config.PCA_DEFLECT_PCA_COMPONENTS)
    pca_result = pca.fit_transform(flattened_features)
    
    # K-means clustering to identify poison vs clean clusters
    kmeans = KMeans(n_clusters=config.PCA_DEFLECT_KMEANS_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_result)
    
    # Identify smaller cluster (assumed poisonous)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    smallest_cluster_label = min(cluster_sizes, key=cluster_sizes.get)
    
    # Get indices of samples in smaller cluster (poison candidates)
    poison_candidates = []
    for i, label in enumerate(cluster_labels):
        if label == smallest_cluster_label:
            poison_candidates.append(target_indices[i])
    
    print(f"[PCA-DEFLECT] PCA clustering identified {len(poison_candidates)} poison candidates from smaller cluster")
    
    # Apply confidence-based filtering
    confidence_poison_indices = set()
    
    with torch.no_grad():
        for i in range(0, len(target_images), config.PCA_DEFLECT_FILTER_BATCH_SIZE):
            batch = target_images[i:i + config.PCA_DEFLECT_FILTER_BATCH_SIZE]
            
            # Get predictions from both models
            local_outputs = local_model(batch)
            local_confidences = F.softmax(local_outputs, dim=1)
            local_max_conf, local_preds = torch.max(local_confidences, dim=1)
            
            global_outputs = global_model(batch)
            global_confidences = F.softmax(global_outputs, dim=1)
            global_max_conf, global_preds = torch.max(global_confidences, dim=1)
            
            # Check for significant confidence differences
            for j in range(len(batch)):
                idx = i + j
                if idx < len(target_indices):
                    target_idx = target_indices[idx]
                    
                    # Different predictions or large confidence gap indicates poison
                    confidence_diff = torch.abs(global_max_conf[j] - local_max_conf[j])
                    prediction_diff = (local_preds[j] != global_preds[j])
                    
                    if (confidence_diff > config.PCA_DEFLECT_CONFIDENCE_FILTER_THRESHOLD or prediction_diff):
                        confidence_poison_indices.add(target_idx)
    
    print(f"[PCA-DEFLECT] Confidence filtering identified {len(confidence_poison_indices)} poison candidates")
    
    # Combine PCA and confidence-based detection
    final_poison_indices = set(poison_candidates).union(confidence_poison_indices)
    final_clean_indices = [idx for idx in all_indices if idx not in final_poison_indices]
    
    print(f"[PCA-DEFLECT] Final filtering results:")
    print(f"[PCA-DEFLECT] - Total samples: {len(all_indices)}")
    print(f"[PCA-DEFLECT] - Clean samples: {len(final_clean_indices)}")
    print(f"[PCA-DEFLECT] - Poison samples: {len(final_poison_indices)}")
    
    # Calculate detection statistics
    detection_stats = {
        'total_samples': len(all_indices),
        'clean_samples': len(final_clean_indices),
        'poison_samples': len(final_poison_indices),
        'trigger_label_samples': len(target_samples),
        'pca_poison_candidates': len(poison_candidates),
        'confidence_poison_candidates': len(confidence_poison_indices)
    }
    
    return final_clean_indices, list(final_poison_indices), detection_stats

def is_client_flagged(client_id):
    """Check if a client is flagged as malicious."""
    global flagged_malicious_clients
    
    # Check for exact match first
    if client_id in flagged_malicious_clients:
        return True
    
    # Try string/int conversions for flexible matching
    for flagged_client in flagged_malicious_clients:
        if (str(client_id) == str(flagged_client) or 
            client_id == flagged_client):
            return True
    
    return False

def should_apply_filtering(client_id, epoch):
    """Check if filtering should be applied for this client."""
    return (config.PCA_DEFLECT_ENABLE_FILTERING and 
            config.PCA_DEFLECT_ENABLE_DUAL_TRAINING and
            is_client_flagged(client_id) and 
            detected_trigger_label is not None and
            epoch >= config.PCA_DEFLECT_MIN_ROUNDS_FOR_TRIGGER_DETECTION)

def store_client_models(client_id, clean_model, poison_model):
    """Store clean and poison models for a client."""
    global client_clean_models, client_poison_models
    
    if clean_model is not None:
        client_clean_models[client_id] = copy.deepcopy(clean_model)
        print(f"[PCA-DEFLECT] Stored clean model for client {client_id}")
    
    if poison_model is not None:
        client_poison_models[client_id] = copy.deepcopy(poison_model)
        print(f"[PCA-DEFLECT] Stored poison model for client {client_id}")

def get_client_models(client_id):
    """Get stored clean and poison models for a client."""
    global client_clean_models, client_poison_models
    
    clean_model = client_clean_models.get(client_id, None)
    poison_model = client_poison_models.get(client_id, None)
    
    return clean_model, poison_model

def create_filtered_datasets(original_dataset, clean_indices, poison_indices):
    """Create separate datasets for clean and poison data."""
    from torch.utils.data import Subset
    
    clean_dataset = Subset(original_dataset, clean_indices) if clean_indices else None
    poison_dataset = Subset(original_dataset, poison_indices) if poison_indices else None
    
    return clean_dataset, poison_dataset

def calculate_average_tp_percentage(malicious_clients):
    """Calculate average true positive percentage across malicious clients."""
    total_tp = 0
    total_triggers = 0
    
    for client_id in malicious_clients:
        total_tp += detection_metrics['true_positives'].get(client_id, 0)
        total_triggers += detection_metrics['total_triggers'].get(client_id, 0)
    
    if total_triggers > 0:
        return (total_tp / total_triggers) * 100
    return 0.0

def get_flagged_clients():
    """Get currently flagged malicious clients."""
    global flagged_malicious_clients
    return list(flagged_malicious_clients)

def reset_detection_state():
    """Reset detection state for new run."""
    global flagged_malicious_clients, detected_trigger_label, global_vs_local_executed
    flagged_malicious_clients.clear()
    detected_trigger_label = None
    global_vs_local_executed = False
    reset_detection_metrics()

def reset_detection_metrics():
    """Reset detection metrics for new epoch."""
    detection_metrics['true_positives'].clear()
    detection_metrics['false_positives'].clear()
    detection_metrics['total_triggers'].clear()