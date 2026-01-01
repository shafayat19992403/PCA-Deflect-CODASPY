import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import config
import os

def extract_features_from_model(model, dataloader, layer_name='fc2', device='cuda'):
    """Extract features from a specific layer of the model."""
    features_list = []
    labels_list = []
    
    # Hook to capture features
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0].detach() if isinstance(input, tuple) else input.detach()
        return hook
    
    # Find and register hook
    target_layer = None
    if hasattr(model, layer_name):
        target_layer = getattr(model, layer_name)
    else:
        # Try to find layer in classifier or other modules
        for name, module in model.named_modules():
            if layer_name in name or (isinstance(module, nn.Linear) and name.endswith('fc2')):
                target_layer = module
                break
    
    if target_layer is None:
        # Fallback to second-to-last linear layer
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(linear_layers) >= 2:
            target_layer = linear_layers[-2]
            layer_name = 'second_to_last'
    
    if target_layer is None:
        raise ValueError(f"Could not find suitable layer for feature extraction")
    
    handle = target_layer.register_forward_hook(get_activation(layer_name))
    
    # Extract features
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            _ = model(data)
            
            if layer_name in activation:
                features = activation[layer_name]
                features_list.append(features.cpu())
                labels_list.extend(targets.cpu().numpy())
    
    handle.remove()
    
    # Concatenate all features
    all_features = torch.cat(features_list, dim=0)
    all_labels = np.array(labels_list)
    
    return all_features.numpy(), all_labels

def cluster_features_pca(features, n_components=2, n_clusters=2):
    """Apply PCA and clustering to features."""
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_scaled)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_features[:, 0].reshape(-1, 1))
    
    # Calculate cluster statistics
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = dict(zip(unique, counts))
    
    return pca_features, cluster_labels, cluster_stats, pca

def identify_suspicious_samples(features, labels, target_label, pca_features=None, cluster_labels=None):
    """Identify suspicious samples based on clustering and other metrics."""
    # If PCA/clustering not provided, compute them
    if pca_features is None or cluster_labels is None:
        pca_features, cluster_labels, _, _ = cluster_features_pca(features)
    
    # Find samples with target label
    target_mask = labels == target_label
    target_indices = np.where(target_mask)[0]
    
    if len(target_indices) == 0:
        return set(), {}
    
    # Analyze cluster distribution for target samples
    target_clusters = cluster_labels[target_mask]
    unique, counts = np.unique(target_clusters, return_counts=True)
    cluster_dist = dict(zip(unique, counts))
    
    # Find minority cluster (likely poisoned)
    if len(cluster_dist) > 1:
        minority_cluster = min(cluster_dist, key=cluster_dist.get)
        suspicious_mask = (target_mask) & (cluster_labels == minority_cluster)
        suspicious_indices = set(np.where(suspicious_mask)[0])
    else:
        suspicious_indices = set()
    
    stats = {
        'total_target_samples': len(target_indices),
        'cluster_distribution': cluster_dist,
        'suspicious_samples': len(suspicious_indices)
    }
    
    return suspicious_indices, stats

def compute_confidence_scores(model, dataloader, device='cuda'):
    """Compute confidence scores for all predictions."""
    model.eval()
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            outputs = model(data)
            confidences = F.softmax(outputs, dim=1)
            max_conf, predictions = torch.max(confidences, dim=1)
            
            all_confidences.extend(max_conf.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(targets.numpy())
    
    return np.array(all_confidences), np.array(all_predictions), np.array(all_labels)

def compare_model_behaviors(global_model, local_model, dataloader, device='cuda'):
    """Compare predictions between global and local models."""
    global_conf, global_pred, labels = compute_confidence_scores(global_model, dataloader, device)
    local_conf, local_pred, _ = compute_confidence_scores(local_model, dataloader, device)
    
    # Find disagreements
    disagreements = global_pred != local_pred
    confidence_gaps = np.abs(global_conf - local_conf)
    
    # Identify suspicious samples
    suspicious_mask = disagreements | (confidence_gaps > config.PCA_DEFLECT_CONFIDENCE_THRESHOLD)
    suspicious_indices = set(np.where(suspicious_mask)[0])
    
    stats = {
        'total_disagreements': np.sum(disagreements),
        'avg_confidence_gap': np.mean(confidence_gaps),
        'suspicious_by_disagreement': np.sum(disagreements),
        'suspicious_by_confidence': np.sum(confidence_gaps > config.PCA_DEFLECT_CONFIDENCE_THRESHOLD)
    }
    
    return suspicious_indices, stats

def visualize_pca_clusters(pca_features, cluster_labels, suspicious_indices=None, save_path=None):
    """Visualize PCA results with clusters."""
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.6)
    
    # Highlight suspicious samples if provided
    if suspicious_indices:
        suspicious_indices = list(suspicious_indices)
        plt.scatter(pca_features[suspicious_indices, 0], 
                   pca_features[suspicious_indices, 1],
                   color='red', s=100, alpha=0.8, 
                   edgecolors='black', linewidths=2,
                   label='Suspicious Samples')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Visualization of Sample Features')
    plt.colorbar(scatter, label='Cluster')
    if suspicious_indices:
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_detection_statistics(detection_metrics, client_ids, adversary_list):
    """Calculate comprehensive detection statistics."""
    stats = {
        'per_client': {},
        'overall': {
            'total_tp': 0,
            'total_fp': 0,
            'total_triggers': 0,
            'avg_tp_rate': 0.0,
            'detected_clients': 0
        }
    }
    
    # Calculate per-client statistics
    for client_id in client_ids:
        tp = detection_metrics['true_positives'].get(client_id, 0)
        fp = detection_metrics['false_positives'].get(client_id, 0)
        total = detection_metrics['total_triggers'].get(client_id, 0)
        
        tp_rate = (tp / total * 100) if total > 0 else 0.0
        
        stats['per_client'][client_id] = {
            'true_positives': tp,
            'false_positives': fp,
            'total_triggers': total,
            'tp_rate': tp_rate,
            'is_adversary': client_id in adversary_list
        }
        
        # Update overall stats for adversaries
        if client_id in adversary_list:
            stats['overall']['total_tp'] += tp
            stats['overall']['total_fp'] += fp
            stats['overall']['total_triggers'] += total
            if total > 0:
                stats['overall']['detected_clients'] += 1
    
    # Calculate overall average TP rate
    if stats['overall']['total_triggers'] > 0:
        stats['overall']['avg_tp_rate'] = (stats['overall']['total_tp'] / 
                                           stats['overall']['total_triggers'] * 100)
    
    return stats

def save_detection_report(stats, epoch, save_dir=None):
    """Save detection statistics to file."""
    if save_dir is None:
        save_dir = config.PCA_DEFLECT_RESULTS_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    report_path = os.path.join(save_dir, f'detection_report_epoch_{epoch}.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"PCA-Deflect Detection Report - Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"  Average TP Rate: {stats['overall']['avg_tp_rate']:.2f}%\n")
        f.write(f"  Total TP: {stats['overall']['total_tp']}\n")
        f.write(f"  Total FP: {stats['overall']['total_fp']}\n")
        f.write(f"  Total Triggers: {stats['overall']['total_triggers']}\n")
        f.write(f"  Detected Adversaries: {stats['overall']['detected_clients']}\n\n")
        
        f.write("Per-Client Statistics:\n")
        for client_id, client_stats in stats['per_client'].items():
            if client_stats['is_adversary']:
                f.write(f"  Client {client_id} (Adversary):\n")
                f.write(f"    TP Rate: {client_stats['tp_rate']:.2f}%\n")
                f.write(f"    TP: {client_stats['true_positives']}, "
                       f"FP: {client_stats['false_positives']}, "
                       f"Total: {client_stats['total_triggers']}\n")
    
    print(f"[PCA-DEFLECT] Detection report saved to {report_path}")

def create_filtered_dataloader(dataset, indices_to_keep, batch_size, shuffle=False):
    """Create a DataLoader with filtered indices."""
    filtered_dataset = Subset(dataset, indices_to_keep)
    return DataLoader(filtered_dataset, batch_size=batch_size, shuffle=shuffle)

def merge_suspicious_indices(*index_sets):
    """Merge multiple sets of suspicious indices."""
    merged = set()
    for index_set in index_sets:
        merged.update(index_set)
    return merged

def calculate_cluster_purity(cluster_labels, true_labels, target_label):
    """Calculate purity of clusters with respect to target label."""
    n_clusters = len(np.unique(cluster_labels))
    purities = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            purity = np.sum(cluster_true_labels == target_label) / len(cluster_true_labels)
            purities.append(purity)
        else:
            purities.append(0.0)
    
    return purities