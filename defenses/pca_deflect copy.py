from typing import List, Tuple, Dict, Union, List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
# from cuml.decomposition import PCA
# from cuml.cluster import DBSCAN



Scalar = Union[bool, bytes, float, int, str, List[int]]
Metrics = Dict[str, Scalar]
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define the function to extract client model weights and flatten them
# def extract_client_weights(client_models):
#     client_weights = []
#     for client_model in client_models:  # list of `Parameters` objects
#         weights = parameters_to_ndarrays(client_model)  # Convert Parameters to ndarray
#         flat_weights = np.concatenate([w.flatten() for w in weights])  # Flatten the weights
#         client_weights.append(flat_weights)
#     return client_weights

# def extract_client_weights(state_dicts: List[Dict[str,torch.Tensor]]):
#      """Turn a list of state_dicts into a list of 1D numpy vectors."""
#      flat_updates = []
#      for sd in state_dicts:
#          parts = []
#          for name, tensor in sd.items():
#              # skip tied embeddings or any buffers you don’t want to include
#              if 'decoder.weight' in name or '__' in name:
#                  continue
#              parts.append((tensor).flatten().cpu().numpy())
#          flat_updates.append(np.concatenate(parts, axis=0))
#      return flat_updates

def extract_client_weights(state_dicts: List[Dict[str, torch.Tensor]]):
    """Turn a list of state_dicts into normalized 1D numpy vectors."""
    flat_updates = []
    for sd in state_dicts:
        parts = []
        for name, tensor in sd.items():
            # skip tied embeddings or any buffers you don’t want to include
            if 'decoder.weight' in name or '__' in name:
                continue
            parts.append((tensor).flatten().cpu().numpy())
        flat_update = np.concatenate(parts, axis=0)

        # Normalize the update to unit length to make PCA/DBSCAN robust to non-IID scale differences
        # norm = np.linalg.norm(flat_update) + 1e-8  # avoid division by zero
        # flat_update = flat_update / norm

        flat_updates.append(flat_update)
    return flat_updates




# def apply_pca_to_weights(client_weights, client_ids, rnd, flagged_malicious_clients):
#     # client_weights is already numpy arrays here
   

#     pca = PCA(n_components=2)
#     # print(client_weights)
#     reduced_weights = pca.fit_transform(client_weights)

#     # Extract PC1 values and reshape for clustering
#     pc1_values = reduced_weights[:, 0].reshape(-1, 1)
    
#     # Dynamic epsilon based on previous detections
#     # eps_value = 1.2 if len(flagged_malicious_clients) > 0 else 1
#     eps_value = 0.3
    
#     # DBSCAN clustering
#     dbscan = DBSCAN(eps=eps_value, min_samples=2)
#     cluster_labels = dbscan.fit_predict(pc1_values)
#     # distance_matrix = cosine_distances(pc1_values)
#     # cluster_labels = dbscan.fit_predict(distance_matrix)
    
#     # Find outliers (malicious clients)
#     label_counts = Counter(cluster_labels)
#     outliers = []
#     print(label_counts, eps_value)
    
#     if len(label_counts) > 1:
#         # smallest_cluster_size = min(label_counts.values())
#         # outlier_labels = [label for label, count in label_counts.items() 
#         #                  if count == smallest_cluster_size]
#         # outliers = [client_ids[i] for i, label in enumerate(cluster_labels) 
#         #           if label in outlier_labels]
#         largest_cluster_size = max(label_counts.values())
#         outlier_labels = [label for label, count in label_counts.items() if count == largest_cluster_size]
#         outliers = [client_ids[i] for i,label in enumerate(cluster_labels) if label not in outlier_labels]
#     else:
#         outlier_labels = []
#         outliers = []

#     # Visualization (optional)
#     plt.figure(figsize=(10, 6))
#     plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], c=cluster_labels, cmap='viridis')
    
#     if outliers:
#         outlier_indices = [client_ids.index(client_id) for client_id in outliers]
#         plt.scatter(reduced_weights[outlier_indices, 0], 
#                    reduced_weights[outlier_indices, 1], 
#                    color='red', marker='x', s=100)
    
#     plt.title(f"Round {rnd}: PCA of Client Weights (Outliers: {outliers})")
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.colorbar(label='Cluster')
#     plt.savefig(f"pca_round_{rnd}.png")
#     plt.close()

#     return outliers, []



def apply_pca_to_weights(client_weights, client_ids, rnd, flagged_malicious_clients):
    # client_weights is already numpy arrays here
    pca = PCA(n_components=2)
    reduced_weights = pca.fit_transform(client_weights)

    # Extract PC1 values and reshape for clustering
    pc1_values = reduced_weights[:, 0].reshape(-1, 1)
    
    # Dynamic epsilon based on previous detections
    # eps_value = 0.05 #-->Cerp
    # eps_value = 100 #-->DBA CAS
    eps_value = 5 #-->Badnet
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps_value, min_samples=2)
    cluster_labels = dbscan.fit_predict(pc1_values)
    # distance_matrix = cosine_distances(pc1_values)
    # cluster_labels = dbscan.fit_predict(distance_matrix)
    
    # Find outliers (malicious clients)
    label_counts = Counter(cluster_labels)
    outliers = []
    print(label_counts, eps_value)
    
    if len(label_counts) > 1:
        smallest_cluster_size = min(label_counts.values())
        outlier_labels = [label for label, count in label_counts.items() 
                         if count == smallest_cluster_size]
        outliers = [client_ids[i] for i, label in enumerate(cluster_labels) 
                  if label in outlier_labels]
    else:
        outlier_labels = []
        outliers = []

    # Visualization (optional)
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
    plt.savefig(f"pca_round_{rnd}.png")
    plt.close()

    return outliers, []
