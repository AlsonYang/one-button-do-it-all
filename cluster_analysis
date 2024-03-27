"""
  - Entrypoint: `clustering_analysis`
  - Special thanks: ChatGPT
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime

def timeit(fn):
    def wrapper(*args, **kwargs):
        t1 = datetime.now()
        ret = fn(*args, **kwargs)
        print(f"it took {fn.__name__} {(datetime.now()-t1).seconds} seconds to run")
        return ret
    return wrapper

@timeit
def find_clusters(data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    data = data.copy()
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)
    data['cluster'] = clusters
    return data

@timeit
def visualize_clusters(data: pd.DataFrame):
    data = data.copy()
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data.drop('cluster', axis=1))
    data['pca1'] = pca_result[:, 0]
    data['pca2'] = pca_result[:, 1]
    data['pca3'] = pca_result[:, 2]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cluster_label in set(data['cluster']):
        cluster_data = data[data['cluster'] == cluster_label]
        ax.scatter(cluster_data['pca1'], cluster_data['pca2'], cluster_data['pca3'], label=f'Cluster {cluster_label}')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.legend()
    plt.show()

@timeit
def get_top_behaviors(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    what it does:
      1. Calculate Behavior Frequencies: We calculate the mean frequency of each behavior across all customers (excluding the 'cluster' column) to understand how common each behavior is overall.
      2. Calculate Cluster Behavior Frequencies: We group the data by the 'cluster' column and calculate the mean frequency of each behavior within each cluster. This gives us an idea of how each behavior is represented within each cluster.
      3. Calculate Relative Frequency: We subtract the overall behavior frequency from the cluster behavior frequency. This helps us identify behaviors that are more common within a specific cluster compared to their overall frequency.
      4. Identify Top Behaviors: For each cluster, we sort the behaviors based on their relative frequency in descending order. We then select behaviors that have a relative frequency greater than 0.1 (you can adjust this threshold) and take the top n behaviors.
    """
    data = data.copy()
    behavior_freq = data.drop(['cluster'], axis=1).mean()
    cluster_behavior_freq = data.groupby('cluster').mean()
    relative_freq = cluster_behavior_freq.sub(behavior_freq, axis=1)
    top_behaviors = pd.DataFrame()
    for i, cluster_relative_freq in relative_freq.iterrows():
        sorted_behaviors = cluster_relative_freq.sort_values(ascending=False)
        meaningful_behaviors = sorted_behaviors[sorted_behaviors > 0.1].head(n)
        top_behaviors = pd.concat([top_behaviors, meaningful_behaviors.rename(f'Cluster {i}')], axis=1)
    return top_behaviors


# entrypoint function
def clustering_analysis(data: pd.DataFrame, n_clusters: int, top_n_feature_per_cluster=5) -> pd.DataFrame:
    """
    - data: one observation per row. One feature per column
    - n_cluster: number of clusters you want to find
    - top_n_feature_per_cluster: how many most features per cluster you want to see
    """
    clustered_data = find_clusters(data, n_clusters)
    visualize_clusters(clustered_data)
    top_behaviors = get_top_behaviors(clustered_data, top_n_feature_per_cluster)
    return clustered_data, top_behaviors
