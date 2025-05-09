# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from kneed import KneeLocator

# Set up output directory
if not os.path.exists('output'):
    os.makedirs('E:/Python codeptit/Bai_tap_lon/Bai_3/output')

def clustering_analysis(df):
    """
    Perform K-means clustering and PCA visualization for player statistics.
    Determine optimal number of clusters and provide comments.
    """
    if df.empty:
        print("Empty DataFrame; skipping clustering")
        return df, 0

    # Step 1: Prepare data
    # Exclude non-numeric columns
    numeric_cols = [col for col in df.columns if col not in ['Name', 'Nation', 'Squad', 'Pos']]
    X = df[numeric_cols].copy()
    
    # Handle missing values and non-numeric entries
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Determine optimal number of clusters using elbow method
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig('E:/Python codeptit/Bai_tap_lon/Bai_3/output/elbow_plot.png')
    plt.close()
    
    # Choose optimal k (e.g., where elbow bends, typically 3-5 for football roles)
    elbow_pos = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
    optimal_k = elbow_pos.knee
    
    # Step 3: Apply K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Step 4: PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA Explained Variance Ratio: {explained_variance.sum():.2f} ({explained_variance[0]:.2f}, {explained_variance[1]:.2f})")
    
    # Plot 2D clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title('2D PCA Cluster Visualization of Premier League Players')
    plt.xlabel(f'PCA Component 1 ({explained_variance[0]*100:.1f}% variance)')
    plt.ylabel(f'PCA Component 2 ({explained_variance[1]*100:.1f}% variance)')
    plt.grid(True)
    plt.savefig('E:/Python codeptit/Bai_tap_lon/Bai_3/output/pca_clusters.png')
    plt.close()
    
    # Step 5: Analyze clusters
    cluster_summary = df.groupby('Cluster').agg({
        'Pos': lambda x: x.mode()[0] if not x.mode().empty else 'N/a',
        'Gls': 'mean',
        'Ast': 'mean',
        'xG': 'mean',
        'xAG': 'mean',
        'Min': 'mean',
        'Squad': lambda x: x.value_counts().index[0] if not x.value_counts().empty else 'N/a'
    }).reset_index()
    
    # Save cluster summary
    cluster_summary.to_csv('E:/Python codeptit/Bai_tap_lon/Bai_3/output/cluster_summary.csv', index=False)
    
    return df, optimal_k, cluster_summary

def main():
    # Load data from Part I
    try:
        df = pd.read_csv('E:/Python codeptit/Bai_tap_lon/Bai_1/results.csv')
    except FileNotFoundError:
        print("results.csv not found; ensure Part I is completed")
        return
    
    # Run clustering analysis
    df_clustered, k, cluster_summary = clustering_analysis(df)
    
    if not df_clustered.empty:
        print(f"Clustering completed with {k} clusters")
        print("\nCluster Summary:")
        print(cluster_summary)
        df_clustered.to_csv('E:/Python codeptit/Bai_tap_lon/Bai_3/output/results_with_clusters.csv', index=False)
        print("Results saved to output/results_with_clusters.csv")
    else:
        print("Clustering failed due to empty data")

if __name__ == "__main__":
    main()