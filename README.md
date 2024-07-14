# PCA-ASSIGNMENT
import pandas as pd
df=pd.read_csv(r"C:\Users\haree\OneDrive\Desktop\wine.csv")
df.head()
df.info()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns):
    if df[col].dtype in ['int64', 'float64']:
        plt.subplot(3, 3, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(col)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns):
    if df[col].dtype in ['int64', 'float64']:
        plt.subplot(3, 3, i + 1)
        sns.boxplot(y=df[col])
        plt.title(col)
plt.tight_layout()
plt.show()
correlation_matrix = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()import numpy as np
import matplotlib.pyplot as plt

# Plotting the explained variance ratio (scree plot)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plotting cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True)
plt.show()
sns.pairplot(df, hue='Cluster', palette='viridis', plot_kws={'alpha': 0.6}, diag_kind='hist')
plt.suptitle('Pairplot of Clusters')
plt.show()
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(df, cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')
from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(df, cluster_labels)
print(f'Davies–Bouldin Index: {db_index}')
plt.figure(figsize=(10, 6))

# Example with two principal components (PC1 and PC2)
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', edgecolors='k', s=50)
plt.title('K-means Clustering (PCA-transformed data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
sns.pairplot(df_pca, hue='Cluster', palette='viridis', plot_kws={'alpha': 0.6}, diag_kind='hist')
plt.suptitle('Pairplot of Clusters (PCA-transformed data)')
plt.show()
