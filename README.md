# PCA-ASSIGNMENT
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
data=pd.read_csv(r"C:\Users\haree\OneDrive\Desktop\wine.csv")
data
data.info
data.describe()
data[data.duplicated()]
data.isnull()
data.corr()
sns.heatmap(data)
sns.boxplot(data=data,x="Type", y="Alcohol",showmeans=True)
plt.title("Distribution of Alcohol by Wine Type") 
plt.show()
sns.pairplot(data)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# standardizing the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
#implementing pca
pca = PCA()
pca_data = pca.fit_transform(scaled_data)
# scree plot
explained_variance = pca.explained_variance_ratio_
plt.plot(explained_variance)
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(True)
plt.show()
pca = PCA(n_components=2)  
reduced_data = pca.fit_transform(scaled_data) # transforming the data again
print(reduced_data)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
kmeans = KMeans(n_clusters=3) 
kmeans.fit(data)
cluster_labels = kmeans.labels_
# using scatterplot to visualize scatter plot
plt.scatter(data["Type"], data["Alcohol"], c=cluster_labels)  # Replace features with your desired axes
plt.xlabel("Type")
plt.ylabel("Alcohol")
plt.title("K-means Clustering Results")
plt.show()
silhouette_coeff = silhouette_score(data, cluster_labels)
print("Silhouette Score:", silhouette_coeff)
db_score = davies_bouldin_score(data, cluster_labels)
print("Davies-Bouldin Score:", db_score)
# K-means clustering on PCA-transformed data 
kmeans_pca = KMeans(n_clusters=3)  
kmeans_pca.fit(reduced_data)
cluster_labels_pca = kmeans_pca.labels_
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels_pca, cmap='Spectral')  # Using different colormap
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-means Clustering on PCA-transformed Data")
plt.show()
print("**Comparison:**")
print("Silhouette Score (Original):", silhouette_score(data, cluster_labels))
print("Silhouette Score (PCA):", silhouette_score(reduced_data, cluster_labels_pca))
print("Davies-Bouldin Score (Original):", davies_bouldin_score(data, cluster_labels))
print("Davies-Bouldin Score (PCA):", davies_bouldin_score(reduced_data, cluster_labels_pca)) 
