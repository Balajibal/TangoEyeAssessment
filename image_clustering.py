import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load Pretrained Model
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load and Preprocess Images
def load_images(folder_path, batch_size=500):
    filenames = []
    features = []
    model = load_model()
    
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for file in batch_files:
            img = load_img(file, target_size=(224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)
            batch_images.append(img)
            filenames.append(file)

        batch_images = np.array(batch_images)
        batch_features = model.predict(batch_images, verbose=1)
        features.append(batch_features)

    features = np.vstack(features)
    return features, filenames

# Perform Clustering
def perform_clustering(features, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels, kmeans

# PCA Visualization
def visualize_pca(features, cluster_labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('PCA of Image Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

# Attribute Extraction (Dominant Color)
def extract_dominant_color(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(img)
    return kmeans.cluster_centers_[0]

# Query-Based Retrieval
def find_similar_images(query_image_path, features, filenames, kmeans_model):
    model = load_model()
    query_img = load_img(query_image_path, target_size=(224, 224))
    query_img = img_to_array(query_img)
    query_img = preprocess_input(query_img)
    query_img = np.expand_dims(query_img, axis=0)

    query_feature = model.predict(query_img)
    cluster_label = kmeans_model.predict(query_feature)[0]

    similar_images = [filenames[i] for i in range(len(filenames)) if kmeans_model.labels_[i] == cluster_label]
    return similar_images

# Main Function
def main(image_folder, query_image_path=None):
    print("Loading images and extracting features...")
    features, filenames = load_images(image_folder)

    print("Performing clustering...")
    cluster_labels, kmeans = perform_clustering(features, n_clusters=100)

    print("Visualizing clusters with PCA...")
    visualize_pca(features, cluster_labels)

    if query_image_path:
        print("Finding similar images...")
        similar_images = find_similar_images(query_image_path, features, filenames, kmeans)
        print(f"Query Image: {query_image_path}")
        print("Similar Images:")
        for img_path in similar_images:
            print(img_path)

# Run Script
if __name__ == "__main__":
    image_folder = r"C:\Users\Balaji N\Downloads\tango-cv-assessment-dataset\tango-cv-assessment-dataset\input"
    query_image_path = r"C:\Users\Balaji N\Downloads\tango-cv-assessment-dataset\tango-cv-assessment-dataset\output\1267_c1s5_069416_00.jpg"  # Provide a query image path

    main(image_folder, query_image_path)
