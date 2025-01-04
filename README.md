# Image Clustering and Retrieval Project

This project involves clustering and retrieval of images using machine learning techniques. The dataset contains **22,914 images**, and the goal is to efficiently group similar images and implement a robust image retrieval system.

## Table of Contents

- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Detailed Workflow](#detailed-workflow)
- [Advanced Topics](#advanced-topics)
- [Roadmap](#roadmap)

---

## Dataset

The dataset used for this project is hosted on Google Drive and can be accessed using the link below:

**[Download Dataset](https://drive.google.com/drive/folders/1lICo1MXPo5AmkQv__FUVZDd_rTfujk3G?usp=drive_link)**

### Details:
- **Number of images:** 22,914
- **Format:** JPG/PNG
- **Usage:** Image clustering and similarity-based retrieval

---

## Project Overview

This project focuses on:

1. **Image Clustering**:
   - Grouping images with similar features using unsupervised learning techniques.
   - Techniques explored: K-Means clustering.

2. **Image Retrieval**:
   - Implementing a similarity-based retrieval system.
   - Utilizing feature extraction methods (e.g., CNNs like ResNet50).

### Workflow:
1. **Preprocessing:** Standardizing, resizing, and normalizing the dataset.
2. **Feature Extraction:** Using a pre-trained ResNet50 model.
3. **Clustering:** Applying clustering algorithms and evaluating results.
4. **Retrieval:** Building a retrieval engine for querying similar images.

---

## Technologies Used

- **Programming Language:** Python
- **Frameworks and Libraries:**
  - TensorFlow
  - Scikit-learn
  - OpenCV
  - Matplotlib (for visualization)

---

## Setup and Installation

### Prerequisites:
- Python 3.8+
- GPU-enabled environment (optional but recommended for faster processing)
- Required libraries:
  ```bash
  pip install numpy matplotlib scikit-learn opencv-python tensorflow
  ```

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Balajibal/TangoEyeAssessment
   
   ```
2. Download the dataset using the link provided in the [https://drive.google.com/drive/folders/1lICo1MXPo5AmkQv__FUVZDd_rTfujk3G?usp=drive_link](#dataset) section.
3. Ensure your environment has sufficient GPU memory for processing large datasets.

---

## Usage

1. **Update Variables in the Script:**
   - `image_folder`: Path to the folder containing your dataset.
   - `query_image_path`: Path to the query image for retrieval.

2. **Run the Script:**
   ```bash
   python clustering_and_retrieval.py
   ```

3. **Output:**
   - Clusters visualized using PCA.
   - Similar images retrieved based on the query image.

---

## Detailed Workflow

### 1. Feature Extraction

The script uses the ResNet50 model pre-trained on ImageNet to extract high-dimensional feature vectors from images.

- **Steps:**
  1. Images are resized to `224x224` and preprocessed.
  2. The ResNet50 model computes feature vectors.
  3. Features are stored as a NumPy array for further processing.

### 2. Clustering

- **Method:** K-Means clustering
- **Objective:** Partition feature vectors into `k` clusters (default: 100).
- **Implementation:** Features are fed into Scikit-learn's `KMeans` implementation.

### 3. PCA Visualization

- Reduces high-dimensional feature vectors to 2D for visualization.
- PCA scatterplots illustrate the distribution of clusters.

### 4. Query-Based Retrieval

- A query image is processed through the ResNet50 model to extract its feature vector.
- The query's feature vector is used to identify images in the same cluster.

---

## Advanced Topics

### Dimensionality Reduction
High-dimensional features are reduced using:

- **Principal Component Analysis (PCA):** To retain 95% variance.
- **t-SNE:** For 2D visualization of clusters.

### Scalability

- The system is designed to handle large-scale datasets by leveraging GPU acceleration.
- Batch processing is implemented for feature extraction and clustering.

---

## Roadmap

Future enhancements include:

1. Adding support for additional clustering algorithms (e.g., DBSCAN).
2. Implementing a web-based interface for the retrieval system.
3. Enhancing evaluation metrics with user feedback.
4. Optimizing the system for deployment on cloud platforms.
