# Image Clustering and Retrieval Project

This project involves clustering and retrieval of images using machine learning techniques. The dataset contains **22,914 images**, and the goal is to efficiently group similar images and implement a robust image retrieval system.

## Table of Contents

- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

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
   - Techniques explored: K-Means, DBSCAN, and hierarchical clustering.

2. **Image Retrieval**:
   - Implementing a similarity-based retrieval system.
   - Utilizing feature extraction methods (e.g., CNNs like ResNet, VGG).
   - Nearest neighbor search using FAISS for efficient retrieval.

### Workflow:
1. **Preprocessing:** Standardizing, resizing, and normalizing the dataset.
2. **Feature Extraction:** Using deep learning models or traditional descriptors (e.g., SIFT).
3. **Clustering:** Applying clustering algorithms and evaluating results.
4. **Retrieval:** Building a retrieval engine for querying similar images.

---

## Technologies Used

- **Programming Language:** Python
- **Frameworks and Libraries:**
  - TensorFlow / PyTorch
  - Scikit-learn
  - OpenCV
  - FAISS
  - Matplotlib (for visualization)

---

## Setup and Installation

### Prerequisites:
- Python 3.8+
- Google Colab (preferred) or local environment with GPU support
- Required libraries:
  ```bash
  pip install numpy pandas matplotlib scikit-learn opencv-python tensorflow faiss-gpu
  ```

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Balajibal/TangoEyeAssessment

   ```
2. Download the dataset using the link provided in the [https://drive.google.com/drive/folders/1lICo1MXPo5AmkQv__FUVZDd_rTfujk3G?usp=drive_link](#dataset) section.

### Additional Notes
- Ensure your environment has sufficient GPU memory for processing large datasets.
- For traditional methods, the required libraries for feature extraction like OpenCV must be installed and tested.

---

## Usage

1. **Data Preprocessing:**
   - Use the scripts in `src/preprocessing/` to clean and prepare the dataset.
   - Example command:
     ```bash
     python src/preprocessing/data_cleaning.py --input-path data/raw --output-path data/processed
     ```

2. **Feature Extraction:**
   - Run `src/feature_extraction.py` to extract features from images.
   - Example command:
     ```bash
     python src/feature_extraction.py --input-path data/processed --output-path features
     ```

3. **Clustering:**
   - Use `src/clustering.py` to apply clustering algorithms and visualize results.
   - Example command:
     ```bash
     python src/clustering.py --features-path features --clusters 10 --output-path results
     ```

4. **Image Retrieval:**
   - Execute `src/retrieval.py` to test the retrieval system.
   - Example command:
     ```bash
     python src/retrieval.py --query-path data/query.jpg --features-path features --output-path results
     ```

5. **Visualization:**
   - Scripts are provided to visualize clustering results and retrieval outcomes.

---

## Detailed Workflow

### 1. Data Preprocessing
Data preprocessing involves cleaning, resizing, and normalizing images. Scripts are provided to ensure consistent formatting.

- Resizing all images to 224x224 pixels.
- Normalizing pixel values to range [0, 1].
- Deduplication and removal of corrupted images.

### 2. Feature Extraction
Feature extraction uses both traditional methods (e.g., SIFT) and deep learning-based CNNs like ResNet and VGG.

- For CNN-based features, pre-trained models from TensorFlow or PyTorch are used.
- For traditional methods, descriptors are extracted using OpenCV.

### 3. Clustering
Clustering is implemented using various techniques:

- **K-Means Clustering:**
  - Requires specifying the number of clusters (`k`).
  - Results can be visualized using t-SNE or PCA.

- **DBSCAN:**
  - Density-based clustering that does not require pre-specifying `k`.
  - Ideal for datasets with noise.

- **Hierarchical Clustering:**
  - Creates a dendrogram to visualize clustering hierarchy.

### 4. Image Retrieval
The retrieval system is built using the FAISS library for fast similarity searches.

- Features are stored in a vector index.
- Queries are processed to find nearest neighbors.
- Results are displayed as a ranked list of similar images.

### Performance Evaluation
Evaluation metrics include:

- **Clustering Quality:**
  - Silhouette score, Davies-Bouldin index.

- **Retrieval Accuracy:**
  - Precision, Recall, and Mean Average Precision (mAP).

---

## Advanced Topics

### Dimensionality Reduction
High-dimensional features are reduced using:

- **Principal Component Analysis (PCA):** To retain 95% variance.
- **t-SNE:** For 2D visualization of clusters.

### Optimization
- Hyperparameter tuning is performed for clustering algorithms (e.g., `k` in K-Means).
- Efficient index building using FAISS optimizations for large datasets.

### Scalability
- The system is designed to handle large-scale datasets by leveraging GPU acceleration.
- Batch processing is implemented for feature extraction and clustering.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork:
   ```bash
   git add .
   git commit -m "Add new feature"
   git push origin feature-name
   ```
4. Submit a pull request.

### Guidelines:
- Write clear, concise commit messages.
- Follow the code style and conventions used in the project.
- Ensure that all tests pass before submitting.

---



## Roadmap

Future enhancements include:

1. Adding support for additional clustering algorithms (e.g., OPTICS).
2. Implementing a web-based interface for the retrieval system.
3. Enhancing evaluation metrics with user feedback.
4. Optimizing the system for deployment on cloud platforms.

---



---
