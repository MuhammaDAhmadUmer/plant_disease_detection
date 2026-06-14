Student 1

Name: Muhammad Ahmad Umer

Roll Number: F23BDOCS1E02054

Student 2

Name: Zeeshan

Roll Number: 02083

Section: 2E

Semester: 6

AI-Based Plant Disease Detection and Clustering System
Project Overview

This project presents an AI-Based Plant Disease Detection and Clustering System that uses Computer Vision and Machine Learning techniques to analyze plant leaf images and group them according to their visual characteristics. The system extracts image features using the Histogram of Oriented Gradients (HOG) method, reduces feature dimensionality through Incremental Principal Component Analysis (Incremental PCA), and performs unsupervised classification using MiniBatch K-Means Clustering.

The primary objective of this project is to assist in the early identification and organization of plant diseases by automatically clustering similar leaf images, which can support agricultural research and intelligent farming applications.

Objectives
Extract meaningful features from plant leaf images.
Reduce high-dimensional image data efficiently.
Group similar plant disease images using clustering algorithms.
Visualize clustering results for better understanding and analysis.
Demonstrate the application of Artificial Intelligence in agriculture.
Technologies Used
Python
OpenCV
NumPy
Pandas
Pillow (PIL)
Scikit-learn
Matplotlib
Machine Learning Workflow
1. Dataset Loading

The PlantVillage dataset is loaded from the local storage, and image paths along with their corresponding labels are collected.

2. Image Preprocessing
Read image using OpenCV.
Convert image to grayscale.
Resize image to 128 × 128 pixels.
Prepare image for feature extraction.
3. Feature Extraction

The Histogram of Oriented Gradients (HOG) technique is used to extract important texture and shape information from leaf images.

4. Dimensionality Reduction

Incremental PCA is applied to reduce the large feature space into a lower-dimensional representation while maintaining important information.

5. Clustering

MiniBatch K-Means clustering is used to group similar plant disease images into clusters.

6. Prediction

New test images can be processed and assigned to the most suitable cluster.

7. Visualization

The clustered images are displayed using Matplotlib for easy interpretation.

Project Structure
Plant-Disease-Clustering/
│
├── plant_disease_clustering.py
├── README.md
├── requirements.txt
├── data/
│   └── PlantVillage/
└── Testing/
Features
Automated image preprocessing
HOG feature extraction
Incremental PCA for efficient dimensionality reduction
MiniBatch K-Means clustering
Prediction of unseen plant images
Graphical visualization of clustering results
Scalable approach for large datasets
