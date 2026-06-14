import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans




import warnings
warnings.filterwarnings("ignore")


dataset_path = r"C:\\Users\\ABDULLAH COMPUTERS\\Desktop\\New folder\\Plant Disease\\data\\PlantVillage"
print(os.listdir(dataset_path))


image_paths = []
image_labels = []


for cls in os.listdir(dataset_path):
    cls_path = os.path.join(dataset_path, cls)
    if os.path.isdir(cls_path):
        for img in os.listdir(cls_path):
            image_paths.append(os.path.join(cls_path, img))
            image_labels.append(cls)


print("Total images:", len(image_paths))


def extract_hog(image):
    hog = cv2.HOGDescriptor()
    return hog.compute(image).flatten()

def preprocess_image(path, size=(128, 128)):
    img = cv2.imread(path)
    
    if img is None:
        return None
    
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    img = cv2.resize(img, size)
 
    return extract_hog(img)




features = []
valid_labels = []

for path, label in zip(image_paths, image_labels):
    feat = preprocess_image(path)
    if feat is not None:
        features.append(feat)
        valid_labels.append(label)

X = np.array(features)
print("Feature shape:", X.shape)



def extract_hog(image):
    hog = cv2.HOGDescriptor(
        _winSize=(128,128),
        _blockSize=(16,16),
        _blockStride=(8,8),
        _cellSize=(8,8),
        _nbins=9
    )
    return hog.compute(image).flatten().astype(np.float32)


def feature_batch_generator(image_paths, batch_size=128):
    batch = []
    for path in image_paths:
        feat = preprocess_image(path)
        if feat is not None:
            batch.append(feat.astype(np.float32))
        if len(batch) == batch_size:
            yield np.array(batch)
            batch = []
    if batch:
        yield np.array(batch)

        

ipca = IncrementalPCA(n_components=50, batch_size=128)

for X_batch in feature_batch_generator(image_paths):
    ipca.partial_fit(X_batch)

print("Incremental PCA trained")


kmeans = MiniBatchKMeans(
    n_clusters=3,
    random_state=42,
    batch_size=128,
    n_init=10
)

for X_batch in feature_batch_generator(image_paths):
    X_pca = ipca.transform(X_batch)
    kmeans.partial_fit(X_pca)

print("MiniBatch K-Means trained")

import os

test_folder = r"C:\Users\ABDULLAH COMPUTERS\Desktop\Testing"

print("\n🔍 Cluster predictions:\n")

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    
    feat = preprocess_image(img_path)
    if feat is None:
        print(f"{img_name} →  Invalid image")
        continue
    
    feat_pca = ipca.transform(feat.reshape(1, -1))
    cluster = kmeans.predict(feat_pca)[0]
    
    print(f"{img_name} →  Cluster {cluster}")






test_folder = r"C:\Users\ABDULLAH COMPUTERS\Desktop\Testing"


images = []
labels = []

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    

    feat = preprocess_image(img_path)
    if feat is None:
        print(f"{img_name} →  Invalid image")
        continue
    
    
    feat_pca = ipca.transform(feat.reshape(1, -1))
    cluster = kmeans.predict(feat_pca)[0]
    
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    images.append(img)
    labels.append(cluster)
    
    print(f"{img_name} →  Cluster {cluster}")

# Visualization


num_images = len(images)
cols = 4  
rows = (num_images + cols - 1) // cols  

plt.figure(figsize=(16, 4 * rows))

for i, (img, cluster) in enumerate(zip(images, labels)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(f"Cluster {cluster}")
    plt.axis('off')

plt.tight_layout()
plt.show()





desktop_path = r"C:\Users\ABDULLAH COMPUTERS\Desktop"


image_extensions = (".jpg", ".jpeg", ".png")

for file_name in os.listdir(desktop_path):
    if file_name.lower().endswith(image_extensions):
        
        img_path = os.path.join(desktop_path, file_name)

      
        feat = preprocess_image(img_path)

        if feat is None:
            print(f"{file_name} →  Invalid image")
            continue

    
        feat_pca = ipca.transform(feat.reshape(1, -1))

        
        cluster = kmeans.predict(feat_pca)[0]

        print(f" {file_name} belongs to Cluster {cluster}")

       
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"{file_name}\nCluster: {cluster}")
        plt.axis("off")
        plt.show()


