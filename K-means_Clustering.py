import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering

image = cv2.imread('img.avif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
shape = image.shape
pixels = image.reshape(-1, 3)

clusters = 4
kmeans = KMeans(n_clusters=clusters, random_state=42)
k_labels = kmeans.fit_predict(pixels)
k_image = kmeans.cluster_centers_[k_labels].reshape(shape).astype(np.uint8)

hier = AgglomerativeClustering(n_clusters=clusters)
h_labels = hier.fit_predict(pixels)
h_image = np.zeros_like(pixels)
for i in range(clusters):
    h_image[h_labels == i] = np.mean(pixels[h_labels == i], axis=0)
h_image = h_image.reshape(shape).astype(np.uint8)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(k_image)
plt.title('K-Means')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(h_image)
plt.title('Hierarchical')
plt.axis('off')

plt.show()
