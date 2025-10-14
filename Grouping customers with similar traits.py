#Grouping customers with similar traits or purchasing behaviors to create target marketing strategies.

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
     
dataPoints, _ = make_blobs(n_samples=300, n_features=4, centers=5, random_state=42)
customerData = pd.DataFrame(dataPoints, columns=['AnnualIncome','SpendingScore','Age','Purchases'])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(customers)


scalerModel = MinMaxScaler()
scaledValues = scalerModel.fit_transform(customerData)


kmeansModel = KMeans(n_clusters=5, n_init=12, random_state=7)
clusterLabels = kmeansModel.fit_predict(scaledValues)
customerData['Cluster'] = clusterLabels


plt.scatter(customerData['AnnualIncome'], customerData['SpendingScore'],
            c=customerData['Cluster'], cmap='rainbow', alpha=0.7)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()



