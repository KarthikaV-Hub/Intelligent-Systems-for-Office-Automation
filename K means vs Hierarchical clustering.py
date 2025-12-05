from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

sc = StandardScaler()
Xn = sc.fit_transform(X)

km = KMeans(n_clusters=3, random_state=42)
km_labels = km.fit_predict(Xn)

hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
hc_labels = hc.fit_predict(Xn)

km_ari = adjusted_rand_score(y, km_labels)
hc_ari = adjusted_rand_score(y, hc_labels)

print("K-Means ARI:", km_ari)
print("Hierarchical ARI:", hc_ari)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(Xn[:,0], Xn[:,1], c=km_labels)
plt.title("K-Means")

plt.subplot(1,2,2)
plt.scatter(Xn[:,0], Xn[:,1], c=hc_labels)
plt.title("Hierarchical")

plt.tight_layout()
plt.show()
