import numpy as np
from sklearn.cluster import KMeans
CLASS_INDEX = 6
a = np.loadtxt('classe0.csv', delimiter=',') 
class0 = np.array([row for row in a])


kmeans = KMeans(n_clusters=260).fit(class0)

final =[];
for x in range(len(kmeans.cluster_centers_)):
	final.append(kmeans.cluster_centers_[x])

print(len(final))

np.savetxt("kmeansDaClasse0.csv", final, delimiter=",",fmt='%.8f')
