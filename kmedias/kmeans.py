import numpy as np
from sklearn.cluster import KMeans
CLASS_INDEX = 6
a = np.loadtxt('mammography-consolidatedSemDuplicada.csv', delimiter=',') 
class0 = np.array([row for row in a])
for x in range(class0):
	print(x)

kmeans = KMeans(n_clusters=260).fit(class0)

for x in range(kmeans):
	print(x)

np.savetxt("kmeansDaClasse0.csv", a, delimiter=",",fmt='%.8f')
