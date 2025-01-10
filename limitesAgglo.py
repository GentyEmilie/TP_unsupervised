import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import scipy.cluster.hierarchy as shc


# visualisation des données 
path = './artificial/'
databrut1 = arff.loadarff(open(path+"atom.arff", 'r'))

datanp1 = np.array([[x[0], x[1]] for x in databrut1[0]])

f0_1 = datanp1[:, 0] 
f1_1 = datanp1[:, 1] 


plt.scatter(f0_1 , f1_1 , s=8)
plt.title(" Donnees pour atom.arff")

# étude itérative banana.arff sur les distances
distances = [10, 20, 30, 40, 50, 60] 
linkages = ['single', 'average', 'complete', 'ward']  

tpstotal1 = time.time()
for dist in distances:
    print(dist)
    for linkage in linkages:
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(distance_threshold=dist, linkage=linkage, n_clusters=None)
        model = model.fit(datanp1)
        tps2 = time.time()
        labels = model.labels_
        k = model.n_clusters_
        leaves = model.n_leaves_
    
        if (k == 2):
            plt.figure()
            plt.scatter(f0_1 , f1_1 , c=labels , s=8 )
            title = "atom.arff : linkage =", linkage
            plt.title(title)

        # métriques
        try:
            silhouette = silhouette_score(datanp1, labels)
            davies = davies_bouldin_score(datanp1, labels)
            calinski = calinski_harabasz_score(datanp1, labels)
            print("k =",k, "link=",linkage, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb feuilles =", leaves , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")
        except Exception as e:
            print("k = 1, link=",linkage, " : runtime=", round(( tps2 - tps1 )*1000 , 2 ), "ms")
        
tpstotal2 = time.time()
print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")


plt.show()

