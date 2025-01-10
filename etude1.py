import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import scipy.cluster.hierarchy as shc


# visualisation des données 
path = './dataset-rapport/'
databrut1 = np.loadtxt(open(path+"x1.txt", 'r'))
databrut2 = np.loadtxt(open(path+"x2.txt", 'r'))
databrut6 = np.loadtxt(open(path+"zz1.txt", 'r'))
databrut7 = np.loadtxt(open(path+"zz2.txt", 'r'))

f0_1 = databrut1[:, 0] 
f1_1 = databrut1[:, 1] 
f0_2 = databrut2[:, 0] 
f1_2 = databrut2[:, 1] 
f0_6 = databrut6[:, 0] 
f1_6 = databrut6[:, 1] 
f0_7 = databrut7[:, 0] 
f1_7 = databrut7[:, 1] 


# étude itérative k-means
donnes = [databrut1, databrut2, databrut6, databrut7]

for d in donnes:
    tpstotal1 = time.time()
    for k in range(2, 20):
        tps1 = time.time ()
        model = cluster.KMeans(n_clusters=k, init ='k-means++')
        model.fit(d)
        tps2 = time.time ()
        labels = model.labels_
        iteration = model.n_iter_
        # métriques
        silhouette = silhouette_score(d, labels)
        davies = davies_bouldin_score(d, labels)
        calinski = calinski_harabasz_score(d, labels)
        print("k =", k, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb iter =", iteration , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")

    tpstotal2 = time.time()
    print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")

# étude itérative clustering agglomeratif sur k
donnes = [databrut1, databrut2, databrut6, databrut7]

for d in donnes:
    tpstotal1 = time.time()
    for k in range(2, 20):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(linkage='ward', n_clusters=k)
        model = model.fit(d)
        tps2 = time.time()
        labels = model.labels_
        k = model.n_clusters_
        leaves = model.n_leaves_

        # métriques
        try:
            silhouette = silhouette_score(d, labels)
            davies = davies_bouldin_score(d, labels)
            calinski = calinski_harabasz_score(d, labels)
            print("k =",k, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb feuilles =", leaves , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")
        except Exception as e:
            print("k = 1 : runtime=", round(( tps2 - tps1 )*1000 , 2 ), "ms")

    tpstotal2 = time.time()
    print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")



plt.show()