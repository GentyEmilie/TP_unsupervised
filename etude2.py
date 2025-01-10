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
databrut3 = np.loadtxt(open(path+"x3.txt", 'r'))
databrut4 = np.loadtxt(open(path+"x3.txt", 'r'))
databrut5 = np.loadtxt(open(path+"y1.txt", 'r'))

f0_3 = databrut3[:, 0] 
f1_3 = databrut3[:, 1] 
f0_4 = databrut4[:, 0] 
f1_4 = databrut4[:, 1] 
f0_5 = databrut5[:, 0] 
f1_5 = databrut5[:, 1] 


# étude itérative k-means
donnes = [databrut3, databrut4] # ajouter databrut5 pour tester le temps de calcul long

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
donnes = [databrut3, databrut4]

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