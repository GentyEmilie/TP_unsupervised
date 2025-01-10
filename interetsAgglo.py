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
databrut1 = arff.loadarff(open(path+"banana.arff", 'r'))
databrut2 = arff.loadarff(open(path+"2d-4c.arff", 'r'))

datanp1 = np.array([[x[0], x[1]] for x in databrut1[0]])
datanp2 = np.array([[x[0], x[1]] for x in databrut2[0]])

f0_1 = datanp1[:, 0] 
f1_1 = datanp1[:, 1] 
f0_2 = datanp2[:, 0] 
f1_2 = datanp2[:, 1] 


plt.scatter(f0_1 , f1_1 , s=8)
plt.title(" Donnees pour banana.arff")

plt.figure()
plt.scatter(f0_2 , f1_2 , s=8)
plt.title(" Donnees pour 2d-4c.arff")

# étude itérative banana.arff sur les distances
distances = [0.04, 0.4, 0.8, 15] 
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
        
        # décommenter ligne ci dessous pour visualiser l'annexe 1
        #if (k == 2):
        #    plt.figure()
        #    plt.scatter(f0_1 , f1_1 , c=labels , s=8 )
        #    title = "banana.arff : distance =",dist, ", linkage =",linkage 
        #    plt.title(title)

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


# étude itérative 2d-4c.arff sur les distances
distances = [10, 20, 30, 100] 
linkages = ['single', 'average', 'complete', 'ward']  

tpstotal1 = time.time()
for dist in distances:
    print(dist)
    for linkage in linkages:
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(distance_threshold=dist, linkage=linkage, n_clusters=None)
        model = model.fit(datanp2)
        tps2 = time.time()
        labels = model.labels_
        k = model.n_clusters_
        leaves = model.n_leaves_
        
        # décommenter ligne ci dessous pour visualiser les clusters obtenus
        #if (k == 4):
        #    plt.figure()
        #    plt.scatter(f0_2 , f1_2 , c=labels , s=8 )
        #    title = "2d-4c.arff : distance =",dist, ", linkage =",linkage 
        #    plt.title(title)

        # métriques
        try:
            silhouette = silhouette_score(datanp2, labels)
            davies = davies_bouldin_score(datanp2, labels)
            calinski = calinski_harabasz_score(datanp2, labels)
            print("k =",k, "link=",linkage, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb feuilles =", leaves , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")
        except Exception as e:
            print("k = 1, link=",linkage, " : runtime=", round(( tps2 - tps1 )*1000 , 2 ), "ms")
        
tpstotal2 = time.time()
print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")


# étude itérative banana.arff sur la valeur de k 
tpstotal1 = time.time()
for k in range(2, 10):
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(linkage='single', n_clusters=k)
    model = model.fit(datanp1)
    tps2 = time.time()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_

    # à decommenter pour visualiser les clusters
    #if (k == 2):
    #        plt.figure()
    #        plt.scatter(f0_1 , f1_1 , c=labels , s=8 )
    #        plt.title("banana.arff : linkage = single")

    # métriques
    try:
        silhouette = silhouette_score(datanp1, labels)
        davies = davies_bouldin_score(datanp1, labels)
        calinski = calinski_harabasz_score(datanp1, labels)
        print("k =",k, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb feuilles =", leaves , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")
    except Exception as e:
        print("k = 1 : runtime=", round(( tps2 - tps1 )*1000 , 2 ), "ms")
    
tpstotal2 = time.time()
print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")


# étude itérative 2d-4c.arff avec k

tpstotal1 = time.time()
for k in range(2, 10):
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(linkage='single', n_clusters=k)
    model = model.fit(datanp2)
    tps2 = time.time()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_

    # à decommenter pour visualiser les clusters
    #if (k == 4):    
    #        plt.figure()
    #        plt.scatter(f0_2 , f1_2 , c=labels , s=8 )
    #        plt.title("2d-4c.arff : linkage = single")

    # métriques
    try:
        silhouette = silhouette_score(datanp2, labels)
        davies = davies_bouldin_score(datanp2, labels)
        calinski = calinski_harabasz_score(datanp2, labels)
        print("k =",k, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb feuilles =", leaves , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")
    except Exception as e:
        print("k = 1 : runtime=", round(( tps2 - tps1 )*1000 , 2 ), "ms")
    
tpstotal2 = time.time()
print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")


plt.show()

