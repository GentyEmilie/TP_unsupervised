import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# visualisation des données 
path = './artificial/'
databrut1 = arff.loadarff(open(path+"triangle1.arff", 'r'))
databrut2 = arff.loadarff(open(path+"2d-4c.arff", 'r'))
databrut3 = arff.loadarff(open(path+"2d-20c-no0.arff", 'r'))

datanp1 = np.array([[x[0], x[1]] for x in databrut1[0]])
datanp2 = np.array([[x[0], x[1]] for x in databrut2[0]])
datanp3 = np.array([[x[0], x[1]] for x in databrut3[0]])

f0_1 = datanp1[:, 0] 
f1_1 = datanp1[:, 1] 
f0_2 = datanp2[:, 0] 
f1_2 = datanp2[:, 1] 
f0_3 = datanp3[:, 0] 
f1_3 = datanp3[:, 1] 


plt.scatter(f0_1 , f1_1 , s=8)
plt.title(" Donnees pour triangles1.arff")

plt.figure()
plt.scatter(f0_2 , f1_2 , s=8)
plt.title(" Donnees pour 2d-4c.arff")

plt.figure()
plt.scatter(f0_3 , f1_3 , s=8)
plt.title(" Donnees pour 2d-20c-no0.arff")

# étude itérative triangle1.arff
tpstotal1 = time.time()
for k in range(2, 10):
    tps1 = time.time ()
    model = cluster.KMeans(n_clusters=k, init ='k-means++')
    model.fit(datanp1)
    tps2 = time.time ()
    labels = model.labels_
    iteration = model.n_iter_
    # métriques
    silhouette = silhouette_score(datanp1, labels)
    davies = davies_bouldin_score(datanp1, labels)
    calinski = calinski_harabasz_score(datanp1, labels)

    print("k =", k, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb iter =", iteration , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")
tpstotal2 = time.time()
print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")

# étude itérative 2d-4c.arff
tpstotal1 = time.time()
for k in range(2, 10):
    tps1 = time.time ()
    model = cluster.KMeans(n_clusters=k, init ='k-means++')
    model.fit(datanp2)
    tps2 = time.time ()
    labels = model.labels_
    iteration = model.n_iter_
    # métriques
    silhouette = silhouette_score(datanp2, labels)
    davies = davies_bouldin_score(datanp2, labels)
    calinski = calinski_harabasz_score(datanp2, labels)

    print("k =", k, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb iter =", iteration , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")
tpstotal2 = time.time()
print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")

# étude itérative 2d-20c-no0.arff
tpstotal1 = time.time()
for k in range(2, 25):
    tps1 = time.time ()
    model = cluster.KMeans(n_clusters=k, init ='k-means++')
    model.fit(datanp3)
    tps2 = time.time ()
    labels = model.labels_
    iteration = model.n_iter_
    # métriques
    silhouette = silhouette_score(datanp3, labels)
    davies = davies_bouldin_score(datanp3, labels)
    calinski = calinski_harabasz_score(datanp3, labels)

    print("k =", k, ": Silhouette =", round(silhouette,2), ", Davies =", round(davies,2), ", Calinski =", round(calinski,2),  ", nb iter =", iteration , ", runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")
tpstotal2 = time.time()
print("runtime total = ", round (( tpstotal2 - tpstotal1 )*1000 , 2 ) ,"ms\n")


plt.show()

