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
databrut3 = np.loadtxt(open(path+"x3.txt", 'r'))
databrut4 = np.loadtxt(open(path+"x4.txt", 'r'))
databrut5 = np.loadtxt(open(path+"y1.txt", 'r'))
databrut6 = np.loadtxt(open(path+"zz1.txt", 'r'))
databrut7 = np.loadtxt(open(path+"zz2.txt", 'r'))

f0_1 = databrut1[:, 0] 
f1_1 = databrut1[:, 1] 
f0_2 = databrut2[:, 0] 
f1_2 = databrut2[:, 1] 
f0_3 = databrut3[:, 0] 
f1_3 = databrut3[:, 1] 
f0_4 = databrut4[:, 0] 
f1_4 = databrut4[:, 1] 
f0_5 = databrut5[:, 0] 
f1_5 = databrut5[:, 1] 
f0_6 = databrut6[:, 0] 
f1_6 = databrut6[:, 1] 
f0_7 = databrut7[:, 0] 
f1_7 = databrut7[:, 1] 

plt.scatter(f0_1 , f1_1 , s=8)
plt.title(" Donnees pour x1.arff")

plt.figure()
plt.scatter(f0_2 , f1_2 , s=8)
plt.title(" Donnees pour x2.arff")

plt.figure()
plt.scatter(f0_3 , f1_3 , s=8)
plt.title(" Donnees pour x3.arff")

plt.figure()
plt.scatter(f0_4 , f1_4 , s=8)
plt.title(" Donnees pour x4.arff")

plt.figure()
plt.scatter(f0_5 , f1_5 , s=8)
plt.title(" Donnees pour y1.arff")

plt.figure()
plt.scatter(f0_6 , f1_6 , s=8)
plt.title(" Donnees pour zz1.arff")

plt.figure()
plt.scatter(f0_7 , f1_7 , s=8)
plt.title(" Donnees pour zz2.arff")

plt.show()