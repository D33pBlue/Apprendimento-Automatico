from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import scipy.cluster.hierarchy as h
import numpy as np

iris = datasets.load_iris()
X = iris.data
Y = iris.target

km = KMeans(n_clusters=3,init='random',random_state=23)
km.fit(X)

print km.cluster_centers_#sono i centroidi
print km.labels_#sono i cluster
print "inerzia:",km.inertia_#inerzia = somma distanza al quadrato degli esempi dal loro centroide

#randindex implementato da scikit non e' quella pura ma adjusted randindex per non tenere conto dello sbilanciamento fra classi ecc.. usa normalizzazione (con formula complicata)
print "adj randindex:",metrics.adjusted_rand_score(km.labels_,Y)
print "adj mutual:",metrics.adjusted_mutual_info_score(km.labels_,Y)#metrica alternativa che usa mutua informazione fra le classi


#Scatter plot
pl.scatter(X[:,0],X[:,1],c=iris.target)#1^feature e seconda
# aggiungo al grafico i centroidi proiettati
pl.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='o',s=100)
pl.show()

# CLUSTERING GERARCHICO
# usiamo libreria scipy


#single link
H = h.single(X)
print H.shape
#sono tutti i link effettuati (#esempi-1) e per ciascuno abbiamo
# coppie di cluster uniti, distanza e #esempi contenuti in nuovo cluster
h.dendrogram(H)
pl.show()
#il dendogramma e' lungo perche' c'e' chain effect tipico problema del single link


#comlpete link
H = h.complete(X)
h.dendrogram(H)
pl.show()


#average link
H = h.average(X)
h.dendrogram(H)
pl.show()


#centroid link
H = h.centroid(X)
h.dendrogram(H)
pl.show()
#ci sono delle inversioni perche' la distanza qui non e' monotona

#per ottenere un cluster devo definire una distanza
H = h.average(X)
C = h.fcluster(H,1.9,criterion='distance')#la soglia 3.5 sembra buona dal grafico
#per vedere il numero di cluster:
print "n cluster:",len(np.unique(C))
print "adj randindex gerarc:",metrics.adjusted_rand_score(C,Y)
