import numpy as np

#CARICAMENTO DATASETS
from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
#print breast.DESCR
X, y = breast.data, breast.target

#URL
'''
import urllib
raw_data = urllib.urlopen("http://www.math.unipd.it/~mpolato/didattica/ml1718/tic-tac-toe.svmlight")
'''

#SVMLIGHT
'''
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file("tic-tac-toe.svmlight") #o raw_data se caricato da URL
X = X.toarray() #Necessario per convertire da matrice sparsa a densa (alternativa todense)
'''

#PREPROCESSING
from sklearn import preprocessing as pp
X_scaled = pp.scale(X) #scala i dati in modo che la media (default) sia 0 e la varianza 1
#print "mean:", np.mean(X_scaled), "std:", np.std(X_scaled)
X_norm = pp.normalize(X) #normalizza in modo che la norma (l2 di default) degli esempi sia = 1


#TRAINING SET e TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#VALUTAZIONE
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
def evaluate(y_test, y_pred):
	print "accuracy:", accuracy_score(y_test, y_pred)
	print "precision:", precision_score(y_test, y_pred)
	print "recall:", recall_score(y_test, y_pred)
	#print "AUC:", roc_auc_score(y_test, y_pred)
	print
	

#ALBERI DI DECISIONE
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
#DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
clf_tree = clf_tree.fit(X_train, y_train)
y_pred = clf_tree.predict(X_test)
print "DT"
evaluate(y_test, y_pred)

#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=1, tol=1e-8, learning_rate_init=.01)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print "MLP"
evaluate(y_test, y_pred)


#SVM
from sklearn import svm
svc = svm.SVC(gamma=0.0001, C=100.0) #kernel : string, optional (default=rbf) linear, poly, rbf, sigmoid
svc.fit(X_train, y_train)
#SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,  gamma=0.001, kernel='rbf', max_iter=-1, probability=False,  random_state=None, shrinking=True, tol=0.001, verbose=False)
y_pred = svc.predict(X_test)
print "SVM"
evaluate(y_test, y_pred)


#MODEL SELECTION e VALUTAZIONE
from sklearn.model_selection import GridSearchCV, KFold

#Dizionario di dizionari che contiene la griglia dei parametri per SVM (kernel lineare, RBF e custom) e NN
p_grid = { "rbf": {"C" : [2**i for i in range(-5,5)], "gamma" : [10**i for i in range(-4, 4)]},
			"linear" : { "C": [2**i for i in range(-5,5)], "kernel": ["linear"] },
			"precomputed" : {"C": [2**i for i in range(-5,5)], "kernel": ["precomputed"]},
			#TODO polynomial kernel
			"nn" : {"alpha" : [10**i for i in range(-5,1)], 'hidden_layer_sizes': [(10,), (50,), (100,), (200,)]} }

kernel = "rbf" #precomputed/linear/poly
skf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accs = []
for train, test in skf.split(X, y):
	print "FOLD:", fold
	X_train, X_test = X[train],  X[test]
	
	##Il contenuto di X deve essere il custom kernel!
	#if kernel == "precomputed": 
	#	X_train, X_test = X_train[:,train], X_test[:,train]
	
	#GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise', return_train_score='warn')
	#clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid[kernel], cv=5, scoring='accuracy') #SVM
	clf = GridSearchCV(mlp, param_grid=p_grid["nn"], cv=5, scoring='accuracy') #NN
	clf.fit(X_train, y[train])
	
	#print "CV info:", clf.cv_results_.keys()
	print "VALIDATION score:", clf.best_score_
	print "BEST parameters:", clf.best_params_
	
	#y_pred = clf.decision_function(X_test) #Notare che e' lo score e non la predizione! Utile per calcolare l'AUC (ranking metric)
	y_pred = clf.predict(X_test)
	y_true = y[test]
	
	#auc = roc_auc_score(y_true, y_pred)
	acc = accuracy_score(y_true, y_pred)
	print "TEST score:", acc
	print
	
	accs.append(acc)
	fold += 1

print "AVG ACCURACY:", np.mean(accs), "+-", np.std(accs)


