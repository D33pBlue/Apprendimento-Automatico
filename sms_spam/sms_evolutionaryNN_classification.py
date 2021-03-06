# -*- coding: utf-8 -*-
from __future__ import division
import csv,random,pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV


def saveVar(var,name):
    """Save a variable to file"""
    with open(name+'.pickle','wb') as fl:
        pickle.dump(var,fl)

def loadVar(name):
    """Load a variable from file"""
    with open(name+'.pickle','rb') as fl:
        return pickle.load(fl)


def preprocess(X):
    X2 = []
    for x in X:
        words = []
        for w in x.lower().split():
            words.append(''.join([c for c in w if c.isalpha()]))
        X2.append(" ".join(words))
    return np.array(X2)

def bag_word(X):
    words = set()
    for x in X:
        for w in x.split():
             words.add(w)
    bag = []
    for w in words:
        bag.append(w)
    bag.sort()
    return bag

def get_features(X,bag):
    X_f = []
    for x in X:
        ftr = []
        for word in bag:
            ftr.append(x.find(word))
        X_f.append(ftr)
    return np.array(X_f)


X,y = [],[]
# with open('sms_spam.csv', 'rb') as csvfile:
#     recordfile = csv.reader(csvfile, delimiter=',', quotechar='"')
#     for row in recordfile:
#         X.append(row[1].decode('latin-1'))
#         if row[0]=='spam':
#             y.append(1)
#         else:
#             y.append(0)
#
# saveVar(X,"X")
# saveVar(y,"y")
# raise

X = loadVar("X")
y = loadVar("y")

X = np.array(X[1:])
y = np.array(y[1:])
X = preprocess(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

print(str(X.shape)+str(y.shape))
bag = bag_word(X_train)
print("bag length:"+str(len(bag)))
X_train = get_features(X_train,bag)
X_test = get_features(X_test,bag)
print(str(X_train.shape))




mlp = MLPClassifier()

p_grid = {
    #"alpha" : [10**i for i in range(-8,1)],
    "alpha": np.logspace(10**(-8), 1, num=25, base=10),
    'hidden_layer_sizes': [(50,)],
    "verbose":[True]
    }

random.seed(1)

cv = EvolutionaryAlgorithmSearchCV(estimator=mlp,
                                   params=p_grid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=1,
                                   population_size=5,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=4)

#clf = GridSearchCV(mlp, param_grid=p_grid, cv=5, scoring='accuracy')
cv.fit(X_train, y_train)
print("VALIDATION score:"+str(cv.best_score_))
print("BEST parameters:"+str(cv.best_params_))
y_pred = cv.predict(X_test)

print("accuracy:"+str(accuracy_score(y_test, y_pred)))

"""
result:
VALIDATION score: 0.980712563622
BEST parameters: {'alpha': 1e-05, 'verbose': True, 'hidden_layer_sizes': (50,)}
accuracy: 0.975530179445

"""
