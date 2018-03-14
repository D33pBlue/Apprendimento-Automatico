import csv,math,pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def saveVar(var,name):
    """Save a variable to file"""
    with open(name+'.pickle','wb') as fl:
        pickle.dump(var,fl)

def loadVar(name):
    """Load a variable from file"""
    with open(name+'.pickle','rb') as fl:
        return pickle.load(fl)

def evaluate(y_test, y_pred):
	print "accuracy:", accuracy_score(y_test, y_pred)
	#print "precision:", precision_score(y_test, y_pred)
	#print "recall:", recall_score(y_test, y_pred)
	print

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
        n = len(x)
        ftr = []
        for word in bag:
            # f = x.find(word)
            # if f<0:
            #     f = 10**10
            f = x.split().count(word)/float(n)
            ftr.append(f)
        X_f.append(ftr)
    return np.array(X_f)

def test(Yp,Y,verbose=True):
	N = len(Y)
	errors = 0
	logloss = 0
	for i,y in enumerate(Y):
		probs = Yp[i]
		choosen = max(zip(probs,['EAP','HPL','MWS','XXX']))[1]
		if choosen !=y:
			errors += 1
			if verbose:
				print probs,choosen,y
		for j,k in enumerate(['EAP','HPL','MWS','XXX']):
			if k==y:
				p = max(min(probs[j],1-10**(-15)),10**(-15))
				logloss += math.log(p)
	print "logloss",((-1.0)/N)*logloss
	print "errors",errors/float(N)*100

X,y = [],[]
with open('data/train.csv', 'rb') as csvfile:
    recordfile = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in recordfile:
        X.append(row[1])
        y.append(row[2])
X = np.array(X[1:1000])
y = np.array(y[1:1000])
X = preprocess(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

print X.shape,y.shape
print X[14],y[14]
#bag = bag_word(X_train)
bag = loadVar('resources/r_function_words')
bag = bag + [k[0] for k in loadVar('resources/r_sigft_words')]
print "bag length:",len(bag)
X_train = get_features(X_train,bag)
print X_train.shape
X_test = get_features(X_test,bag)
print X_test.shape



mlp = svm.SVC()


Cs = range(-5,50)
p_grid = {
	"C" : [2**i for i in Cs],
	#"gamma" : [10**i for i in range(-30, -5)],
	"verbose":[True],"kernel": ["linear"],"probability":[True]}

clf = GridSearchCV(mlp, param_grid=p_grid, cv=2, scoring='accuracy')
clf.fit(X_train, y_train)
print "VALIDATION score:", clf.best_score_
print "BEST parameters:", clf.best_params_

print clf.grid_scores_[0].mean_validation_score
scores = [x.mean_validation_score for x in clf.grid_scores_]
print Cs,scores
plt.plot([2**i for i in Cs], scores)
plt.plot([2**i for i in Cs], scores, 'ro')
plt.legend()
plt.xlabel('C (logarithmic scale)')
plt.ylabel('Mean score')
plt.show()
# y_pred = clf.predict(X_test)
# Yp = clf.predict_proba(X_test)
#
#
# evaluate(y_test, y_pred)
# test(Yp,y_test)

"""
su tutto il training set con tutte le features:
VALIDATION score: 0.69733933064
BEST parameters: {'kernel': 'linear', 'C': 3896, 'verbose': False, 'probability': True}
accuracy: 0.720984215413
logloss 0.706750318093
errors 28.1337047354

"""


"""
VALIDATION score: 0.676668159427
BEST parameters: {'kernel': 'linear', 'C': 4096, 'verbose': True, 'probability': True}
accuracy: 0.692424242424



con 1000 esempi:
    VALIDATION score: 0.472346786248
    BEST parameters: {'kernel': 'linear', 'C': 0.03125, 'verbose': True, 'gamma': 0.0001}
    accuracy: 0.478787878788
con 500 rbf:
	VALIDATION score: 0.362275449102
	BEST parameters: {'kernel': 'rbf', 'C': 2, 'verbose': True, 'gamma': 0.0001}
	accuracy: 0.442424242424

VALIDATION score: 0.464071856287
BEST parameters: {'kernel': 'rbf', 'C': 16, 'verbose': True, 'gamma': 1e-06}
accuracy: 0.539393939394

VALIDATION score: 0.497005988024
BEST parameters: {'kernel': 'rbf', 'C': 512, 'verbose': True, 'gamma': 1e-07}
accuracy: 0.557575757576


VALIDATION score: 0.5
BEST parameters: {'kernel': 'rbf', 'C': 524288, 'verbose': True, 'gamma': 1e-10}
accuracy: 0.569696969697

VALIDATION score: 0.5
BEST parameters: {'kernel': 'rbf', 'C': 524288, 'probability': True, 'gamma': 1e-10, 'verbose': True}
accuracy: 0.569696969697

#------------------------------

VALIDATION score: 0.542600896861
BEST parameters: {'kernel': 'rbf', 'C': 262144, 'probability': True, 'gamma': 1e-27, 'verbose': True}
accuracy: 0.606060606061

VALIDATION score: 0.502242152466
BEST parameters: {'kernel': 'linear', 'C': 9.5367431640625e-07, 'verbose': True, 'probability': True}
accuracy: 0.530303030303

VALIDATION score: 0.502242152466
BEST parameters: {'kernel': 'linear', 'C': 8.881784197001252e-16, 'verbose': True, 'probability': True}
accuracy: 0.530303030303

VALIDATION score: 0.538116591928
BEST parameters: {'kernel': 'rbf', 'C': 256, 'probability': True, 'gamma': 1e-24, 'verbose': True}
accuracy: 0.60303030303

#-----------

VALIDATION score: 0.533632286996
BEST parameters: {'kernel': 'linear', 'C': 8.470329472543003e-22, 'verbose': True, 'probability': True}
accuracy: 0.606060606061

VALIDATION score: 0.688343371198
BEST parameters: {'kernel': 'linear', 'C': 8.470329472543003e-22, 'verbose': True, 'probability': True}
accuracy: 0.707830393067

"""
