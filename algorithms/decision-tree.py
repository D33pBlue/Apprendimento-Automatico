"""
Decision Tree - Bizzaro Francesco
fonti:
http://scikit-learn.org/stable/modules/tree.html#classification
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
http://www.graphviz.org/Download_windows.php
https://pypi.python.org/pypi/decision-tree-id3/0.1.2
"""
#from sklearn import tree
from id3 import Id3Estimator
from id3 import export_graphviz
import numpy as np
import graphviz

#           | 0     | 1         | 2
#Outlook    | Sunny | Overcast  | Rain
#Temperature| Hot   | Mild      | Cool
#Humidity   | High  | Normal    | -
#Wind       | Weak  | Strong    | -

x_labels=["Outlook","Temperature","Humidity","Wind"]

X = np.array([
    [0,0,0,0],
    [0,0,0,1],
    [1,0,0,0],
    [2,1,0,0],
    [2,2,1,0],
    [2,2,1,1],
    [1,2,1,1],
    [0,1,0,0],
    [0,2,1,0],
    [2,1,1,0],
    [0,1,1,1],
    [1,1,0,1],
    [1,0,1,0],
    [2,1,0,1]
])

Y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])

#clf = tree.DecisionTreeClassifier()
clf = Id3Estimator(min_samples_split=3)
clf.fit(X,Y)
dot_data = export_graphviz(clf.tree_,"decisiontree.dot",x_labels)
#predictions = clf.predict(X)
#for i in range(len(X)):
#    print X[i],Y[i],"->",predictions[i]
