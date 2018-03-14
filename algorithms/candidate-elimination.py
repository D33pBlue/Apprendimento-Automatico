import numpy as np
import copy

DataSet = np.array([
    ["Sunny","Warm","Normal","Strong","Warm","Same","Yes"],
    ["Sunny","Warm","High","Strong","Warm","Same","Yes"],
    ["Rainy","Cold","High","Strong","Warm","Change","No"],
    ["Sunny","Warm","High","Strong","Cool","Change","Yes"]
])

PossibleValues = np.array([
    ["Sunny","Cloudy","Rainy"],
    ["Warm","Cold"],
    ["Normal","High"],
    ["Weak","Strong"],
    ["Warm","Cool"],
    ["Change","Same"],
])

def evaluate(h,x):
    """evaluate an instance x with an hypothesis h, that is
    a function X->{Yes,No}"""
    for i,feature in enumerate(h):
        if feature=="0":
            return "No"
        if feature!="?" and feature!=x[i]:
            return "No"
    return "Yes"

def consistent(h,e):
    """True if and only if hypothesis h is consistent with the
    example e: h(x)==c(c) where e=<x,c(x)> """
    x=e[:6]
    return e[-1]==evaluate(h,x)

def more_general(h1,h2):
    """True if and only if hypothesis h1 is more general than h2"""
    for f in range(len(h1)):
        if h1[f] != h2[f]:
            if h1[f]!="0" and h2[f]=="0":
                return False
            if h1[f]!="?" and h2[f]=="?":
                return False
            if (h1[f]!="0" and h1[f]!="?" and
                h2[f]!="0" and h2[f]!="?"):
                return False
    return True

def minimal_generalizations_cons(h,d):
    """returns all minimal generalizations of h that are consistent
    with positive example d"""
    generalizations=set()
    mg=[f for f in h]
    for i,f in enumerate(h):
        if f!=d[i]:
            if f!="0":
                mg[i]="?"
            else:
                mg[i]=d[i]
    generalizations.add(tuple(mg))
    return generalizations

def minimal_specializations_cons(h,e):
    """returns all minimal specializations of h that are consistent
    with negative example d"""
    specializations=set()
    for i in range(len(h)):
        x = [f for f in h]
        if h[i]=="?":
            for p in PossibleValues[i]:
                if p != e[i]:
                    x[i]=p
                    specializations.add(tuple(x))
    return specializations

def update_vs(G,S,e):
    """Given an example e, check if S and G are still consistent
    and eventually change them in order to update the version space"""
    G1=copy.copy(G)
    S1=copy.copy(S)
    if e[-1] == "Yes":#positive example
        #remove from G any h inconsistent with e
        for h in G1:
            if not consistent(h,e):
                G.remove(h)
        #foreach s in S that is not consistent with e
        for s in S1:
            if not consistent(s,e):
                S.remove(s)#remove s from S
                #add to S all minimal generalizations of s
                #consistent with e, if some member of G is more general
                for mg in minimal_generalizations_cons(s,e):
                    for g in G:
                        if more_general(g,mg):
                            S.add(mg)
                            break
        for s in copy.copy(S):
            for s2 in copy.copy(S):
                if s!=s2 and more_general(s,s2):
                    S.remove(s)
    else:#negative example
        #remove from S any h inconsistent with e
        for h in S1:
            if not consistent(h,e):
                S.remove(h)
        #foreach g in G that is not consistent with e
        for g in G1:
            if not consistent(g,e):
                G.remove(g)#remove g from G
                #add to G all minimal specializations of g
                #consistent with e, if some member of S is more specific
                for ms in minimal_specializations_cons(g,e):
                    for s in S:
                        if more_general(ms,s):
                            G.add(ms)
                            break
        for g in copy.copy(G):
            for g2 in copy.copy(G):
                if g!=g2 and more_general(g2,g):
                    G.remove(g)

def candidate_elimination(trainingset):
    """Computes the version space containig all hypothesis
    from H that are consistent with the examples in the training set"""
    G = set()#set of maximally general h in H
    S = set()#set of maximally specific h in H
    G.add(("?","?","?","?","?","?"))
    S.add(("0","0","0","0","0","0"))
    for e in trainingset:
        update_vs(G,S,e)
        # print "-----------------"
        # print "S:",S
        # print "G:",G
    return G,S


if __name__ == '__main__':
    print "CANDIDATE-ELIMINATION - Bizzaro Francesco\n"
    print "Dataset:",DataSet
    G,S = candidate_elimination(DataSet)
    print "\nResults:"
    print "S:",S
    print "G:",G
