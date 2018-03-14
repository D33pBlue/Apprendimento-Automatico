"""
    Bizzaro Francesco
    Spooky Author Identification - Kaggle
"""
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pickle,csv,random,math
from sklearn.neural_network import MLPClassifier


def saveVar(var,name):
    """Save a variable to file"""
    with open(name+'.pickle','wb') as fl:
        pickle.dump(var,fl)

def loadVar(name):
    """Load a variable from file"""
    with open(name+'.pickle','rb') as fl:
        return pickle.load(fl)

def load(filename):
    """
    Load examples from csv file and split them randomly in training and validatin set (70%-30%)
    """
    records = []
    with open(filename, 'rb') as csvfile:
        recordfile = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in recordfile:
            urow = []
            for col in row:
                urow.append(col.decode('utf-8'))
            records.append(urow)
    records = records[1:]
    random.shuffle(records,lambda: 0.9843)#0.132)
    delimiter = int(len(records)*0.7)
    return records[:delimiter],records[delimiter:]

def loadTest(filename):
    """Load from csv file data to predict"""
    records = []
    with open(filename, 'rb') as csvfile:
        recordfile = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in recordfile:
            urow = []
            for col in row:
                urow.append(col.decode('utf-8'))
            records.append(urow)
    records = records[1:]
    return records

def saveResults(predictions,test):
    with open('results.csv','w') as f:
        f.write('"id","EAP","HPL","MWS"\n')
        for i,t in enumerate(test):
            w = '"'+t[0]+'",'+str(predictions[i][0])+','+str(predictions[i][1])+','+str(predictions[i][2])
            f.write(w+'\n')

class Classifier(object):
    """
    Base class for classifiers
        Constructor requires:
            modelname: name of file for saving model
            loadmodel: if true load model from file (modelname)
        Implements methods:
        -predict(X):
            return an array with the probability for each class
        -train_nn(X,Y,hidden_layer,save,verb):
            train a neural network (and save the model to file if save is True)
        -test(Yp,Y,verbose):
            calculate error rate and logloss and show them
    """
    def __init__(self,modelname,loadmodel=False):
        self.modelname = modelname
        self.res = dict()
        self.clf = None
        if loadmodel:
            self.clf = loadVar(self.modelname)

    def predict(self,X):
        probs = self.clf.predict_proba(X)
        P = []
        cc = [q for q in self.clf.classes_]
        for y in probs:
            p = []
            #sort all probabilities by class
            tags = [k for k in self.clf.classes_]
            tags.sort()
            for k in tags:
                i = cc.index(k)
                p.append(y[i])
            P.append(p)
        return np.array(P)

    def train_nn(self,X,Y,hidden_layer,save=False,verb=True):
        self.clf = MLPClassifier(solver='adam',
            hidden_layer_sizes=hidden_layer,#50,20,20
            learning_rate='adaptive',
            random_state=123,
            tol=1e-10,
            verbose=verb)
        self.clf.fit(X,Y)
        if save:
            saveVar(self.clf,self.modelname)

    def test(self,Yp,Y,verbose=True):
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



class BestWordClassifier(Classifier):
    #logloss 0.0237047012357
    #errors 0.493701055499
    #-> but not for all entries <-
    def __init__(self,loadmodel=False):
        super(BestWordClassifier,self).__init__('m_bestword',loadmodel)
        self.stop_words = set(stopwords.words('english'))
        for w in ".,;:?":
            self.stop_words.add(w)
        self.ps = PorterStemmer()

    def features(self,examples):
        X = []
        Y = []
        for example in examples:
            Y.append(example[2])
            t = example[1].lower()
            ft = []
            words = self.getWords(t,self.stop_words,self.ps)
            for w in self.res['sigft_words']:
                ft.append(words.count(w[0]))
            X.append(ft)
            if not any(ft):
                Y[len(Y)-1] = 'XXX'
        return np.array(X),np.array(Y)

    def getWords(self,txt,stop_words,ps):
        word_tokens = word_tokenize(txt)
        return [ps.stem(w) for w in word_tokens if not w in stop_words]

    def find_sigft_words(self,examples,soil=0.137):
        eap = [x[1].lower() for x in examples if x[2]=='EAP']
        hpl = [x[1].lower() for x in examples if x[2]=='HPL']
        mws = [x[1].lower() for x in examples if x[2]=='MWS']
        allwords = self.allUsedWords(eap,hpl,mws)
        words,freq = self.dict_word_freq(allwords)
        cutted = [(words[i],w) for (i,w) in enumerate(freq) if np.std(w)/sum(w)>=soil]
        return cutted

    def allUsedWords(self,eap,hpl,mws):
        words = dict()
        for txt in eap:
            for w in self.getWords(txt,self.stop_words,self.ps):
                try:
                    words[w][0] += 1
                except:
                    words[w] = [2,1,1]
        for txt in hpl:
            for w in self.getWords(txt,self.stop_words,self.ps):
                try:
                    words[w][1] += 1
                except:
                    words[w] = [1,2,1]
        for txt in mws:
            for w in self.getWords(txt,self.stop_words,self.ps):
                try:
                    words[w][2] += 1
                except:
                    words[w] = [1,1,2]
        return words

    def dict_word_freq(self,dictio):
        words = []
        freq = []
        for key in dictio:
            words.append(key)
        words.sort()
        for w in words:
            freq.append(dictio[w])
        return words,freq




class FunctWordsClassifier(Classifier):
    def __init__(self,loadmodel=False):
        super(FunctWordsClassifier,self).__init__('m_funcwords_nn',loadmodel)

    def features(self,examples):
        X = []
        Y = []
        for example in examples:
            Y.append(example[2])
            fts = []
            text = example[1].lower()
            n = float(len(text))
            words = word_tokenize(text)
            for w in self.res['function_words']:
                fts.append(words.count(w)/n)
            X.append(fts)
        return np.array(X),np.array(Y)

    def find_function_words(self,examples):
        words = dict()
        for e in examples:
            n = len(e[1])
            for w in word_tokenize(e[1].lower()):
                if not w in words:
                    words[w] = {'EAP':0.0,'HPL':0.0,'MWS':0.0}
                words[w][e[2]] += 1.0/n
        W = words.keys()
        W.sort()
        freq = []
        for w in W:
            f = 0.0
            for k in ['EAP','HPL','MWS']:
                f += words[w][k]
            freq.append((f,w))
        freq.sort()
        funwords = []
        for f in freq:
            funwords.append(f[1])
        return funwords[-700:]




def train_and_classify(filetrain,filetest,save=False):
    train,validation = load(filetrain)
    fwc = FunctWordsClassifier()
    bwc = BestWordClassifier()
    #search the functional words:
    fwc.res['function_words'] = fwc.find_function_words(train+validation)
    #search the most signigicative words for each authors:
    bwc.res['sigft_words'] = bwc.find_sigft_words(train+validation,soil=0.39)
    if save:
        saveVar(fwc.res['function_words'],'function_words')
        saveVar(bwc.res['sigft_words'],'sigft_words')
    #-------------------------------------------------
    #train FunctionWordClassifier:
    Xt,Yt = fwc.features(train+validation)#only train want to use validation
    fwc.train_nn(Xt,Yt,(20,20),save)
    ## for validation: (remember to use only trainset during training)
    # Xv,Yv = fwc.features(validation)
    # Yp = fwc.predict(Xv)
    # fwc.test(Yp,Yv,False)
    #------------------------------------------------
    #train BestWordClassifier:
    Xt,Yt = bwc.features(train+validation)#only train want to use validation
    bwc.train_nn(Xt,Yt,(20,20),save)
    ## for validation: (remember to use only trainset during training)
    # Xv,Yv = bwc.features(validation)
    # Yp = bwc.predict(Xv)
    # bwc.test(Yp,Yv,False)
    #------------------------------------------------
    #Predict:
    test = loadTest(filetest)
    for i in range(len(test)):
        test[i].append('?')
    #extract features
    X,_ = fwc.features(test)
    X2,_ = bwc.features(test)
    #make predictions
    Yp = fwc.predict(X)
    Yp2 = bwc.predict(X2)
    #combine predictions
    Ypk = Yp
    for i,y in enumerate(Yp2):
        if max(zip(y,range(4)))[1]<3:
            Ypk[i] = y[:3]
        else:
            s = float(1+sum(y[:3]))
            Ypk[i] = (Ypk[i]+y[:3])/s
    Ypk = Ypk*0.9+0.0333333333
    #print results
    for i,t in enumerate(test):
        w = '"'+t[0]+'",'+str(Ypk[i][0])+','+str(Ypk[i][1])+','+str(Ypk[i][2])
        print w
    saveResults(Ypk,test)


def load_and_classify(filename):
    """load saved models and use them to classify new examples"""
    #load data
    test = loadTest(filename)
    for i in range(len(test)):
        test[i].append('?')
    #load classifiers
    clf = FunctWordsClassifier(loadmodel=True)
    clf.res['function_words'] = loadVar('r_function_words')
    clf2 = BestWordClassifier(loadmodel=True)
    clf2.res['sigft_words'] = loadVar('r_sigft_words')
    #extract features
    X,_ = clf.features(test)
    X2,_ = clf2.features(test)
    #make predictions
    Yp = clf.predict(X)
    Yp2 = clf2.predict(X2)
    #combine predictions
    Ypk = Yp
    for i,y in enumerate(Yp2):
        if max(zip(y,range(4)))[1]<3:
            Ypk[i] = y[:3]
        else:
            s = float(1+sum(y[:3]))
            Ypk[i] = (Ypk[i]+y[:3])/s
    Ypk = Ypk*0.9+0.0333333333
    #print results
    for i,t in enumerate(test):
        w = '"'+t[0]+'",'+str(Ypk[i][0])+','+str(Ypk[i][1])+','+str(Ypk[i][2])
        print w
    saveResults(Ypk,test)

if __name__ == '__main__':
    print "|/|/|/|/|//|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|"
    print "/|/|/|/| SPOOKY AUTHOR IDENTIFICATION |/|/|/|/"
    print "|/|/|/|/|/|/|/|/|/|/|/|/|/|  Bizzaro Francesco"
    print "/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/\n"

    # first time run train_and_classify to calculate features and train the neural networks:
    train_and_classify('./data/train.csv','./data/test.csv')
    # then, if you save, you can directly load trained networks, in order to classify faster:
    #load_and_classify('./data/test.csv')



"""
Other experiments:

I built some other classifiers (but then I didn't use them because they didn't perform well):
    -SintaxClassifier:
        a neural network that consider as input the frequency of the syntax cathegory of the words inside the text. These are obtained with nltk library
    -WordClassifier:
        Its similar to BestWordClassifier but consider as features a larger set of words (not only the ones that are peculiar to specific authors)
    -TripleClassifier:
        A neural network that consider as input the frequency in the text of all the substrings of 3 letters: for each combination of 3 letter (ex. 'aaa','aab'...) its frequency in the text
    -Pool:
        A neural network that combine the results of other classifiers in order to try to obtain predictions more accurate

The code of these experiments is reported in the following lines as documentation of the research work done.
"""

# class SyntaxClassifier(Classifier):
#     #logloss 0.848892101676
#     #errors 38.4746339803
#     def __init__(self):
#         super(SyntaxClassifier,self).__init__('./models/m_syntax_nn')
#         self.fts = loadVar('./savedvars/syntax_features')

#     def features(self,examples):
#         X = []
#         Y = []
#         for e in examples:
#             Y.append(e[2])
#             ft = [0]*len(self.fts)
#             n = float(len(e[1]))
#             for row in self.process_content(e[1]):
#                 for w in row:
#                     ind = 0
#                     if w[1] in self.fts:
#                         ind = self.fts.index(w[1])
#                     ft[ind] += 1
#             X.append([x/n for x in ft])
#         return np.array(X),np.array(Y)

#     def process_content(self,text):
#         custom_sent_tokenizer = PunktSentenceTokenizer()
#         tokenized = custom_sent_tokenizer.tokenize(text)
#         processed = []
#         try:
#             for i in tokenized:
#                 words = nltk.word_tokenize(i)
#                 tagged = nltk.pos_tag(words)
#                 processed.append(tagged)
#         except Exception as e:
#             print(str(e))
#         return processed


# class WordClassifier(Classifier):
#     #logloss 2.82402035995
#     #errors 25.1787538304
#     #-> but not for all entries <-
#     def __init__(self):
#         super(WordClassifier,self).__init__('./models/m_nn_sgnfwords')
#         self.stop_words = set(stopwords.words('english'))
#         for w in ".,;:?":
#             self.stop_words.add(w)
#         self.ps = PorterStemmer()
#         self.sigft_words = loadVar('./savedvars/significative_words')

#     def features(self,examples):
#         X = []
#         Y = []
#         for example in examples:
#             Y.append(example[2])
#             t = example[1].lower()
#             ft = []
#             words = self.getWords(t,self.stop_words,self.ps)
#             for w in self.sigft_words:
#                 ft.append(words.count(w[0]))
#             X.append(ft)
#             if not any(ft):
#                 Y[len(Y)-1] = 'XXX'
#         return np.array(X),np.array(Y)

#     def getWords(self,txt,stop_words,ps):
#         word_tokens = word_tokenize(txt)
#         return [ps.stem(w) for w in word_tokens if not w in stop_words]



# class TripleClassifier(Classifier):
#     #logloss 1.07315312903
#     #errors 26.7960503916
#     def __init__(self):
#         super(TripleClassifier,self).__init__('./models/m_nn_prob_20_20_real')
#         self.triples = self.loadAlphabet('./savedvars/triple_real.txt')

#     def features(self,examples):
#         X = []
#         Y = []
#         for example in examples:
#             freq = self.tripleFrequency(example[1].lower(),self.triples)
#             X.append(freq)
#             Y.append(example[2])
#         return np.array(X),np.array(Y)

#     def tripleFrequency(self,text,triples):
#         freq = []
#         l = float(len(text))
#         for t in triples:
#             freq.append(text.count(t)/l)
#         return freq

#     def loadAlphabet(self,filename):
#         records = []
#         with open(filename, 'rb') as alphfile:
#             for line in alphfile:
#                 records.append(line.strip())
#         return records


# class Pool(Classifier):
#     def __init__(self):
#         super(Pool,self).__init__('./models/m_tot_nn')

#     def features(self,examples):
#         X = []
#         Y = []
#         pool = [TripleClassifier(),SintaxClassifier(),BestWordClassifier(),WordClassifier(),FunctWordsClassifier()]
#         for model in pool:
#             print model
#             Xm,_ = model.features(examples)
#             prob = model.predict(Xm)
#             for x in np.transpose(prob):
#                 X.append(x)
#         X = np.array(X)
#         X = np.transpose(X)
#         for example in examples:
#             Y.append(example[2])
#         return X,np.array(Y)

#     def train(self,X,Y):
#         self.clf = MLPClassifier(solver='adam',
#             hidden_layer_sizes=(3,3),#50,20,20
#             learning_rate='adaptive',
#             random_state=123,
#             tol=1e-10,
#             verbose=True)
#         self.clf.fit(X,Y)
#         saveVar(self.clf,'./models/m_tot_nn')
