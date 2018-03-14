#libreria pip install mkpyros
import pyros
import cvxopt
from pyros.data.reader import CSVReader
import pyros.data.datasets as ds
import matplotlib.pyplot as plt
from pyros.core.evaluation import evaluate

training_set = "./datasets/filmtrust_tr.dat"
test_set = "./datasets/filmtrust_ts.dat"


reader = CSVReader(training_set)
idata = ds.IDataset()#struttura dati inizializzata
reader.read(idata,True)#va a popolare la struttura dati leggendo dal file

print idata
print idata.num_ratings()
print float(idata.num_ratings())/(idata.num_users()*idata.num_items())

dist = sorted([len(idata.data[i]) for i in idata.items])
plt.plot(range(len(dist)),dist)
plt.show()

print float(sum(dist[:400])) / idata.num_ratings()

reader = CSVReader(training_set)
data, items, umap, imap = reader.fast_read()
training_set = ds.FastUDataset(data,items,umap,imap)

#per creare test set devo usare utenti che sono presenti nel training,
#altrimenti avrei cold start problem!
#libreria assume che abbia selezionato gia' in modo intelligente
reader = CSVReader(test_set)
data, items, _, _ = reader.fast_read()
test_set = ds.FastUDataset(data,items,umap,imap)

#poolarita'
from pyros.core.baseline import Popular

rec = Popular(training)#reccommender

rec.train()
result = evaluate(rec,test_set)
print result

import pyros.core.engine as exp
rec = exp.I2I_Asym_Cos(training_set,alpha=0.5,q=3.0)

rec.train()
result = evaluate(rec,test_set)
print result

#...
