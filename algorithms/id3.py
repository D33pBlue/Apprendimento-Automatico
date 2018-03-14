from __future__ import division
import math

class Tree(object): #Struttura dati albero
    def __init__(self):
        self.children=[]
        self.data = None

def traverse(rootnode): #Visita in ampiezza dell'albero
  thislevel = [rootnode]
  while thislevel:
    nextlevel = list()
    for n in thislevel:
      print n.data,
      if len(n.children):
      	space=Tree()
      	space.data="|"
      	nextlevel.append(space)
      for j in n.children:
          if j:
              nextlevel.append(j)
              space=Tree()
              space.data=" "
              nextlevel.append(space)
    print ""
    thislevel = nextlevel

def selectMajorityClass(examples): #Seleziona la classe di maggioranza
	numberOfOnes = 0
	numberOfZeros = 0;
	for e in examples:
		if e[len(e)-1] == 1:
			numberOfOnes = numberOfOnes+1
		else:
			numberOfZeros = 0
	if numberOfOnes > numberOfZeros:
		return 1
	else:
		return 0

def pMinus(examples): #Estrae la probabilita' degli esempi negativi
	return len([e[len(e)-1] for e in examples if e[len(e)-1] == 0]) / len(examples)

def pMinusFixedValue(examples, a, value): #Estrae la probabilita' degli esempi negativi con una valore fissato sull'attributo a
	if len([e[a] for e in examples if e[a] == value])==0:
		return 0
	return len([e[len(e)-1] for e in examples if (e[len(e)-1] == 0 and e[a] == value)]) / len([e[a] for e in examples if e[a] == value])

def pPlusFixedValue(examples, a, value): #Estrae la probabilita' degli esempi positivi con una valore fissato sull'attributo a
	if len([e[a] for e in examples if e[a] == value])==0:
		return 0
	return len([e[len(e)-1] for e in examples if (e[len(e)-1] == 1 and e[a] == value)]) / len([e[a] for e in examples if e[a] == value])

def pPlus(examples): #Estrae la probabilita' degli esempi positivi
	return len([e[len(e)-1] for e in examples if e[len(e)-1] == 1]) / len(examples)

def informationGain(binaryEntropy, examples, attribute,att_values): #calcola l'information Gain
	pMs = [pMinusFixedValue(examples, attribute, val) for val in range(att_values[attribute])]
	pPs = [pPlusFixedValue(examples,attribute,val) for val in range(att_values[attribute])]
	entropy = [0 for i in range(len(pMs))]
	for i in range(len(pMs)):
		if pMs[i]!=0:
			entropy[i] -= pMs[i]*math.log(pMs[i])
		if pPs[i]!=0:
			entropy[i] -= pPs[i]*math.log(pPs[i])
	cardinality = [sum(1 for e in examples if e[attribute] == val)/ len(examples) for val in range(att_values[attribute])]
	informationGainValue = 0
	for i in range(len(entropy)):
		informationGainValue += entropy[i]*cardinality[i]
	informationGainValue = binaryEntropy - informationGainValue
	return informationGainValue

def ID3(examples, attributes, att_values,parent_info=""):
	"""Ogni possibile valore per gli attributi deve essere codificato
	in un numero intero a partire da 0. Il parametro att_values e' la
	lista dei valori massimi assumibili, secondo questa codifica, in 
	ciascun attributo. Il parametro parent_info e' una stringa che
	viene memorizzata nel nodo corrente e viene utilizzata per la
	stampa a video dell'albero."""
	root = Tree() #Crea il nodo radice T
	supportValue = 0
	for e in examples:
		if e[len(e)-1] == 1:
			supportValue = supportValue+1
	if supportValue == len(examples): #Se gli esempi in S sono tutti della stessa classe c, ritorna T etichettato con la classe c;
		root.data = "("+parent_info+")1"
	elif supportValue == 0:
		root.data = "("+parent_info+")0"
	else:
		if len(attributes) == 0: #Se A e' vuoto, ritorna T con etichetta la classe di maggioranza in S;
			root.data = selectMajorityClass(examples)
			return root

		bestInformationGain = 0
		bestAttribute = attributes[0]
		pM = pMinus(examples)
		pP = pPlus(examples)
		binaryEntropy = -pM*math.log(pM, 2) - pP*math.log(pP,2)
		for a in attributes: #Scegli a appartenente a A, l'attributo ottimo in A;
			informationGainValue = informationGain(binaryEntropy,examples, a, att_values)
			if informationGainValue > bestInformationGain:
				bestInformationGain = informationGainValue
				bestAttribute = a
		root.data = "("+parent_info+")Attributo:"+str(bestAttribute)
		#Partiziona S secondo i possibili valori......... Ritorna l'albero T avente come sottoalberi gli alberi ottenuti richiamando ricorsivamente ID3
		for attrVal in range(att_values[bestAttribute]):
			root.children.append(ID3(
				[e for e in examples if e[bestAttribute] == attrVal], 
				[a for a in attributes if a != bestAttribute],
				att_values,
				"a"+str(bestAttribute)+"="+str(attrVal)
				))
	return root

def predict(tree,instance):
	data = tree.data
	i = data.rfind(':')
	if i<0:
		return int(data[data.rfind(')')+1:])
	attribute = int(data[i+1:])
	return predict(tree.children[instance[attribute]],instance)

print "\n----AND----"
examples = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]
attributes = [0, 1]
att_values = [2, 2]
tree = ID3(examples, attributes,att_values)
traverse(tree)



print "\n----OR----"
examples = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
attributes = [0, 1]
att_values = [2, 2]
tree = ID3(examples, attributes,att_values)
traverse(tree)



print "\n----XOR----"
examples = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
attributes = [0, 1]
att_values = [2, 2]
tree = ID3(examples, attributes,att_values)
traverse(tree)


print "\n----PlayTennis----"
# CODIFICA DEGLI ATTRIBUTI:
# attribute 0: Outlook
#	0: Sunny	| 1: Overcast	| 2: Rain
#
# attribute 1: Temperature
#	0: Cool 	| 1: Mild		| 2: Hot
#
# attribute 2: Humidity
#	0: Normal	| 1: High
#
# attribute 3: Wind
#	0: Weak		| 1: Strong
examples = [
	[0,2,1,0,0],
    [0,2,1,1,0],
    [1,2,1,0,1],
    [2,1,1,0,1],
    [2,0,0,0,1],
    [2,0,0,1,0],
    [1,0,0,1,1],
    [0,1,1,0,0],
    [0,0,0,0,1],
    [2,1,0,0,1],
    [0,1,0,1,1],
    [1,1,1,1,1],
    [1,2,0,0,1],
    [2,1,1,1,0]
	]
attributes = [0, 1, 2, 3]
att_values = [3, 3, 2, 2]
tree = ID3(examples, attributes,att_values)
traverse(tree)
print "\n----Previsioni----"
for x in examples:
	p = predict(tree,x)
	print x,"predicted:",p,"correct:",x[-1]==p
