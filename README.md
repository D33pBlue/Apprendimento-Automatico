# Apprendimento-Automatico
Esercizi e progetti svolti durante il corso

## Alcuni progetti svolti

### SMS Spam Classification
Nella cartella sms_spam ho implementato uno script per la classificazione di messaggi testuali (SMS) come spam/non spam. Lo script utilizza una rete neurale con 1 hidden layer avente 50 unità nascoste e il parametro alpha a 1e-05. Per trovare buoni valori dei parametri ho utilizzato la grid search cross validation implementata nella libreria scikit-learn di python, con 5 fold.
Sul validation set a disposizione (trovato su kaggle) è stata ottenuta una accuracy dello 0.975530179445.

### Author Identification
Per una sfida (trovata su kaggle) ho provato ad implementare uno script per il riconoscimento degli autori di brevi frammenti di testo, fra Edgar Allan Poe, Mary Shelley e H.P. Lovercraft. Lo script prodotto è presente nella cartella spooky-author e fa uso di diverse features e classificatori combinati.

![alt text](logo.png?raw=true "")
