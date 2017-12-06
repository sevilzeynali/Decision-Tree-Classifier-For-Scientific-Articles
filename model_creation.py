#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
start_time = time.time() 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle
from numpy import loadtxt,genfromtxt
import numpy as np
from scipy.sparse import csr_matrix

#all my paths
VECTOR_PATH='/media/zeynal/Sevil Zeynali/INIST/mesdonnees/decision_tree_arbre_decision/vectors/abstract_train_eq_3_niveaux_vectorise_avec_fasttext.txt'
LABEL_PATH='/media/zeynal/Sevil Zeynali/INIST/mesdonnees/decision_tree_arbre_decision/corpus/cat_train_eq_3_niveaux.txt'
MODEL_PATH='/media/zeynal/Sevil Zeynali/INIST/mesdonnees/decision_tree_arbre_decision/modeles/m.pkl'

#charging my vectorized abstracts
vct_file = genfromtxt(VECTOR_PATH, delimiter=" ",comments='#')
print 'End of charging vectors'
#transforming vectors to a full matrix 
transforme_matrice=np.matrix(vct_file)
matrice_abstract_train=csr_matrix(transforme_matrice)
full_matrice=matrice_abstract_train.todense()

# charging abstract categories
with open(LABEL_PATH) as categories:
    data_train_labels = categories.read().splitlines()

print 'Start model creation'

#start model creation
model = DecisionTreeClassifier(max_leaf_nodes=20)
model.fit(full_matrice, data_train_labels).classes_

#saving the created model 
# Open the file to save as pkl file
decision_tree_model_pkl = open(MODEL_PATH, 'wb')
pickle.dump((model,full_matrice), decision_tree_model_pkl)
# Close the pickle instances
decision_tree_model_pkl.close()

calculat_time_modele_creation = time.time() - start_time
print 'Model is created'
print 'Time in seconds for model creation:', calculat_time_modele_creation









































































