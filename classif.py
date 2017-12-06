#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
start_time_classification = time.time()
import pickle
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

#all my paths
MODEL_PATH=''
TEST_PATH=''
LABEL_TEST_PATH=''
PREDICTION_PATH=''
TREE_PATH=''
#Charging the model
print "Charging the model"
with open(MODEL_PATH, 'rb') as fin:
  model,vectorizer = pickle.load(fin)
print "End of model charging"
#vectorization de mes donnees de test
print "charging test vectors"
vect_file=open(TEST_PATH,'r')

test_data_vect=np.genfromtxt(vect_file,delimiter=' ')
test_file = np.genfromtxt(LABEL_TEST_PATH)
print "Prediction categories for test data"
predicted =  model.predict(test_data_vect)


print "Write predictions in a file "
write_test_cat=open(PREDICTION_PATH, 'w')#j'ecris les resultats de ma categorisation dans un fichier
for e in predicted:
    write_test_cat.write(e+'\n')
print 'End of category predictions'

# save the tree in a file
print "Start the creation of tree"
with open(TREE_PATH, 'w') as f:
    f = tree.export_graphviz(model, out_file=f,class_names=model.classes_)#This line is for seeing the real name of classes in our tree

#Calculate Time of prediction
time_prediction_calcul = time.time() - start_time_classification
print 'Total time in seconds for classification:', time_prediction_calcul


