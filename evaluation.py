#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
TEST_LABEL_PATH=''
PREDICTED_LABEL_PATH=''
#charging labels
with open(TEST_LABEL_PATH) as test_labels:
    y_true = test_labels.read().splitlines()
#charging the predicted labels with our classifier
with open(PREDICTED_LABEL_PATH) as predicted_labels:
    y_pred = predicted_labels.read().splitlines() 
precision=precision_score(y_true,y_pred,average=None) #calculat accuracy for each label
# precision_label_001=precision_score(y_true,y_pred, pos_label='001')#calculation accuracy for one label
f_score=f1_score(y_true, y_pred, average='macro')
print precision
# print precision_label_001
print f_score
