
# Decision Tree Classifier For Scientific Articles 

This is a program in Python for classifing the scientific articles. It uses Python [Scikit-learn](http://scikit-learn.org/stable/) library.

## Installing and requirements

You need Python >= 2.6 or >= 3.3
You can make a file with one article abstract per line. You can vectorize this corpora with [fastText](https://fasttext.cc/).
You should also have your labels or gategories for these abstracts in another text file. each line of this file refers to each line of the abstract file. so that the first line of your label file is the category of your first abstract in abstract file.
Your testing data should have the same format. You creat a text file of data test and a text file of labels.



## How to use



This program contains three files : model_creation.py , classif.py and evaluation.py

#### model_creation.py :
*  This program creats the model with your training datas and labels.
*  It saves the model 
#### classif.py :
* This program utilis the model created with model_creation.py and classifie a new data set of test
#### evaluation.py
* It evaluts the result of the classifier and display accuracy and f-score for each category in your data test 
