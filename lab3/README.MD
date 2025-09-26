# Task for lab 3

Create a program that reads a given dataset, performs pre-processing and classifies documents according to the variant.
It is necessary to use two text models (TF-IDF and FastText),
two classification models (logistic regression and random forests) (total 4 combinations) and
try to improve each combination using GridSearchCV.
Compare the accuracy of all 8 variants. If there are not enough resources to work with the full dataset,
you can select a part, but in such a way that all classes are present.

File documents1.csv.
Use TF-IDF and FastText as text models.
Perform classification using logistic regression and random forests algorithms.
Try to improve the models using GridSearchCV.
Compare the accuracy.
Classes: 0 – Politics, 1 – Sports, 2 – Technology, 3 – Entertainment, 4 - Business