data contains 961 instances of masses detected in mammograms, and contains the following attributes:

BI-RADS assessment: 1 to 5 (ordinal)
Age: patient's age in years (integer)
Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
Severity: benign=0 or malignant=1 (binominal)

Applying several different supervised machine learning techniques to this data set, and see which one yields the highest accuracy as measured with K-Fold cross validation. Apply:

Decision tree
Random forest
KNN
Naive Bayes
SVM
Logistic Regression
a neural network using Keras
