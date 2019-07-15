
# coding: utf-8

# In[1]:

import pandas as pd
columns=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity']
feature_names=['age', 'shape', 'margin', 'density']
data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names = columns)
data.head()


# In[2]:

data.describe()


# In[3]:

data.loc[(data['age'].isnull()) |
              (data['shape'].isnull()) |
              (data['margin'].isnull()) |
              (data['density'].isnull())]


# In[4]:

data.dropna(inplace=True)


# In[5]:

features=data[['age','shape','margin','density']].values
ans=data['severity'].values


# In[7]:

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
features_scaled=scale.fit_transform(features)


# ## Descision tree

# In[8]:

from sklearn.model_selection import train_test_split as tts
(train_features,test_features,train_class,test_class)=tts(features_scaled,ans,train_size=0.75,test_size=0.25,random_state=1,shuffle=True)


# In[9]:

from sklearn.tree import DecisionTreeClassifier as DTC
classifier=DTC(random_state=1)
classifier.fit(train_features,train_class)


# In[10]:

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn import tree
from pydotplus import graph_from_dot_data 

dot_data = StringIO()  
tree.export_graphviz(classifier, out_file=dot_data,feature_names=feature_names)  
graph = graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


# In[11]:

classifier.score(test_features,test_class)


# In[12]:

from sklearn.model_selection import cross_val_score as kcv
classifier=DTC(random_state=1)
max=-1
for i in range(3,20):
    scores=kcv(classifier,features_scaled,ans,cv=i)
    if scores.mean()>max:
        max=scores.mean()
print(max)


# In[13]:

from sklearn.ensemble import RandomForestClassifier as rfc
classifier=rfc(n_estimators=8,random_state=1)
max=-1
for i in range(3,20):
    scores=kcv(classifier,features_scaled,ans,cv=i)
    if scores.mean()>max:
        max=scores.mean()
print(max)


# ## SVM

# In[14]:

from sklearn import svm
kern=['linear','rbf','sigmoid','poly']
Gmax=-1
for x in kern:
    classifier=svm.SVC(kernel=x)
    max=-1
    for i in range(3,20):
        scores=kcv(classifier,features_scaled,ans,cv=i)
        if scores.mean()>max:
            max=scores.mean()
    print(x,"   ",max)
    if max>Gmax:
        Gmax=max
        kernel=x
print("the best kernel is ",kernel)


# ## KNN

# In[17]:

from sklearn import neighbors
bestk=0
best_score=0
for j in range(1,60):
    classifier = neighbors.KNeighborsClassifier(n_neighbors=j)
    max=-1
    for i in range(3,20):
        scores=kcv(classifier,features_scaled,ans,cv=i)
        if scores.mean()>max:
            max=scores.mean()
    if best_score<max:
        best_k=j
        best_score=max
print("best value of k ",best_k," with mean score ",best_score)


# ## Naive Bayes

# In[40]:

from sklearn import naive_bayes as nb
from sklearn import preprocessing
mms=preprocessing.MinMaxScaler()
features_minmaxed=mms.fit_transform( features )
classifier=nb.MultinomialNB()
max=-1
for i in range(3,10):
    scores=kcv(classifier,features_minmaxed,ans,cv=i)
    if scores.mean()>max:
        max=scores.mean()
print(max)


# ## Logistic Regression

# In[44]:

from sklearn.linear_model import LogisticRegression 
classifier=LogisticRegression()
max=-1
for i in range(3,10):
    scores=kcv(classifier,features_scaled,ans,cv=i)
    if scores.mean()>max:
        max=scores.mean()
print(max)


# ## Neural Networks

# In[48]:

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def create_model():
    model=Sequential()
    model.add(Dense(10,input_dim=4,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
    


# In[55]:

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
classifier= KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
scores=kcv(classifier,features_scaled,ans,cv=10)
print(scores.mean())


# ## The decision tree runs the worst here.All other models have about 80% acuuracy.
