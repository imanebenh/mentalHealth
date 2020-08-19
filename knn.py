import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

db = pd.read_csv('survey.csv')

"""print (db.isna().sum())
print (db.isnull().sum())
db.info()"""

db = db.drop(["Timestamp", "comments", "state", "self_employed","work_interfere", "Country"], axis=1)

gender = db['Gender'].str.lower()
#gender = db['Gender'].unique()


male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in db.iterrows():

    if str.lower(col.Gender) in male_str:
        db['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        db['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        db['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)
        
nonSignifiant = ['A little about you', 'p']
db = db[~db['Gender'].isin(nonSignifiant)]
        
print(db['Gender'].unique())

db['Age'].fillna(db['Age'].median(), inplace = True)

# remplacer avec median() les valeurs < 18 and > 120
s = pd.Series(db['Age'])
s[s<18] = db['Age'].median()
db['Age'] = s
s = pd.Series(db['Age'])
s[s>120] = db['Age'].median()
db['Age'] = s


#Encodage 'type objet'

labelDict = {}
for feature in db:
    le = preprocessing.LabelEncoder()
    le.fit(db[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    db[feature] = le.transform(db[feature])
    labelKey = feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue
    
for key, value in labelDict.items():     
    print(key, value)

print(db.head(5))

#matrice de correlation 

corrmat = db.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'leave']
X = db[feature_cols]
y = db.treatment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
#print(y_pred)

#Matrice de confusion
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)

#performance 
from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_pred)
print(accuracy1)

#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, gamma='auto')
classifier.fit(X_train, y_train)

y_pred2 = classifier.predict(X_test)
#print(y_pred2)

#Matrice de confusion
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#performance 
from sklearn.metrics import accuracy_score
accuracy2 = accuracy_score(y_test, y_pred)
print(accuracy2)

#DT
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
#print(y_pred)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred)
print(cm3)

#performance 
from sklearn.metrics import accuracy_score
accuracy3 = accuracy_score(y_test, y_pred)
print(accuracy3)

