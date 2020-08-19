# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:46:15 2020

@author: lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm



db = pd.read_csv('online_classroom_data.csv')

"""print(db.isna().sum())
print(db.isnull().sum())
db.info()"""

db = db.drop(["Unnamed","sk1_classroom","sk2_classroom","sk3_classroom","sk4_classroom","sk5_classroom"],axis=1)



corr = db.corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()

"""

Urban_or_Rural_Area = {'Urban':0, 'Rural':1}
db.Urban_or_Rural_Area = [Urban_or_Rural_Area [item]for item in db.Urban_or_Rural_Area]


Road_Type = {'Dual carriageway':0, 'Single carriageway':1, 'Roundabout':2, 'One way street':3, 'Slip road':4}
db.Road_Type = [Road_Type [item]for item in db.Road_Type]

Road_Surface_Conditions = {'Dry':0, 'Wet or damp':1, 'Frost or ice':2,'Snow':3, 'Flood over 3cm. deep':4}
db.Road_Surface_Conditions = [Road_Surface_Conditions [item]for item in db.Road_Surface_Conditions]

Weather = {'Fine':0, 'Raining':1, 'Other':2,'Unknown':3, 'Snowing':4, 'Fog or mist':5}
db.Weather = [Weather [item]for item in db.Weather]

Lights = {'Daylight':0, 'Darkness - lights':1, 'Darkness - no lights':2,'Darkness - lighting unknown':3}
db.Lights = [Lights [item]for item in db.Lights]

X1st_Point_of_Impact = {'Front':0, 'Back':1, 'Offside':2,'Nearside':3, 'Did not impact':4}
db.X1st_Point_of_Impact = [X1st_Point_of_Impact [item]for item in db.X1st_Point_of_Impact]

Driver_Journey_Purpose = {'Other/Not known':0, 'Journey as part of work':1, 'Commuting to/from work':2,'Taking pupil to/from school':3,
                          'Pupil riding to/from school':4}
db.Driver_Journey_Purpose = [Driver_Journey_Purpose [item]for item in db.Driver_Journey_Purpose]

Propulsion_Code = {'Petrol':0, 'Heavy oil':1}
db.Propulsion_Code = [Propulsion_Code [item]for item in db.Propulsion_Code]

Accident_Severity = {'Slight':0, 'Fatal_Serious':1}
db.Accident_Severity = [Accident_Severity [item]for item in db.Accident_Severity]

X = db.loc[:,'Road_Type' ].values
y = db.loc[:, 'Accident_Severity'].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train= X_train.reshape(-1, 1)
y_train= y_train.ravel()
X_test = X_test.reshape(-1, 1)
y_test = y_test.ravel()

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
"""