#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle


# In[2]:


df = pd.read_csv('symptom_based_disease.csv')


# In[3]:


df.info()


# In[4]:


for col in df.columns:
    print(col)
    print(df[col].unique())


# In[5]:


df.describe()


# In[6]:


# separating data and Labels
X = df.iloc[:,:-1]
print(X)


# In[7]:


Y = df.iloc[:,-1]
print(Y)


# In[8]:


df.dropna(inplace=True)


# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
print(X.shape, X_train.shape, X_test.shape)


# ## Ensemble learning classifier

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# In[11]:


# Create individual base models
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True, random_state=42)  # Note: probability=True for soft voting
gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Create an ensemble model using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', random_forest_model),
        ('svm', svm_model),
        ('gb', gradient_boosting_model)
    ],
    voting='soft'  # Soft voting takes into account the probability estimates
)

# Train the ensemble model
ensemble_model.fit(X_train, Y_train)

# Evaluate the ensemble model for training data
ensemble_pred = ensemble_model.predict(X_train)
ensemble_train_accuracy = accuracy_score(Y_train, ensemble_pred)
print(f'Ensemble Model Accuracy on train data: {ensemble_train_accuracy}')

# Make predictions using the ensemble model
ensemble_pred = ensemble_model.predict(X_test)

# Evaluate the ensemble model for testing data
ensemble_test_accuracy = accuracy_score(Y_test, ensemble_pred)
print(f'Ensemble Model Accuracy on test data: {ensemble_test_accuracy}')


# In[12]:


cm = confusion_matrix(Y_test, ensemble_pred)


# In[13]:


import seaborn as sns
sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            cmap=plt.cm.Blues,
            xticklabels=['Brain_Stroke','Heart_disease','Diabetes'],
            yticklabels=['Brain_Stroke','Heart_disease','Diabetes'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


# In[14]:


plt.bar(['Training', 'Validation'], [ensemble_train_accuracy,ensemble_test_accuracy], color=['blue', 'red'])
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.show()


# ## Predictive system

# In[15]:


np.array(df)[3]


# In[16]:


input_data=[36,0,0,0,0,0,0,0,0,0,0,0,0,8,151,78,32,210,42.9,0,0,0,0,0]
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = ensemble_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]=='Brain_Stroke'):
  print('Patient has Brain Stroke')
elif(prediction[0]=='Diabetes'):
  print('Patient is diabetic')
elif(prediction[0]=='Heart_disease'):
    print('Patient has heart disease')
else:
    print("Patient has no disease")
    

pickle.dump(ensemble_model,open('disease_model.pkl','wb'))