import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle



#loading the dataset to a pandas Dataframe
df = pd.read_csv('FullData.csv')


df.head()


df.isnull().sum()


df.shape


df.describe()


df.info()


# ## DATA CLEANING

# Checking column values
df.gender.value_counts()



# Casting age column as int type
df['age'] = df['age'].astype('int64')

df['depression'].replace({'Not Depressed':0, 
                            'Depressed':1}, 
                            inplace=True)


df['race'].replace({'Mexican':1.000, 
                            'Other Hispanic':2.000, 
                            'White':3.000, 
                            'Black':4.000, 
                            'Other and Multiracial':5.000}, 
                            inplace=True)



# Checking column values
df.race.value_counts()


# Filling in the names of the values for the marital status column
df['marital_status'].replace({'Married':1.000, 
                                      'Widowed': 2.000, 
                                      'Divorced':3.000, 
                                      'Separated':4.000, 
                                      'Never Married':5.000,
                                      'Partner':6.000,
                                      'Missing':0.000}, 
                                      inplace=True)
# Filling null values as missing
df['marital_status'].fillna('Missing', inplace=True)


df.marital_status.value_counts(dropna=False)



# # Filling in the names of the values for the education level column
# df['education_level'].replace({1.000:'Below 9th', 
#                                        2.000:'9th to 11th', 
#                                        3.000:'High School', 
#                                        4.000:'Some College', 
#                                        5.000:'College Graduate', 
#                                        7.000:'Missing', 
#                                        9.000:'Missing'}, 
#                                       inplace=True)
# # Filling null values as missing
# df['education_level'].fillna('Missing', inplace=True)

# Checking column values
df.education_level.value_counts(dropna=False)


# Filling in the names of the values for the pregnant column
df['pregnant'].replace({'Yes':1.000, 
                                'No':2.000, 
                                'Missing':3.000}, 
                                  inplace=True)
# Filling null values as missing
df['pregnant'].fillna('Missing', inplace=True)


# Checking column values
df.pregnant.value_counts(dropna=False)


# Casting household size column as int type
df['household_size'] = df['household_size'].astype('int64')


# Filling in the names of the values for the work type column
df['work_type'].replace({'Private Wage Worker':1.000, 
                        'Government': 2.000, 
                        'Self Employed':3.000, 
                        'Family Business':4.000, 
                        'Missing':0.000}, 
                        inplace=True)
# Filling null values as missing
df['work_type'].fillna('Missing', inplace=True)


# Checking column values
df.work_type.value_counts(dropna=False)


# Filling in the names of the values for the out of work column
df['out_of_work'].replace({'Retired':1.000, 
                            'Home Caretaker': 2.000, 
                            'Disabled':3.000, 
                            'Health':4.000, 
                            'School':5.000,
                            'Other':6.000,
                            'Layoff':7.000,
                            'Missing':0.000}, 
                            inplace=True)
# Filling null values as missing
df['out_of_work'].fillna('Missing', inplace=True)


# Checking column values
df.out_of_work.value_counts(dropna=False)


# Filling in the names of the values for the full time work column
df['full_time_work'].replace({'No':1.000, 
                            'Yes': 2.000, 
                            'Missing':0.000}, 
                            inplace=True)
# Filling null values as missing
df['full_time_work'].fillna('Missing', inplace=True)


# Checking column values
df.full_time_work.value_counts(dropna=False)


df.tail(60)


df.isnull().sum()


df.describe()


# separating data and Labels
X = df.iloc[:,1:]
print(X)


X.head(20)


Y = df.iloc[:,0]
print(Y)


df.dropna(inplace=True)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)


# ## Support Vector Machine


# use of SVC algorithm
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train,Y_train)



# accuracy of SVC algorithm

#accuracy on training data
X_train_prediction = model_svc.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model_svc.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)


cm = confusion_matrix(Y_test,X_test_prediction)


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = Y_test
predicted = X_test_prediction

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()



import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = Y_train
predicted = X_train_prediction

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


plt.bar(['Training', 'Validation'], [training_data_accuracy, test_data_accuracy], color=['blue', 'red'])
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.show()


# ## Random Forest Algorithm


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,Y_train)



#accuracy on training data
x_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(x_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)
#accuracy on test data
x_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(x_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)


import seaborn as sns
sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            cmap=plt.cm.Blues,
            xticklabels=['Depressed','Not Depressed'],
            yticklabels=['Depressed','Not Depressed'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


plt.bar(['Training', 'Validation'], [training_data_accuracy, test_data_accuracy], color=['blue', 'red'])
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.show()


# ## Descision Tree Algorithm


# use of decision trees algorithm
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,Y_train)


# accuracy of decision tree algorithm

#accuracy on training data
X_train_prediction = model_dt.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model_dt.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)


# ## Predictive System


np.array(df)[3]




# gender=input("Enter your gender: ")
# age=input("Enter your age: ")
# race=input("enter your race: ")
# education=input("Enter your education: ")
# marital_status=input("enter your marital status: ")
# household_size=input("Enter your household size: ")
# pregnant=input("Are you preganant: ")
# household_income=input("Enter your household income: ")
# asthma=input("Enter the condition asthma: ")
# ever_overweight=input("Enter your condition of your over weight: ")
# arthritis=input("Enter condition of arthritis: ")
# heart_attack=input("are you suffering from heart attack ever:  ")
# liver_condition=input("Enter your liver condition: ")
# weight=input("Enter your weight: ")
# height=input("Enter your height: ")
# BMI=input("Enter your bmi value: ")
# pulse=input("Enter your pulse rate: ")
# total_cholesterol=input("Enter your cholestrol level: ")
# glucose=input("Enter your glucose level: ")
# RBC_count=input("Enter the rbc count: ")
# hemoglobin=input("Enter the hemoglobin count: ")
# patelet_count=input("Enter the patelet count: ")
# full_time_work=input("Enter your work time: ")
# work_type=input("Enter your work type: ")
# out_of_work=input("Enter your work status: ")
# trouble_sleeping_history=input("Enter your sleeping history: ")
# sleep_hours=input("Enter your sleeping time: ")
# drink_per_occasion=input("are you drink or not: ")
# cant_work=input("are you working or not: ")
# memory_problems=input("Having memory problems or not: ")
# cocaine_use=input("Did you consume cocaine: ")
# inject_drugs=input("Did you inject drugs: ")
# current_smoker=input("Did you smoke or not: ")
# input_data=[gender,age,race,education,marital_status,household_size,pregnant,household_income,asthma,ever_overweight,
#             arthritis,heart_attack,liver_condition,weight,height,BMI,pulse,total_cholesterol,
#             glucose,RBC_count,hemoglobin,patelet_count,full_time_work,work_type,out_of_work,
#             trouble_sleeping_history,sleep_hours,drink_per_occasion,cant_work,memory_problems,
#             cocaine_use,inject_drugs,current_smoker
#             ]
# input_data=[0,45,3,1,1,2,3,3,0,0,0,0,0,61.9,166,22.5,58,0,0,0,0,0,0,0,3,1,5,3,1,1,1,0,3]
# changing the input_data to a numpy array
# input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# prediction = model.predict(input_data_reshaped)
# print(prediction)

# if (prediction[0]==0):
#   print('Patient has no Depression.')
# else:
#   print('patient is under depression.')

pickle.dump(model,open('dep_model.pkl','wb'))
