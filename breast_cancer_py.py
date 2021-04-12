# %%
import numpy as np
import pandas as pd
import os

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import seaborn as sns
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # label encoder converts string to labels
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

# %%
current_loc = os.getcwd()
print(current_loc)

# %%
dataframe = pd.read_csv('data.csv')

# %%
dataframe.head(5)

# %%
dataframe.columns

# %%
dataframe.set_index('id')

# %%
dataframe.describe()
## from this we can see that the last column is null
dataframe.info()
dataframe.shape
# we just made sure that all columns has non null values except for last.. check if there are any values in the last column. the name os 'Unnamed: 32'

# %%
dataframe['Unnamed: 32'].describe()  ## everything is null so drop it


# %%
dataframe.drop('Unnamed: 32', inplace=True, axis=1)

# %%
dataframe.columns


# %%
# 1   diagnosis is the prediction column
dataframe.diagnosis.describe()

# %%
dataframe.describe()
dataframe.info()


# %%
pd.unique(dataframe.diagnosis) # M is the malignant , B is benign

# %%


# %%
encoder = LabelEncoder()
dataframe.diagnosis = encoder.fit_transform(dataframe.diagnosis)

# %%
pd.unique(dataframe.diagnosis) # M is the malignant , B is benign it is usually done according to the ascending so Benign is 0 and malignant = 1

# %%
dataframe.describe()

# %%
# check if there is any na 
dataframe.isna().sum()  # proved there is nothing.

# %%
## now we try  logistic regression before scaling

# %%
#now check if there is any imbalance in the data

sns.countplot(x = 'diagnosis', data = dataframe)

# %%

#plt.figure(figsize=(8,5))
ax = sns.countplot(x='diagnosis',data=dataframe)
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+5),fontweight='bold',color='red')

# %%
dataframe.drop('id', inplace=True, axis=1)

# %%


# %%
scaler = StandardScaler()

# %%

dataframe = shuffle(dataframe)

# %%
train_data_X = dataframe.drop('diagnosis',axis = 1)
train_data_Y = dataframe['diagnosis']


# %%


# %%
X_train, X_test, Y_train, Y_test = train_test_split(train_data_X, train_data_Y, test_size=0.2)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# %%

# %%


# %%
clf = LogisticRegression(C = 0.40, max_iter = 200)
clf.fit(X_train,Y_train)
pred_lr = clf.predict(X_test)

print(classification_report(Y_test, pred_lr))
print(confusion_matrix(Y_test, pred_lr))
sns.heatmap(confusion_matrix(Y_test, pred_lr),annot=True,fmt='.0f')

# %%


# %%
clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_train)
pred_rfc = clf.predict(X_test)
print(classification_report(Y_test, pred_rfc))
print(confusion_matrix(Y_test, pred_rfc))
sns.heatmap(confusion_matrix(Y_test, pred_rfc),annot=True,fmt='.0f')


# %%
clf=svm.SVC()
clf.fit(X_train,Y_train)
pred_clf = clf.predict(X_test)

print(classification_report(Y_test, pred_clf))
print(confusion_matrix(Y_test, pred_clf))
sns.heatmap(confusion_matrix(Y_test, pred_clf),annot=True,fmt='.0f')

# %%
mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(X_train,Y_train)
pred_mlpc = mlpc.predict(X_test)

print(classification_report(Y_test, pred_mlpc))
print(confusion_matrix(Y_test, pred_mlpc))
sns.heatmap(confusion_matrix(Y_test, pred_mlpc),annot=True,fmt='.0f')

# %%
model = Sequential()


model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))

##Binary Classification
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100)


loss_df_drop = pd.DataFrame(model.history.history)
loss_df_drop.plot()
pred_NN = model.predict_classes(X_test)
cm_NN = accuracy_score(Y_test, pred_NN)
print(classification_report(y_true=Y_test,y_pred=pred_NN))
print(classification_report(Y_test, pred_NN))
print(confusion_matrix(Y_test, pred_NN))

sns.heatmap(confusion_matrix(Y_test,pred_NN),annot=True,fmt='.0f')


########################################


# %%
cm_lr = accuracy_score(Y_test, pred_lr)
cm_rfc = accuracy_score(Y_test, pred_rfc)
cm_clf = accuracy_score(Y_test, pred_clf)
cm_mlpc = accuracy_score(Y_test, pred_mlpc)


print("Testing Accuracy for")
print("logistic regression Clasification:", cm_lr)
print("Random Forest Clasification:", cm_rfc)
print("SVM Classifier:", cm_clf)
print("Basic MLP:", cm_mlpc)
print("2 layer MLP adams:", cm_NN)

# %%
