import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
import warnings
import pickle
import streamlit as st
warnings.filterwarnings("ignore")

data = pd.read_csv("survey.csv")
# print(data.head(5))
male_str = ["m", "male-ish","male","maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female_str = ["cis female", "f","female" "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in data.iterrows():

    if str.lower(col.Gender) in male_str:
        data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        data['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

#Get rid of useless
stk_list = ['A little about you', 'p']
data = data[~data['Gender'].isin(stk_list)]
data['Gender']=data['Gender'].map({'male':0,'female':1, 'trans':2})
data['family_history']=data['family_history'].map({'No':0,'Yes':1})
data['treatment']=data['treatment'].map({'No':0,'Yes':1})
data['obs_consequence']=data['obs_consequence'].map({'No':0,'Yes':1})
data.dropna(axis=0,inplace=True)
print(data['Gender'].value_counts())
# print(data.head(5))
X=data[['Age','Gender','family_history']]
# print(X.head(5))
# data = np.array(data)
Y=data['obs_consequence']
# X = data[1:,1:-1]
# y = data[1:, -1]
# print(y)
# y = y.astype('int')
# X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)
# clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf2.fit(X_train, y_train)

pickle.dump(clf2,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

# print(X_test.head(5))


st.title('Mental Health Prediction')
a=st.text_input("Enter your age",0)
b=st.text_input("Enter your Gender(Enter 0 for Male and 1 for Female)",0)
c=st.text_input("Does you have a family history of mental health(Enter 0 for No and 1 for Yes)",0)
st.write("Here is Your result ")

result  = model.predict([[int(a),int(b),int(c)]])[0]

if result:
    st.success('This is a success message!', icon="✅")
else:
    st.error('This is an error', icon="🚨")


