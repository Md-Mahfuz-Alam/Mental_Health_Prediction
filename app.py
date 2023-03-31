import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
import warnings
import pickle as pkl
import streamlit as st
warnings.filterwarnings("ignore")

# data = pd.read_csv("survey.csv")

# male_str = ["m", "male-ish","male","maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
# trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
# female_str = ["cis female", "f","female" "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

# for (row, col) in data.iterrows():

#     if str.lower(col.Gender) in male_str:
#         data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

#     if str.lower(col.Gender) in female_str:
#         data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

#     if str.lower(col.Gender) in trans_str:
#         data['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

# #Get rid of useless
# stk_list = ['A little about you', 'p']
# data = data[~data['Gender'].isin(stk_list)]
# data['Gender']=data['Gender'].map({'male':0,'female':1, 'trans':2})
# data['family_history']=data['family_history'].map({'No':0,'Yes':1})
# data['treatment']=data['treatment'].map({'No':0,'Yes':1})
# data['obs_consequence']=data['obs_consequence'].map({'No':0,'Yes':1})
# data.dropna(axis=0,inplace=True)
# print(data['Gender'].value_counts())

# X=data[['Age','Gender','family_history']]

# Y=data['obs_consequence']

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

# clf2 = RandomForestClassifier(random_state=1)
# clf2.fit(X_train, y_train)

# pickle.dump(clf2,open('model.pkl','wb'))
# model=pickle.load(open('model.pkl','rb'))

gender1={'Agender': 0, 'All': 1, 'Androgyne': 2, 'Cis Female': 3, 'Cis Male': 4, 'Cis Man': 5, 'Enby': 6, 'F': 7, 'Femake': 8, 'Female': 9, 'Female ': 10, 'Female (cis)': 11, 'Female (trans)': 12, 'Genderqueer': 13, 'Guy (-ish) ^_^': 14, 'M': 15, 'Mail': 16, 'Make': 17, 'Mal': 18, 'Male': 19, 'Male ': 20, 'Male (CIS)': 21, 'Male-ish': 22, 'Malr': 23, 'Man': 24, 'Nah': 25, 'Neuter': 26, 'Trans woman': 27, 'Trans-female': 28, 'Woman': 29, 'cis male': 30, 'cis-female/femme': 31, 'f': 32, 'femail': 33, 'female': 34, 'fluid': 35, 'm': 36, 'maile': 37, 'male': 38, 'male leaning androgynous': 39, 'msle': 40, 'non-binary': 41, 'ostensibly male, unsure what that really means': 42, 'queer': 43, 'queer/she/they': 44, 'something kinda male?': 45, 'woman': 46}

family_history={'No': 0, 'Yes': 1}

benefits={"Don't know": 0, 'No': 1, 'Yes': 2}

care_options={'No': 0, 'Not sure': 1, 'Yes': 2}

anonymity={"Don't know": 0, 'No': 1, 'Yes': 2}

leave={"Don't know": 0, 'Somewhat difficult': 1, 'Somewhat easy': 2, 'Very difficult': 3, 'Very easy': 4}

work_interfere={"Don't know": 0, 'Never': 1, 'Often': 2, 'Rarely': 3, 'Sometimes': 4}


st.title('Mental Health Prediction')
age=int(st.text_input("Enter your age",0))-18
gender=gender1[st.text_input("Enter your Gender(Enter 0 for Male and 1 for Female)","male")]
family=family_history[st.radio("Does you have a family history of mental health",("No","yes"))]
benefit=benefits[st.radio("Does you have benefits",("Don't know","No","Yes"))]
care=care_options[st.radio("Does you get care options",("No","Not sure","Yes"))]
anonym=anonymity[st.radio("Any anonymity",("Don't know","No","Yes"))]
leaves=leave[st.radio("getting leaves",("Don't know","Somewhat difficult","Somewhat easy","Very difficult","Very easy"))]
work=work_interfere[st.radio("work interference",("Don't know","Never","Often","Rarely","Sometimes"))]
st.write("Here is Your result ")


model=pkl.load(open('model.sav','rb'))

result=model.predict([[age,gender,family,benefit,care,anonym,leaves,work]])
print(result)
if result[0]:
     st.error('You are suffered from Mental illness!', icon="🚨")
   
else:
    st.success('You are OK', icon="✅")


