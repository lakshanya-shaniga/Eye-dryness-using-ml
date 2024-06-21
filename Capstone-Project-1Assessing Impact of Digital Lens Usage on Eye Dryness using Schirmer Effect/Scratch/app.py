# kinda una command a execute cheyandi already sklearn install unte uninstall chesi run cheyandi
# pip install scikit-learn==1.3.0

# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# # Load the Random Forest CLassifier model
filename = 'one.pkl'
model = pickle.load(open('./Models/one.pkl', 'rb'))
model1 = pickle.load(open('./Models/two.pkl', 'rb'))
model2 = pickle.load(open('./Models/three.pkl', 'rb'))
model3 = pickle.load(open('./Models/four.pkl', 'rb'))
# one = pd.read_csv('one.csv')
# t = one.iloc[[1]]
# X = t[t.columns[:-1]]
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods= ['GET','POST'])
def predict():
	li = ['Name','Age','sex','option_1','option_2','option_3','option_4','option_5','option_6','option_7','option_8','option_9','option_10','option_11','option_12','option_13','option_14','option_15','option_16','option_19','option_20','option_21']
	if request.method == 'POST':
		temp = dict()
		selected_items = request.form.getlist('option_17')
		st =  "".join(selected_items)
		temp['option_17'] = st
		selected_items = request.form.getlist('option_18')
		st =  "".join(selected_items)
		temp['option_18'] = st
		for i in li:
			temp[i] = request.form[i]
		data = np.array([[temp['Age'],temp['sex'],temp['option_1'],temp['option_2'],temp['option_3'],temp['option_4'],temp['option_5'],temp['option_6'],temp['option_7'],temp['option_8'],temp['option_9'],temp['option_10'],temp['option_11'],temp['option_12'],temp['option_13'],temp['option_14'],temp['option_15'],temp['option_16'],temp['option_21'],temp['option_17'],temp['option_18'],temp['option_19'],temp['option_20']]])
		# data = pd.DataFrame(data)
		m1 = model.predict(data)
		m2 = model1.predict(data)
		m3 = model2.predict(data)
		m4 = model3.predict(data)
	return render_template('result.html', temp = temp, data = {'Schimers1Lefteye':m1,'Schimers1righteye':m2,'Schimers2Lefteye':m3,'Schimers2righteye':m4},name = temp['Name'])