# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:29:14 2020

@author: RUBY
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
location1 = pickle.load(open('location1.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Location = request.form.get('Location')
    Total_Sqft = float(request.form.get('Total_Sqft'))
    Bath = float(request.form.get('Bath'))
    BHK = float(request.form.get('BHK'))
    Price_per_sqft = float(request.form.get('Price_per_sqft'))
    
    int_location=int(location1.transform([Location]))
    
    final_features = [[int_location, Total_Sqft, Bath, BHK, Price_per_sqft]]

    prediction = model.predict(final_features)
    prediction = prediction*100000
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Estimate Price should be Rs {}'.format(output))




if __name__ == "__main__":
    app.run(debug=True)