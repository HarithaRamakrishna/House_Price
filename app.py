#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask ,request ,render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('regressor_model.pkl','rb'))
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    float_features = [float(x) for x in request.form.values()]  # list comphress 
    final_features = [np.array(float_features)]
    # more pre proce
    prediction = model.predict(final_features)
    print(prediction)
    
    return render_template('index.html',prediction_text = "  House Price Predicted Cost is  $ : {}".format(prediction))

if __name__ == '__main__':
    app.run()

