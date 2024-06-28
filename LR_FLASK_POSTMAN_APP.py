from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

#now we should open our model by pickel package
pkl_file=open('Linear_regression_model.pkl','rb')
model=pickle.load(pkl_file)

@app.route('/')
def index():
    return 'Hello World!'



# batch predictions
@app.route('/predict',methods=['POST'])
def predict():
    Data_frame=pd.read_csv(request.files.get('test_file'))# same test_file give in post man also
    #now lets predict
    predictions=model.predict(Data_frame)
    return str(predictions)





if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
