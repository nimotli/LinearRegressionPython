from flask import Flask,render_template,request,redirect,url_for
from flask_bootstrap import Bootstrap
import numpy as np
import math
import os
app = Flask(__name__)

bootstrap = Bootstrap(app)
APP_ROOT=os.path.dirname(os.path.abspath(__file__))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
    target=os.path.join(APP_ROOT,'datasets/')
    if not os.path.isdir(target):
        os.mkdir(target) 
    for dataset in request.files.getlist('dataset'):
        destination =  "/".join([target,'dataset.csv'])
        dataset.save(destination)
    return redirect(url_for('/train'))

@app.route('/train')
def train():
    return render_template('train.html')
@app.route('/predict')
def predict():
    target =  "/".join([os.path.join(APP_ROOT,'models/'),'model.txt'])
    modelData=open(target,'r').read()
    feat=len(modelData.split(','))-1
    return render_template('predict.html',theta=modelData,feat=feat)


def normalize(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    finalX = np.divide((X - mu),sigma)
    return finalX

def calculateCost(X,y,theta):
    m=len(y)*2
    err=np.matmul(X,theta)-y.reshape(len(y),1)
    return np.matmul(err.T,err)/m


def optimize(X,y,alpha,rep):
    m=len(y)
    theta=np.zeros((3,1))
    cost=np.zeros(rep)
    for i in range(0,rep):
        err=np.matmul(X,theta)-y.reshape(len(y),1)
        finalErr=np.matmul(err.T,X)
        scalar=1/m*alpha
        theta=theta-( scalar * finalErr).T
        cost[i] = calculateCost(X,y,theta)
    return [theta,cost]
    #theta = theta - (1/m*alpha .* ((X*theta)-y)'*X)';

@app.route('/training')
def training():
    target=os.path.join(APP_ROOT,'datasets/')
    target =  "/".join([target,'dataset.csv'])
    exists = os.path.isfile(target)
    if exists:
        data=np.genfromtxt(target,delimiter=',')
        rg = int(request.args.get('theRange'))
        dataLn=len(data)
        testLn =  math.floor(dataLn*rg/100)
        trainingLn=dataLn-testLn

        testSet=data[trainingLn:dataLn ,:]
        trainingSet=data[0:trainingLn,:]

        training_features=trainingSet[:,:-1]
        training_labels=trainingSet[:,-1]

        test_features=testSet[:,:-1]
        test_labels=testSet[:,-1]

        training_features = normalize(training_features)
        training_features = np.hstack((np.ones( (len(training_features[:,0]),1) ) , training_features))
        theta = optimize(training_features,training_labels,.1,1000)
        returnString=str(theta[0][0][0])+","+str(theta[0][1][0])+","+str(theta[0][2][0])
        target =  "/".join([os.path.join(APP_ROOT,'models/'),'model.txt'])
        file= open(target,"w")
        file.write(returnString)
        return render_template('trained.html')

    else:
        return 'there is no dataset file'

if __name__=='__main__':
    app.run(debug=True)