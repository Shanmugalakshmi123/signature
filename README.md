Documentation of the Signature Verification project

home.py	This is the streamlit front end file. It has four sections. One section is to input the location of the signature from any of the Bengali or Hindhi or Cedar signatures and predict if it is original or forgery by clicking on the predict button.
The second session has the text box to input the location of the Bengali signature and a predict button to predict if it is original or forgery by clicking on it.
The third session has the text box to input the location of the Hindhi signature and a predict button to predict if it is original or forgery by clicking on it.
The fourth session has the text box to input the location of the Cedar signature and a predict button to predict if it is original or forgery by clicking on it.
bengali.py	This python file created a model bengali1.h5 which is the learning model for Bengali signatures. I pickled the model in home.py.
hindhi.py	This python file created a model hindhi1.h5 which is the learning model for Hindhi signatures. I pickled the model in home.py.
cedar.py	This python file created a model cedar3.h5 which is the learning model for Cedar signatures. I pickled the model in home.py.
overall.py	This python file created an overall model overall3.h5 which is the learning model for all the three signatures. I pickled the model in home.py.
one.ipynb	This is the workfile where I moved all the signature files to different locations like original or forgery etc.
Algorithm design for the convolutional neural network:
I used Sequential model in which I used Conv2D- the convolutional neural network.
I used three relu activation functions and at the end I used one sigmoid function.
