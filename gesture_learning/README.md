Learning SDE Using Recurrent Neural Network with Log Signature Features

====================================
Introduction
====================================
Supported structures and features:
    -LP-Logsig-RNN

Chalearn2013 Data:
    http://sunai.uoc.edu/chalearn/

====================================
Requirements
====================================
1. Python 3
2. Keras
3. Tensorflow
4. iisignature

====================================
Structure
====================================

Directory:

    model:
	LP_logsig_rnn.py: construct LP-Logsig-RNN by Keras and model training                     
	cus_layer.py: customised layers

====================================
Model Training
====================================
Usage:
	python gesture_recognition_example.ipynb
	(Parameter setting are in this file. Details are listed below)

Settings:

features
	'number_of_segment': number of segment
	'deg_of_logsig': degree of log-signature
	'learning_rate': learning_rate for Adam optimizer
	'epochs': number of epochs to train the model
	'batch_size': batch size
	'n_hidden': number of neurals in LSTM layer
	'filter_size': filter size of conv1d layer
	'drop_rate1': drop rate of Dropout layer after conv1d layer
	'drop_rate2': drop rate of Dropout layer after LSTM layer


	
