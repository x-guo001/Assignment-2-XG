"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""


import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

def test_logreg():

	# Check that your gradient is being calculated correctly
	# What is a reasonable gradient? Is it exploding? Is it vanishing? 
	# Check that your loss function is correct and that 

	# you have reasonable losses at the end of training
	# What is a reasonable loss?

    # load data with default settings
	X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                                                  'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol',
                                                                  'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	# initialize and train the model with hyperparameters
	log_model = logreg.LogisticRegression(num_feats=6, max_iter=10, tol=0.01, learning_rate=0.1, batch_size=12)
	log_model.train_model(X_train, y_train, X_val, y_val)

	# print the loss history
	print(log_model.loss_history_train)
	
	# Check that the mean loss at the beginning is greater than the mean loss at the end
	loss_history_train = log_model.loss_history_train
	# mean of the first 10 losses
	mean_loss_start = np.mean(loss_history_train[:10])
	# mean of the last 10 losses
	mean_loss_end = np.mean(loss_history_train[-10:])
	# assertion
	# raise error message if the mean loss at the end is greater than the one in the beginning
	assert mean_loss_end <= mean_loss_start, "Training loss did NOT decrease as expected."
    
	
	
	

def test_predict():


	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output should look like for a binary classification task?

	# Check accuracy of model after training

	# load data with default settings
	X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                                                  'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol',
                                                                  'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	# initialize and train the model with hyperparameters
	log_model = logreg.LogisticRegression(num_feats=6, max_iter=10, tol=0.01, learning_rate=0.1, batch_size=12)
	log_model.train_model(X_train, y_train, X_val, y_val) 
	# add bias term to X_train to ensure dimention match with y_pred
	X_train_padded = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	# test the prediction
	y_pred = log_model.make_prediction(X_train_padded)
	# assert
	# raise error if not all prediction values are between 0 to 1
	assert np.all((y_pred >= 0) & (y_pred <= 1)), "Prediction values are NOT all between 0 and 1."
