import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

def main():

    # load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                                                  'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol',
                                                                  'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    
   

    log_model = logreg.LogisticRegression(num_feats=6, max_iter=10, tol=0.01, learning_rate=0.1, batch_size=120)
    log_model.train_model(X_train, y_train, X_val, y_val)
    #log_model.plot_loss_history()

    print(log_model.loss_history_train)

    """
    # for testing purposes once you've added your code
    # CAUTION & HINT: hyperparameters have not been optimized

    
            
    """

if __name__ == "__main__":
    main()