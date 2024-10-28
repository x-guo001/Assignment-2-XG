
# Assignment 2: Logistic Regression 
## Assignment Goal

The goal of this project is to write a class that performs logistic regression and apply it to the NSCLC dataset to predict whether a patient has small cell or non-small cell lung cancer **(NSCLC)** using demographic, medication, lab values, and procedures performed prior to their diagnosis features.

## Assignment Overview 

In class, we learned how to derive an Ordinary Least Squares (OLS) estimator in a linear regression model, used to identify the best fitting line **(see workshop 2)**.

For this assignment, we will be implementing a logistic regression model using the same framework. 

Logistic regression is useful for classification because the function outputs a value between 0 and 1, corresponding to a categorical classification

For this project, you will be given a set of simulated medical record data ([reference](https://doi.org/10.1093/jamia/ocx079)) from patients with small cell or non-small cell lung cancer. Write a logistic regression to predict whether a person belongs in one class or another. 

## Starting the Assignment

Steps: 

    1. Fork this Repo and clone the fork

    2. Create a conda environment from the requirements.txt file 
        - This will ensure you have the minimum requirements to run this repo
        - If you are unsure of how to do this, please consult this conda cheatsheet 
([Conda Cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf))

        - Activate the Conda Environment
        - Make sure you correctly select the interpreter path for VSCode
        
    3. Read the rest Assignment Context below
        
    4. Complete TODOs in logreg.py 
        - calculate_gradient
        - loss_function
        - make_prediction
    
    5. Install Pytest & Make Local package
        - Use flit: Same process as Assignment 1
        - Note: Required for unit testing

    6. Unit Testing
        - Check if fit appropriately trains model & weights get updated
        - Check loss approaches 0 
        - Check predict works as intended
    
    7. Push to GitHub
        - If you need a refresher, please consult this Git cheatsheet 
        
([Git Cheatsheet](https://education.github.com/git-cheat-sheet-education.pdf))

# Assignment Context

## Logistic Regression 

To allow for binary classification using logistic regression, we use a sigmoid function to model
the data. We will define a loss function to keep track of how the model performs. Instead of using Rsquared, like you've done before in workshops 2 & 3, we will be implementing a log loss
(binary cross-entropy) function. This function will minimize the error when the predicted y is 
close to an expected value of 1 or 0.

Resources to help you get started: 
* Sigmoid Function ([click here](https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e))
* Understanding Binary Cross-Entropy and log loss ([click here](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a))
* Binary Cross-Entropy mathematical implementation ([click here](https://medium.com/@vergotten/binary-cross-entropy-mathematical-insights-and-python-implementation-31e5a4df78f3))



## Dataset 
Class labels are encoded in the NSCLC column of the dataset. This is what you are trying to predict.

* 1 = NSCLC
* 0 = Small cell

A set of features has been pre-selected for you to use in **main.py**, but feel free 
to use other features. 
* Gender
* Penicillin V Potassium 250 MG
* Penicillin V Potassium 500 M
* Computed tomography of chest and abdomen
* Plain chest X-ray (procedure)', 'Diastolic Blood Pressure
* Body Mass Index
* Body Weight
* Body Height
* Systolic Blood Pressure
* LDL Cholesterol
* HDL Cholesterol
* Triglycerides
* etc...
  
A full list of features is provided in utils.py. Note: You do not need to modify anything
in utils.py.


# Grading (20 points total)
## Code (9 points)
* Correct implementation of calculate_gradient function (3)
* Correct implementation of loss_function (3)
* Correct implementation of make_prediction (3)

## Tests (8 points)
* Test cases for gradient (2)
* Test cases for loss (2)
* Test cases for outputs (2)
* Test cases for accuracy (2)

## Documentation & GitHub (3 points)
    9. Write comments documenting your code (2)
    10. Push your finished assignment to GitHub (1)

