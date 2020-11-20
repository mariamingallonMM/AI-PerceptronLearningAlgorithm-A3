"""
you will use the support vector classifiers in the sklearn package to learn a classification model for a chessboard-like dataset. In your starter code, you will find input3.csv, containing a series of data points. Open the dataset in python. Make a scatter plot of the dataset showing the two classes with two different patterns.

Use SVM with different kernels to build a classifier. Make sure you split your data into training (60%) and testing (40%). Also make sure you use stratified sampling (i.e. same ratio of positive to negative in both the training and testing datasets). Use cross validation (with the number of folds k = 5) instead of a validation set. You do not need to scale/normalize the data for this question. Train-test splitting and cross validation functionalities are all readily available in sklearn.

SVM with Linear Kernel. Observe the performance of the SVM with linear kernel. Search for a good setting of parameters to obtain high classification accuracy. Specifically, try values of C = [0.1, 0.5, 1, 5, 10, 50, 100]. Read about sklearn.grid_search and how this can help you accomplish this task. After locating the optimal parameter value by using the training data, record the corresponding best score (training data accuracy) achieved. Then apply the testing data to the model, and record the actual test score. Both scores will be a number between zero and one.
SVM with Polynomial Kernel. (Similar to above).
Try values of C = [0.1, 1, 3], degree = [4, 5, 6], and gamma = [0.1, 0.5].
SVM with RBF Kernel. (Similar to above).
Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100] and gamma = [0.1, 0.5, 1, 3, 6, 10].
Logistic Regression. (Similar to above).
Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100].
k-Nearest Neighbors. (Similar to above).
Try values of n_neighbors = [1, 2, 3, ..., 50] and leaf_size = [5, 10, 15, ..., 60].
Decision Trees. (Similar to above).
Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].
Random Forest. (Similar to above).
Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].
What To Submit. output3.csv (see example). Please follow the exact format, with no extra commas, change in upper/lower case etc. Extra unnecessary commas may make the automated script fail and result in you losing points. There is no need to submit your actual program. The file should contain an entry for each of the seven methods used. For each method, print a comma-separated list as shown in the example, including the method name, best score, and test score, expressed with as many decimal places as you please. There may be more than one way to implement a certain method, and we will allow for small variations in output you may encounter depending on the specific functions you decide to use.
"""


# builtin modules
import os
import psutil
import requests
import sys
import math


# 3rd party modules
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


#TODO: add comments to each of the sklearn functions imported above,
# as to where they are used and why

def get_data(source_file):

# Define input and output filepaths
    input_path = os.path.join(os.getcwd(),'datasets','in', source_file)

    # Read input data
    df = pd.read_csv(input_path)
       
    return df

def plot_inputs(df, names_in:list = ['A','B','label']):
        """
        Plot the input dataset as a scatter plot, showing the two classes with two different patterns.
        - source_file: csv file with the input samples
        - weights: from perceptron_classify function
        - names_in: a list of the names of the columns (headers) in the input df
        returns:
        - a plot of the figure in the default browser, and
        - a PNG version of the plot to the "images" project directory
        """ 
        # Create the figure for plotting the initial data
        fig = go.Figure(data=go.Scatter(x=df[names_in[0]], 
                                        y=df[names_in[1]],
                                        mode='markers',
                                        marker=dict(
                                        color=df[names_in[2]],
                                        colorscale='Viridis',
                                        line_width=1,
                                        size = 16),
                                        text=df[names_in[2]], # hover text goes here
                                        showlegend=False))  # turn off legend only for this item

        ## Create the 1D array for X values from the first feature; this is just to be able to plot a line
        ## within the space defined by the two features explored
        #X = np.linspace(0, max(df[names_in[0]].max(),df[names_in[1]].max()))
        ## Vector Y will calculated from the weights, w1, w2, the bias, b, and the value of X in its 1D linear space
        #Y = []

        #for b, w1, w2 in [weights]: #(matrix.tolist()[0] for matrix in weights):
        #    for x in X:
        #        if w2 == 0:
        #            y = 0.0
        #        else:
        #            y = (-(b / w2) / (b / w1))* x + (-b / w2) # per the equation of a line, e.g. C = Ax + By
        #        Y.append(y)

        ## Add the threshold line to the plot
        #fig.add_trace(go.Scatter(x=X, y=Y,
        #                            mode= 'lines',
        #                            name = 'Threshold'))


        # Give the figure a title
        fig.update_layout(title='Perceptron Algorithm | Classification with support vector classifiers | Problem 3')

        # Show the figure, by default will open a browser window
        fig.show()

        # export plot to png file to images directory
        # create an images directory if not already present
        if not os.path.exists("images"):
            os.mkdir("images")
        ## write the png file with the plot/figure
        return fig.write_image("images/fig3.png")


def plot_model(y_test, predictions):
        """
        Plot the model dataset as a scatter plot:
        - on the X axis, plot the y_test vector, e.g. the 'True Values'
        - on the Y axis, plot the predictions vector, e.g. the 'Predicted Values'
        returns:
        - a plot of the figure in the default browser, and
        - a PNG version of the plot to the "images" project directory
        """ 

        # Create the figure for plotting the model
        fig = go.Figure(data=go.Scatter(x=y_test, 
                                        y=predictions,
                                        mode='markers'))  

        # Give the figure a title and name the x,y axis as well
        fig.update_layout(
            title='Perceptron Algorithm | Classification with support vector classifiers | Labels vs Predictions',
            xaxis_title='True Values',
            yaxis_title='Predicted Values')

        # Show the figure, by default will open a browser window
        fig.show()

        # export plot to png file to images directory
        # create an images directory if not already present
        if not os.path.exists("images"):
            os.mkdir("images")
        ## write the png file with the plot/figure
        return fig.write_image("images/fig3-y_test-predictions.png")

def train_split(df, test_percentage:float = 0.40):

    # only define test_percentage, 
    # by default train_percentage = (1 - test_percentage)

    # our X matrix will be the first two cols of the dataframe: 'A' and 'B'
    X = df[df.columns[0:2]].values
    # our y vector will be the third col of the dataframe: 'label'
    y = df['label']
    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, stratify = y)
    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)
        
    return X, y, X_train, X_test, y_train, y_test

def svm_linear_kernel(X, y, X_train, X_test, y_train, y_test, k):

    # fit a model on the training set    
    svm = linear_model.LinearRegression()
    model = svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    scores = model.score(X_test, y_test)
    print("Score:", model.score(X_test, y_test))

    # let's plot the model
    plot_model(y_test, predictions)

    # Perform 5-fold cross validation
    # note that cv default value if None is 5-fold and
    # for cv = int or None, if the estimator is a classifier 
    # and y is either binary or multiclass, StratifiedKFold is used
    scores_validated = cross_val_score(svm, X, y, cv = k)
    print ("Cross-validated scores:", scores_validated)

    return scores, scores_validated, predictions


def svm_linear(X, y, X_train, X_test, y_train, y_test, k):
    kernel_type = 'linear'
    # C = [0.1, 0.5, 1, 5, 10, 50, 100]
    svm = make_pipeline(StandardScaler(), SVC(C = 0.1, kernel = kernel_type, gamma='auto'))
    model = svm.fit(X, y)
    predictions = svm.predict(X_test)
    test_score = model.score(X_test, y_test)
    print("SVM", kernel_type, "Test Score:", test_score, sep=' ')

    # let's plot the model
    plot_model(y_test, predictions)

    # Perform 5-fold cross validation
    # note that cv default value if None is 5-fold and
    # for cv = int or None, if the estimator is a classifier 
    # and y is either binary or multiclass, StratifiedKFold is used
    scores_validated = cross_val_score(svm, X, y, cv = k)
    best_score = max(scores_validated)
    print ("SVM", kernel_type, "Best of cross-validated scores:", best_score, sep=' ')

    return test_score, best_score, predictions


def svm_poly(X, y, X_train, X_test, y_train, y_test, k):
    kernel_type = 'poly'
    # C = [0.1, 1, 3], degree = [4, 5, 6], and gamma = [0.1, 0.5]
    svm = make_pipeline(StandardScaler(), SVC(C = 0.1, kernel = kernel_type, degree = 4, gamma=0.1))
    model = svm.fit(X, y)
    predictions = svm.predict(X_test)
    test_score = model.score(X_test, y_test)
    print("SVM", kernel_type, "Test Score:", test_score, sep=' ')

    # let's plot the model
    plot_model(y_test, predictions)

    # Perform 5-fold cross validation
    # note that cv default value if None is 5-fold and
    # for cv = int or None, if the estimator is a classifier 
    # and y is either binary or multiclass, StratifiedKFold is used
    scores_validated = cross_val_score(svm, X, y, cv = k)
    best_score = max(scores_validated)
    print ("SVM", kernel_type, "Best of cross-validated scores:", best_score, sep=' ')

    return test_score, best_score, predictions

def write_csv(filename, a, b, c):
        # write the outputs csv file
        filepath = os.path.join(os.getcwd(),'datasets','out', filename)
        df_a = pd.DataFrame(a)
        df_b = pd.DataFrame(b)
        df_c = pd.DataFrame(c)
        df = pd.concat([df_a, df_b, df_c], axis = 1, ignore_index = True)
        #dataframe = df.rename(columns={0:'alpha',1:'number_of_iterations',2:'b_0', 3:'b_age',4:'b_weight'})
        df.to_csv(filepath, index = False, header = False)
        return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')

def main():
    """
    ## $ python3 problem3.py input3.csv output3.csv
    """
    #take string for input data csv file
    #in_data = str(sys.argv[1])
    in_data = 'input3.csv'
    #take string for output data csv file
    #out_data = str(sys.argv[2])
    out_data = 'output3.csv'

    df = get_data(in_data)
    plot_inputs(df)
    X, y, X_train, X_test, y_train, y_test = train_split(df)
    #scores, scores_validated, predictions = svm_linear_kernel(X, y, X_train, X_test, y_train, y_test, 5)

    test_score, best_score, predictions = svm_linear(X, y, X_train, X_test, y_train, y_test, 5)
    test_score_poly, best_score_poly, predictions_poly = svm_poly(X, y, X_train, X_test, y_train, y_test, 5)
    
    a = ['svm_linear', 'svm_polynomial'] #svm_rbf, logistic, knn, decision_tree, random_forest
    b = [test_score, test_score_poly]
    c = [best_score, best_score_poly]

    write_csv(out_data, a, b, c)

if __name__ == '__main__':
    main()
