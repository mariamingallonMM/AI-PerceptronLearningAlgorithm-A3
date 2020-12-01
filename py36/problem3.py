"""
This code implements a support vector classifier using the sklearn package to learn a classification model for a chessboard-like dataset. 
Written using Python 3.7
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
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor


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


def plot_model(X, y, xx, y_, Z, model_type:str):
        """
        Plot the decision boundary from:
        - X: the features dataset,
        - y: the labels vector, 
        - h: step size in the mesh, e.g. 0.02
        - grid_search: model of the grid_search already fitted
        - model_type: str of the type of model used for title of plot and filename of image to export
        returns:
        - a plot of the figure in the default browser, and
        - a PNG version of the plot to the "images" project directory
        """ 

        # Create the figure for plotting the model
        fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], 
                            mode='markers',
                            showlegend=False,
                            marker=dict(size=10,
                                        color=y, 
                                        colorscale='Jet',
                                        line=dict(color='black', width=1))
                            ))
        
        # Add the heatmap to the plot
        fig.add_trace(go.Heatmap(x=xx[0], y=y_, z=Z,
                          colorscale='Jet',
                          showscale=False))
        
        # Give the figure a title and name the x,y axis as well
        fig.update_layout(
            title='Perceptron Algorithm | Classification with support vector classifiers | ' + model_type.upper(),
            xaxis_title='True Values',
            yaxis_title='Predicted Values')

        # Show the figure, by default will open a browser window
        fig.show()

        # export plot to png file to images directory
        # create an images directory if not already present
        if not os.path.exists("images"):
            os.mkdir("images")
        ## write the png file with the plot/figure
        return fig.write_image("images/fig3-" + model_type + ".png")

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


def apply_CSVC(X, y, X_train, X_test, y_train, y_test, model_type:str, k: int, kernel_type:str, parameters:dict):

    if model_type == 'logistic':
        logistic = linear_model.LogisticRegression()
        start = parameters['C'][0]
        stop = parameters['C'][-1]
        num = len(parameters['C'])
        C = np.logspace(start, stop, num)
        penalty = ['l2']
        hyperparameters = dict(C=C, penalty=penalty)
        grid_search = GridSearchCV(logistic, hyperparameters, cv = k, verbose = 0)

    if model_type == 'knn':
        grid_params = parameters
        grid_search = GridSearchCV(KNeighborsClassifier(), grid_params, cv = k, verbose = 0)

    if model_type == 'decision_tree':
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), parameters, verbose=1, cv=3)

    if model_type == 'random_forest':
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), parameters, verbose=1, cv=3)

    if model_type == 'none':
        svc = svm.SVC()
        # specify cv as integer for number of folds in (stratified)KFold, 
        # cv set to perform 5-fold cross validation, although 'None' already uses the default 5-fold cross validation
        grid_search = GridSearchCV(svc, parameters, cv = k) 
    
    grid_search.fit(X, y) # fit the model #TODO: clarify if fit shall be done on train datasets or on complete set
    #get results best and test
    best_score = grid_search.best_score_
    predictions = grid_search.predict(X_test)
    test_score = grid_search.score(X_test, y_test)
    
    #print results
    print("Best parameters for", kernel_type.upper(), "are:", clf.best_params_, sep=' ')
    print("Best score for", kernel_type.upper(), "is:", clf.best_score_, sep=' ')
    print("Test score for", kernel_type.upper(), "is:", test_score, sep=' ')

    # let's plot the model
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h)
                            , np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    Z = grid_search.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #print(Z)
    #Z = Z.reshape((xx.shape[0], xx.shape[1], 3))

    plot_model(X, y, xx, y_, Z, model_type)
    
    return best_score, test_score

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
   
    best_score_linear, test_score_linear = apply_CSVC(X, y, X_train, X_test, y_train, y_test, model_type = 'none', k = 5, kernel_type = 'svm_linear', parameters = {'kernel':('linear', 'linear'), 'C':[0.1, 0.5, 1, 5, 10, 50, 100]})
    best_score_poly, test_score_poly = apply_CSVC(X, y, X_train, X_test, y_train, y_test, model_type = 'none', k = 5, kernel_type = 'svm_polynomial', parameters = {'kernel':('poly', 'poly'), 'gamma':[0.1, 0.5], 'C':[0.1, 1, 3], 'degree':[4, 5, 6]})
    best_score_rbf, test_score_rbf = apply_CSVC(X, y, X_train, X_test, y_train, y_test, model_type = 'none', k = 5, kernel_type = 'svm_rbf', parameters = {'kernel':('rbf', 'rbf'), 'gamma':[0.1, 0.5, 1, 3, 6, 10], 'C':[0.1, 0.5, 1, 5, 10, 50, 100]})
    best_score_log, test_score_log = apply_CSVC(X, y, X_train, X_test, y_train, y_test, model_type = 'logistic', k = 5, kernel_type = 'logistic', parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]})
    best_score_knn, test_score_knn = apply_CSVC(X, y, X_train, X_test, y_train, y_test, model_type = 'knn', k = 5, kernel_type = 'knn', parameters = {'n_neighbors': np.arange(1,51,1),'leaf_size': np.arange(5,61,5)})
    best_score_dt, test_score_dt = apply_CSVC(X, y, X_train, X_test, y_train, y_test, model_type = 'decision_tree', k = 5, kernel_type = 'decision_tree', parameters = {'max_depth': np.arange(1,51,1),'min_samples_split': np.arange(2,11,1)})
    best_score_rf, test_score_rf = apply_CSVC(X, y, X_train, X_test, y_train, y_test, model_type = 'random_forest', k = 5, kernel_type = 'random_forest', parameters = {'max_depth': np.arange(1,51,1),'min_samples_split': np.arange(2,11,1)})


    a = ['svm_linear', 'svm_polynomial', 'svm_rbf', 'logistic', 'knn', 'decision_tree', 'random_forest']
    b = [test_score_linear, test_score_poly, test_score_rbf,  test_score_log, test_score_knn, test_score_dt, test_score_rf]
    c = [best_score_linear, best_score_poly, best_score_rbf, best_score_log, best_score_knn, best_score_dt, best_score_rf]


    write_csv(out_data, a, b, c)

if __name__ == '__main__':
    main()
