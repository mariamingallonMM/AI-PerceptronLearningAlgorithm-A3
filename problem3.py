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

# get input data

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
    #write_csv(out_data, betas_new)
    #print(out_data)

if __name__ == '__main__':
    main()
