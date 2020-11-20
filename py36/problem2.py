"""
# Multivariate Linear Regression Algorithm

### This is the file finally submitted for Assignment 3 for a multivarite linear regression algorithm. Note a few things:
    - this runs with python 3.6
    - it had to be adjusted to pandas 0.23.0 because Vocareum uses python 3.6
    - a version of this file working fine in python 3.7 and associated most up to date versions of numpy and pandas can be found at: 
    [AI-PerceptronLearningAlgorithm-A3\py37\problem2.py]
    - also for running the file in Vocareum the paths to the input.csv and output.csv files was amended to match, again refer to the version of this file
    for python 3.7 for complete paths that will work with this repository.

# Execution
## $ python3 problem2.py input2.csv output2.csv
### This should generate an output file called output2.csv. The outputs file shall include ten cases in total, ten with the specified learning rates (and 100 iterations), the last learning rate being one of my own choice. After each of these ten runs, the program prints a new line to the output file, containing a comma-separated list of alpha, number_of_iterations, b_0, b_age, and b_weight in that order. These represent the regression models that the gradient descents have computed for the given dataset.
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
#import plotly.graph_objects as go


# Prepare data

def get_data(source_file):

# Define input and output filepaths
    input_path = os.path.join(os.getcwd(), source_file)

    # Read input data
    df = pd.read_csv(input_path)
    
    for col in df.columns:
        df[col] = (df[col] - np.mean(df[col]))/np.std(df[col])
        #note: numpy.std returns the population standard deviation
    return df


def gradient_descent(df, iterations:int = 100, learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.02]):

    """
    X:dataframe of feature columns: age and weight
    y: dataframe of label column: height
    beta:
    learning_rate: 
    iterations: 
    """
    df = pd.concat([pd.Series(1, index=df.index, name='b_0'), df], axis=1)
    
    X = df[df.columns[0:3]].values

    y = df[df.columns[3]].values
    
    m = y.shape[0]
    betas = np.zeros(X.shape[1]) #initialize betas as zeros, e.g. array([0., 0., 0.])
    betas_new =[]

    for alfa in learning_rates:
        i = 0
        while i < iterations:
            hypothesis = np.dot(X, betas)
            betas = betas - (1/m) * alfa *(X.T.dot((hypothesis - y)))
            i += 1
        betas_new.append(betas)

    return betas_new
 
def write_csv(filename, betas_new, iterations:int = 100, learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.02]):
        # write the outputs csv file
        filepath = os.path.join(os.getcwd(), filename)
        iter_array = np.full(len(learning_rates), iterations)
        df_iter = pd.DataFrame(iter_array)
        df_lr = pd.DataFrame(learning_rates)
        df_b = pd.DataFrame(betas_new)
        df_final = pd.concat([df_lr, df_iter, df_b], axis = 1, ignore_index = True)
        dataframe = df_final.rename(columns={0:'alpha',1:'number_of_iterations',2:'b_0', 3:'b_age',4:'b_weight'})
        dataframe.to_csv(filepath, index = False, header = False)
        return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')


def main():
    """
    ## $ python3 problem2.py input2.csv output2.csv
    """
    #take string for input data csv file
    in_data = str(sys.argv[1])
    #take string for output data csv file
    out_data = str(sys.argv[2])

    df = get_data(in_data)
    betas_new = gradient_descent(df)
    write_csv(out_data, betas_new)
    print(out_data)

if __name__ == '__main__':
    main()


    
