"""
This code implements a perceptron algorithm (PLA). 
This is the version of the file that runs in Python 3.6:
- it does not plot
- it had to be adjusted to pandas 0.23.0 because Vocareum uses python 3.6

First, we visualise the dataset which contains 2 features. We can see that the dataset can be clearly separated by drawing a straight line between them. The goal is to write an algorithm that finds that line and classifies all of these data points correctly.

The output file (e.g. 'output1_f.csv') contains the values of w1, w2 and b which define the 'threshold' line. The last row will be the most accurate one. Each time it goes through each of the examples in 'input1.csv', it adds a new line to the output file containing a comma-separated list of the weights w_1, w_2, and b (bias) in that order. 

Upon convergence, the program stops, and the final values of w_1, w_2, and b are printed to the output file (output1.csv). This defines the decision boundary that your PLA has computed for the given dataset.

Note: When implementing your PLA, in case of tie (sum of w_jx_ij = 0), please follow the lecture note and classify the datapoint as -1.

Ensure this file can be executed as:
$ python3 problem1.py input1.csv output1.csv

The code includes plotting functions. However those are disabled when executing the code from the command line in the format specified immediately above.

"""

# builtin modules
import os
import psutil
import requests
import sys

# 3rd party modules
import pandas as pd
import numpy as np
#import plotly.graph_objects as go

    
def get_data(source_file):

    # Define input and output filepaths
    input_path = os.path.join(os.getcwd(), source_file)

    # Read input data
    df = pd.read_csv(input_path)

    return df

def perceptron_classify(df, n:int = 200):
    """
    1. set b = w = 0
    2. for N iterations, or until weights do not change
        (a) for each training example xᵏ with label yᵏ
            i. if yᵏ — f(xᵏ) = 0, continue
            ii. else, update wᵢ, △wᵢ = (yᵏ — f(xᵏ)) xᵢ
    """
        
    # transform the dataframe to an array
    data = np.asmatrix(df, dtype = 'float64')

    # get the first two columns as pairs of values
    features = data[:, :-1]
    # get the last column
    labels = data[:, -1]

    # assign zero weight as a starting point to features and labels
    w = np.zeros(shape=(1, features.shape[1]+1)) #e.g. array([0., 0., 0.])
    w_ = np.empty(shape=[0,3]) # declare w_ as an empty matrix of same shape as w

    for iteration in range(0,n):
        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1) # add a column of 1s to represent w0
            f = np.dot(w, x.transpose()) # a scalar
            #print(f)
            if f * label <= 0:
                w += (x * label.item(0,0)).tolist() # because label comes from being a matrix (matrix([[1.]])) and needs to be converted to scalar
            else:
                iteration = n

        w_ = np.vstack((w_, w))

    return w_


   
def write_csv(filename, weights):
    # write the outputs csv file
    filepath = os.path.join(os.getcwd(), filename)
    dataframe = pd.DataFrame(data=weights, columns=('b','w1','w2'))
    # reorder the columns in the dataframe in accordance with assignment 
    order = [1,2,0] # setting column's order, 'b' goes as first column followed by weights
    dataframe = dataframe[[dataframe.columns[i] for i in order]]
    dataframe.to_csv(filepath, index = False, header = False)
    return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')


def main():

    #take string of input data csv file
    in_data = str(sys.argv[1])
    
    #take string of input data csv file
    out_data = str(sys.argv[2])

    if in_data and out_data:
        #add functions execute here
        df = get_data(in_data)
        w_ = perceptron_classify(df)
        if w_.size:
            write_csv(out_data, w_)
            print("Plot and output csv files are ready !")
    else:
        print("Enter valid command arguments !")


if __name__ == '__main__':
    main()