"""
Step1:

cur_x = 3 # The algorithm starts at x=3
rate = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x: 2*(x+5) #Gradient of our function 


Step 2:

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x #Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x) #Grad descent
    previous_step_size = abs(cur_x - prev_x) #Change in x
    iters = iters+1 #iteration count
    print("Iteration",iters,"\nX value is",cur_x) #Print iterations
    
print("The local minimum occurs at", cur_x)

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

def get_data(source_file:str = 'input2.csv', names_in:list = ['age','weight','height']):

# Define input and output filepaths
    input_path = os.path.join(os.getcwd(),'datasets','in', source_file)

    # Read input data
    df = pd.read_csv(input_path, names=names_in)
    
    for col in df.columns:
        df[col] = (df[col] - np.mean(df[col]))/np.std(df[col])

    return df


def gradient_descent(df=df, iterations:int = 100, precision:float = 10**(-6), learning_rates:tuple = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)):

    """
    X:dataframe of feature columns: age and weight
    y: dataframe of label column: height
    beta:
    learning_rate: 
    iterations: 
    """
    X = df[df.columns[0:2]].values
    y = df[df.columns[2]].to_numpy()
    y = y[:, np.newaxis]

    m = y.shape[0]
    
    betas = np.zeros((2, iterations)) #initialize betas as zeros, e.g. array([0., 0., 0.])
    betas_new =[]
    i = 0
    for alfa in learning_rates:
        while i < iterations:
            prediction = np.dot(X, betas)
            beta = betas - (1/m) * alfa *(X.T.dot((prediction - y)))
            i += 1
        betas_new.append(np.mean(beta, axis = 1))
    return betas_new
 

def write_csv(filename:str='outputs_2a.csv', betas_new:list = betas_new, iterations:int = iterations, learning_rates:tuple = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)):
        # write the outputs csv file
        filepath = os.path.join(os.getcwd(),'datasets','out', filename)
        iter_array = np.full(len(learning_rates), iterations)
        df_iter = pd.DataFrame(iter_array)
        df_lr = pd.DataFrame(learning_rates)
        df_b = pd.DataFrame(betas_new)
        df_final = pd.concat([df_iter, df_lr, df_b], axis = 1, ignore_index = True)
        dataframe = df_final.rename(columns={0:'learning_rate',1:'iterations',2:'b_age',3:'b_weight'})
        dataframe.to_csv(filepath)
        return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')

    

#Compare the convergence rate when α is small versus large. What is the ideal learning rate to obtain an accurate model? In addition to the nine learning rates above, come up with your own choice of value for the learning rate. Then, using this new learning rate, run the algorithm for your own choice of number of iterations.

#Implement your gradient descent in a file called problem2.py, which will be executed like so:
#$ python3 problem2.py input2.csv output2.csv
#This should generate an output file called output2.csv. There are ten cases in total, nine with the specified learning rates (and 100 iterations), and one with your own choice of learning rate (and your choice of number of iterations). After each of these ten runs, your program must print a new line to the output file, containing a comma-separated list of alpha, number_of_iterations, b_0, b_age, and b_weight in that order (see example), please do not round you your numbers. These represent the regression models that your gradient descents have computed for the given dataset.

#For the output, please follow the exact format, with no extra commas, change in upper/lower case etc. Extra unnecessary commas may make the automated script fail and result in you losing points.

#What To Submit. problem2.py, which should behave as specified above. Before you submit, the RUN button on Vocareum should help you determine whether or not your program executes correctly on the platform.
