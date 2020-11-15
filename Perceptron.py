"""
This code implements a perceptron algorithm (PLA). 
First, we visualise the dataset which contains 2 features. We can see that the dataset can be clearly separated by drawing a straight line between them. The goal is to write an algorithm that finds that line and classifies all of these data points correctly.

The output file (e.g. 'output1_f.csv') contains the values of w1, w2 and b which define the 'threshold' line. The last row will be the most accurate one. Each time it goes through each of the examples in 'input1.csv', it adds a new line to the output file containing a comma-separated list of the weights w_1, w_2, and b (bias) in that order. 

Upon convergence, the program stops, and the final values of w_1, w_2, and b are printed to the output file (output1.csv). This defines the decision boundary that your PLA has computed for the given dataset.

Note: When implementing your PLA, in case of tie (sum of w_jx_ij = 0), please follow the lecture note and classify the datapoint as -1.

"""

# builtin modules
import os
import psutil
import requests

# 3rd party modules
import pandas as pd
import numpy as np
import plotly.graph_objects as go


class Perceptron:

    def get_data(source_file:str = 'input1.csv', output_file:str = 'output1.csv', names_in:list = ['feature1','feature2','labels'], names_out:list =['A','B','C']):

        # Define input and output filepaths
        input_path = os.path.join(os.getcwd(),'datasets','in', source_file)
        output_path = os.path.join(os.getcwd(),'datasets','out', output_file)

        # Read input data
        df = pd.read_csv(input_path, names=names_in)
        # Read sample output data
        df_out = pd.read_csv(output_path, names=names_out)

        return df, df_out


    def perceptron_classify(df = df, df_out = df_out, n:int = 200, names_in:list = ['feature1','feature2','labels']):
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
                if f * label <= 0:
                    w += (x * label.item(0,0)).tolist() # because label comes from being a matrix (matrix([[1.]])) and needs to be converted to scalar
                else:
                    iteration = n

            w_ = np.vstack((w_, w))
            
        return w_, w_[-1]


    def plot_results(df = df, weights = w, names_in:list = ['feature1','feature2','labels']):
        """
        Plot the Perceptron classifier, from the following inputs:
        - source_file: csv file with the input samples
        - output_file: csv file with a sample output threshold line
        - names_in: a list of the names of the columns (headers) in the input df
        - names_out: a list of the names of the columns (headers) in the output df
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

        # Create the 1D array for X values from the first feature; this is just to be able to plot a line
        # within the space defined by the two features explored
        X = np.linspace(0, max(df[names_in[0]].max(),df[names_in[1]].max()))
        # Vector Y will calculated from the weights, w1, w2, the bias, b, and the value of X in its 1D linear space
        Y = []

        for b, w1, w2 in [weights]: #(matrix.tolist()[0] for matrix in weights):
            for x in X:
                if w2 == 0:
                    y = 0.0
                else:
                    y = (-(b / w2) / (b / w1))* x + (-b / w2) # per the equation of a line, e.g. C = Ax + By
                Y.append(y)

        # Add the threshold line to the plot
        fig.add_trace(go.Scatter(x=X, y=Y,
                                 mode= 'lines',
                                 name = 'Threshold'))


        # Give the figure a title
        fig.update_layout(title='Perceptron Algorithm | Problem 1')

        # Show the figure, by default will open a browser window
        fig.show()

        # export plot to png file to images directory
        # create an images directory if not already present
        if not os.path.exists("images"):
            os.mkdir("images")
        ## write the png file with the plot/figure
        return fig.write_image("images/fig1.png")
    

    def write_csv(filename:str='submission_1.csv', weights = w_):
        # write the outputs csv file
        filepath = os.path.join(os.getcwd(),'datasets','out', filename)
        dataframe = pd.DataFrame(data=weights, columns=('b','w1','w2'))
        # reorder the columns in the dataframe in accordance with assignment 
        order = [1,2,0] # setting column's order, 'b' goes as first column followed by weights
        dataframe = dataframe[[dataframe.columns[i] for i in order]]
        dataframe.to_csv(filepath)
        return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')


    def main():
        df, df_out = get_data()
        w_, w = perceptron_classify()
        plot_results()
        write_csv('submission_1.csv', w_)
        return print('plot and output csv files are ready')

    if __name__ == '__main__':
        main()
        pass
