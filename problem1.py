"""
This code implements a perceptron algorithm (PLA). It generates an output file called output1.csv. With each iteration of the PLA (each time it goes through all examples), the algorithm prints a new line to the output file, containing a comma-separated list of the weights w_1, w_2, and b in that order. Upon convergence, the program stops, and the final values of w_1, w_2, and b are printed to the output file (output1.csv). This defines the decision boundary that your PLA has computed for the given dataset.

Note: When implementing your PLA, in case of tie (sum of w_jx_ij = 0), please follow the lecture note and classify the datapoint as -1.
"""

# builtin modules
import os
import psutil
import requests

# 3rd party modules
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Define input and output filepaths
source_file = 'input1.csv'
output_file = 'output1.csv'
#path = os.path.dirname(os.getcwd())
input_path = os.path.join(os.getcwd(),'datasets','in', source_file)
output_path = os.path.join(os.getcwd(),'datasets','out', output_file)

# Read input data
df = pd.read_csv(input_path, names=['feature1','feature2','a_value'])
# Read sample output data
df_out = pd.read_csv(output_path, names=['A','B','C'])

# Create the figure for plotting the initial data
fig = go.Figure(data=go.Scatter(x=df['feature1'], 
                                y=df['feature2'],
                                mode='markers',
                                marker=dict(
                                color=df['a_value'],
                                colorscale='Viridis',
                                line_width=1,
                                size = 16),
                                text=df['a_value'], # hover text goes here
                                showlegend=False))  # turn off legend only for this item

# Create the 1D array for X values from the first feature; this is just to be able to plot a line
# within the space define by the two features explored
X = np.linspace(0, df['feature1'].max(), df['feature1'].size)
# Vector Y will calculated from the weights, a, b, and c and the value of X in its 1D linear space
Y = []

for a, b, c in zip(df_out['A'],df_out['B'],df_out['C']):
    for x in X:
        y = (c - a * x) / b # per the equation of a line, e.g. C = Ax + By
        Y.append(y)

# Add the threshold line to the plot
fig.add_trace(go.Line(
    x=X, y=Y,
    mode='lines',
    name = 'Threshold'
))

# Give the figure a title
fig.update_layout(title='Perceptron Algorithm | Problem 1')

# Show the figure, by default will open a browser window
fig.show()

# export plot to png file to images directory
# create an images directory if not already present
if not os.path.exists("images"):
    os.mkdir("images")
# write the png file with the plot/figure
fig.write_image("images/fig1.png")

format="png", width=600, height=350, scale=2


#class Perceptron():

#    def read_data():

#        return

#    def main():
#        Perceptron.read_data()

#if __name__ == '__main__':
#    main()

