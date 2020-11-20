# AI-PerceptronLearningAlgorithm-A3
The assignment includes three problems as detailed below:
1. Perceptrong Learning Algorithm
2. Multivariate Linear Regression
3. Classification with support vector classifiers

## Problem 1: Perceptron Learning Algorithm

This code implements the perceptron learning algorithm ("PLA") for a linearly separable dataset.

First, we visualise the dataset which contains 2 features. We can see that the dataset can be clearly separated by drawing a straight line between them. The goal is to write an algorithm that finds that line and classifies all of these data points correctly.

The output csv file contains the values of w1, w2 and b which define the 'threshold' line. The last row will be the most accurate one. Each time it goes through each of the examples in 'input1.csv', it adds a new line to the output file containing a comma-separated list of the weights w_1, w_2, and b (bias) in that order. 

Upon convergence, the program stops, and the final values of w_1, w_2, and b are printed to the output file (output1.csv). This defines the decision boundary that your PLA has computed for the given dataset.

### About the Python version

The code in the main folder is written for python 3.7 and it includes plotting functions (plotly). However those are disabled when executing the code in python 3.6 from the command line in the format specified below for Vocareum. Refer to subfolder py36 for a version of the code that runs in python 3.6 and Vocareum.

**Execute as from Vocareum (version python 3.6)***
$ python3 problem1.py input1.csv output1.csv

### Example of outputs for Python 3.7 version 

The following is a sample of the output obtained when running the code in problem1.py as it is in Python 3.7.

Example:
![Initial Dataset and provided Output Threshold]("images/fig1.png")

### Useful references:

- [Perceptron Learning and its implementation in Python](https://towardsdatascience.com/perceptron-and-its-implementation-in-python-f87d6c7aa428) by [Pallavi Bharadwaj](https://medium.com/@pallavibharadwaj)
- [Calculate the Decision Boundary of a Single Perceptron - Visualizing Linear Separability](https://medium.com/@thomascountz/calculate-the-decision-boundary-of-a-single-perceptron-visualizing-linear-separability-c4d77099ef38) by [Thomas Countz](https://medium.com/@thomascountz)
- [19-line Line-by-line Python Perceptron](https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3) by [Thomas Countz](https://medium.com/@thomascountz)

## Problem 2: Multivariate Linear Regression Algorithm

In this problem, we work on linear regression with multiple features using gradient descent. 

Running the code will generate an output file called output2.csv containing a comma-separated list of:
- alpha (slope)
- number_of_iterations (n = 100 as required in the assignment)
- betas values for: b_0, b_age, and b_weight (the two last betas are the features in X)

A total of ten entries are produced, one per learning rate applied. Nine learning rates were provided in the assignment. The last learning rate is my own choice. Learning rates are as follows: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.02]

The outputs csv file represent the regression models that the gradient descents have computed for the given dataset.

### About the Python version

As above for problem 1.

**Execute as from Vocareum (version python 3.6)***
$ python3 problem2.py input2.csv output2.csv

### Useful references:

- [Multivariate Linear Regression from Scratch in Python](https://medium.com/@lope.ai/multivariate-linear-regression-from-scratch-in-python-5c4f219be6a) by [Lope.AI](https://medium.com/@lope.ai)
- [Multivariate Linear Regression in Python Step by Step](https://towardsdatascience.com/multivariate-linear-regression-in-python-step-by-step-128c2b127171) by [Rashida Nasrin Sucky](https://medium.com/@sucky00)
- [Gradient Descent in Python](https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f) by [Sagar Mainkar](https://medium.com/@sagarmainkar)
- [Implement Gradient Descent in Python](https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1) by [Rohan Joseph](https://medium.com/@rohanjoseph_91119)
- [Youtube video explaining difference between single feature and multivariate](https://www.youtube.com/watch?v=7r0fsvgTtHA) by [intrigano](https://www.youtube.com/channel/UCDf3VLjM5A4MntJvCfls4sQ)

## Problem 3: Classification with support vector classifiers


