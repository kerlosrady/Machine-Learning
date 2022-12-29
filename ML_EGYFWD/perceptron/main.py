import numpy as np

# Setting the random seed, feel free to change it and see different solutions.
import pandas as pd

np.random.seed(42)


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W) + b)[0])


# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate=0.001):
    # Fill in code
    for i in range(len(X)):
        y_hat=prediction(X.iloc[i],W,b)
        if y_hat-y.iloc[i]==-1:
            W[0]=W[0]+X.iloc[i,0]*learn_rate
            W[1] = W[1] + X.iloc[i, 1] * learn_rate
            b=b+learn_rate
        if y.iloc[i]-y_hat==-1:
            W[0]=W[0]-X.iloc[i,0]*learn_rate
            W[1] = W[1] - X.iloc[i, 1] * learn_rate
            b = b - learn_rate
    return W, b


# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate=0.005, num_epochs=200):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
    return boundary_lines

if __name__=='__main__':
    df=pd.read_csv('data.csv')
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    # print(X[1])
    reg_coef=trainPerceptronAlgorithm(X,y)
    import matplotlib.pyplot as plt

    plt.figure()
    x_lin = np.linspace(0, 1, 100)
    # counter = len(reg_coef)
    for i, line in enumerate(reg_coef):
        Θo, Θ1 = line
        print(line)
        if i == len(reg_coef) - 1:
            c, ls, lw = 'k', '-', 2
        else:
            c, ls, lw = 'g', '--', 1.5
        plt.plot(x_lin, Θo * x_lin + Θ1, c=c, ls=ls, lw=lw)
    plt.scatter(X.iloc[:,0], X.iloc[:,1],c=y)
    plt.show()