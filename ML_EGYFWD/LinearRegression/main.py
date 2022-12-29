import numpy as np
np.random.seed(42)

def GD(X,y,W,b,learn_rate=0.001):
    y_pred=np.matmul(X,W)+b
    error=y-y_pred
    W_new=W+np.matmul(error,X)*learn_rate
    b_new=b+learn_rate*error.sum()
    return W_new,b_new

def miniGD(X,y, batch_size=30,learn_rate=0.001,num_iter=250):
    n_point=X.shape[0]
    W=np.zeros(X.shape[1])
    print(W.shape)
    b=0
    reg_coef=[np.hstack((W,b))]
    print(reg_coef)
    for _ in range(num_iter):
        batch=np.random.choice(range(n_point),batch_size)
        X_batch=X[batch,:]
        y_batch=y[batch]
        W,b=GD(X_batch,y_batch,W,b,learn_rate)
        reg_coef.append(np.hstack((W,b)))
        # print(reg_coef)
    return reg_coef

if __name__ =="__main__":
    data=np.loadtxt('data.csv',delimiter=',')
    X=data[:,:-1]
    y=data[:,-1]
    reg_coef=miniGD(X,y)
    # plot the results
    import matplotlib.pyplot as plt

    plt.figure()
    X_min = X.min()
    X_max = X.max()
    counter = len(reg_coef)
    for W, b in reg_coef:
        counter -= 1
        color = [1 - 0.92 ** counter for _ in range(3)]
        plt.plot([X_min, X_max], [X_min * W + b, X_max * W + b], color=color)
    plt.scatter(X, y, zorder=3)
    plt.show()