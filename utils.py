import numpy as np
import tensorflow as tf
import pandas as pd
import random
np.random.seed(42)


#
# The generate_samples_random function and rdc function were inspired by
# GCIT Github repository by Alexis Bellot and Mihaela van der Schaar
# source: https://github.com/alexisbellot/GCIT
#

def same(x):
    return x

def cube(x):
    return np.power(x, 3)

def negexp(x):
    return np.exp(-np.abs(x))


def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=0.7, fixed_function='linear',
                            debug=False, normalize = True, seed = None, dist_z = 'gaussian'):
    '''Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        f1, f2 to be within {x,x^2,x^3,tanh x, e^{-|x|}, cos x}
    Output:
        Samples X, Y, Z
    '''
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if fixed_function == 'linear':
        f1 = same
        f2 = same
    else:
        I1 = random.randint(2, 5)
        I2 = random.randint(2, 5)

        if I1 == 2:
            f1 = np.square
        elif I1 == 3:
            f1 = cube
        elif I1 == 4:
            f1 = np.tanh
        elif I1 == 5:
            f1 = np.cos

        if I2 == 2:
            f2 = np.square
        elif I2 == 3:
            f2 = cube
        elif I2 == 4:
            f2 = np.tanh
        elif I2 == 5:
            f2 = np.cos
    if debug:
        print(f1, f2)

    num = size

    if dist_z =='gaussian':
        cov = np.eye(dz) 
        mu = 0.7 * np.ones(dz)    # linear case
        #mu = np.zeros(dz)     # nonlinear case
        Z = np.random.multivariate_normal(mu, cov, num)
        Z = np.matrix(Z)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z,(num,dz))
        Z = np.matrix(Z)

    elif dist_z == 'uniform':
        Z = np.random.uniform(-2.5, 2.5, (num,dz))
        Z = np.matrix(Z)
        
    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1) 
    Ax = np.matrix(Ax)

    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1) 
    Ay = np.matrix(Ay)

    Axy = np.random.rand(dx, dy)
    for i in range(dy):
        Axy[:, i] = Axy[:, i] / np.linalg.norm(Axy[:, i], ord=1) 
    Axy = np.matrix(Axy)
    
    Azy = np.random.normal(0., 1., (dz, dy))   
    alpha = 2


    if sType == 'CI':
        X = f1(Z * Ax + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(Z * Ay + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    elif sType == 'I':
        X = f1(nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = f1(Z * Ax + np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))  # X and Z are not independent case
        #X = np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)   # X and Z are independent case
        
        Y = f2(alpha * X * Axy + Z * Azy) + nstd * np.random.multivariate_normal(np.zeros(dy),np.eye(dy),num)

    if normalize == True:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    return np.array(X), np.array(Y), np.array(Z)



# Data Generation Mechanisms for the Two Examples
def two_example_data(size=1000, Type='Example1', dz=5, normalize = True):
    num = size

    if Type == 'Example1':
        X = np.random.normal(1., 1., (num, 1))
        a = np.random.uniform(0., 0.3, (1, dz))
        b = np.random.uniform(0., 0.3, (dz, 1))
        Z = np.matmul(X,a) + np.random.normal(0., 1., (num, dz))
        Y = np.matmul(Z,b) + np.random.normal(0., 1., (num, 1))
    else:
        X = np.random.normal(1., 1., (num, 1))
        Y = np.random.normal(1., 1., (num, 1))
        a = np.random.uniform(0.5, 1, (1, dz))
        b = np.random.uniform(0.5, 1, (1, dz))
        Z = X * a + Y * b + np.random.uniform(-1, 1, (num, dz))

    if normalize == True:
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())
        Z = (Z - Z.min()) / (Z.max() - Z.min())

    return np.array(X), np.array(Y), np.array(Z)




#
# For the GCIT method we use the code rewritten by Tianlin Xu et al.
# source: https://github.com/tianlinxu312/dgcit.git
# Paper link: https://arxiv.org/pdf/1907.04068.pdf
#



#
# DGCIT method
# source: https://github.com/tianlinxu312/dgcit.git
# Paper link: https://arxiv.org/pdf/2006.02615.pdf
#



#
# CCIT method
# source: https://github.com/rajatsen91/CCIT.git
# Paper link: https://arxiv.org/abs/1709.06138
#