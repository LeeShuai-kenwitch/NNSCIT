import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
import os 
import math 
import random
from datetime import datetime 
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
from scipy import stats
import time
from collections import defaultdict
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_regression
import glob
from math import log
import sys
import pickle
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.special import gdtr, gdtrix
from scipy.stats import gamma
from scipy.special import digamma
from collections import defaultdict




#
# This python code estimates mutual information (MI) for continuous variables using a nearest neighbors approach. 
# Github repository by Octavio César Mesner and Cosma Rohilla Shalizi
# source: https://github.com/omesner/knncmi.git
# Paper link: https://arxiv.org/abs/1912.03387
#
def getPairwiseDistArray(data, coords = [], discrete_dist = 1):
    '''
    Input: 
    data: pandas data frame
    coords: list of indices for variables to be used
    discrete_dist: distance to be used for non-numeric differences

    Output:
    p x n x n array with pairwise distances for each variable
    '''
    n, p = data.shape
    if coords == []:
        coords = range(p)
    col_names = list(data)
    distArray = np.empty([p,n,n])
    distArray[:] = np.nan
    for coord in coords:
        if np.issubdtype(data[col_names[coord]].dtype, np.number):
            distArray[coord,:,:] = abs(data[col_names[coord]].values -
                                       data[col_names[coord]].values[:,None])
        else:
            distArray[coord,:,:] = (1 - (data[col_names[coord]].values ==
                                    data[col_names[coord]].values[:,None])) * discrete_dist
    return distArray

def getPointCoordDists(distArray, ind_i, coords = list()):
    '''
    Input: 
    ind_i: current observation row index
    distArray: output from getPariwiseDistArray
    coords: list of variable (column) indices

    output: n x p matrix of all distancs for row ind_i
    '''
    if not coords:
        coords = range(distArray.shape[0])
    obsDists = np.transpose(distArray[coords, :, ind_i])
    return obsDists

def countNeighbors(coord_dists, rho, coords = list()):
    '''
    input: list of coordinate distances (output of coordDistList), 
    coordinates we want (coords), distance (rho)

    output: scalar integer of number of points within ell infinity radius
    '''
    
    if not coords:
        coords = range(coord_dists.shape[1])
    dists = np.max(coord_dists[:,coords], axis = 1)
    count = np.count_nonzero(dists <= rho) - 1
    return count

def getKnnDist(distArray, k):
    '''
    input:
    distArray: numpy 2D array of pairwise, coordinate wise distances,
    output from getPairwiseDistArray
    k: nearest neighbor value
    
    output: (k, distance to knn)
    '''
    dists = np.max(distArray, axis = 1)
    ordered_dists = np.sort(dists)
    # using k, not k-1, here because this includes dist to self
    k_tilde = np.count_nonzero(dists <= ordered_dists[k]) - 1
    return k_tilde, ordered_dists[k]

def cmiPoint(point_i, x, y, z, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y, z: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    cmi point estimate
    '''
    n = distArray.shape[1]
    coord_dists = getPointCoordDists(distArray, point_i, x + y + z)
    k_tilde, rho = getKnnDist(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    z_coords = list(range(len(x+y), len(x+y+z)))
    nxz = countNeighbors(coord_dists, rho, x_coords + z_coords)
    nyz = countNeighbors(coord_dists, rho, y_coords + z_coords)
    nz = countNeighbors(coord_dists, rho, z_coords)
    xi = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)
    return xi

def miPoint(point_i, x, y, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    mi point estimate
    '''
    n = distArray.shape[1]
    coord_dists = getPointCoordDists(distArray, point_i, x + y)
    k_tilde, rho = getKnnDist(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    nx = countNeighbors(coord_dists, rho, x_coords)
    ny = countNeighbors(coord_dists, rho, y_coords)
    xi = digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny)
    return xi
    
def cmi(x, y, z, k, data, discrete_dist = 1, minzero = 1):
    '''
    computes conditional mutual information, I(x,y|z)
    input:
    x: list of indices for x
    y: list of indices for y
    z: list of indices for z
    k: hyper parameter for kNN
    data: pandas dataframe

    output:
    scalar value of I(x,y|z)
    '''
    # compute CMI for I(x,y|z) using k-NN
    n, p = data.shape

    # convert variable to index if not already
    vrbls = [x,y,z]
    for i, lst in enumerate(vrbls):
        if all(type(elem) == str for elem in lst) & len(lst) > 0:
            vrbls[i] = list(data.columns.get_indexer(lst))
    x,y,z = vrbls
            
    distArray = getPairwiseDistArray(data, x + y + z, discrete_dist)
    if len(z) > 0:
        ptEsts = map(lambda obs: cmiPoint(obs, x, y, z, k, distArray), range(n))
    else:
        ptEsts = map(lambda obs: miPoint(obs, x, y, k, distArray), range(n))
    if minzero == 1:
        return(max(sum(ptEsts)/n,0))
    elif minzero == 0:
        return(sum(ptEsts)/n)


# 
# This python code for estimating conditional mutual information (CMI) for continuous variables using Classifier-based method.
# Github repository by Sudipto Mukherjee, Himanshu Asnani and Sreeram Kannan.
# source: https://github.com/sudiptodip15/CCMI.git
# Paper link: https://arxiv.org/abs/1906.01824
# 
class Classifier_MI(object):
    
    def __init__(self, data_train, data_eval, dx, h_dim = 256, actv = tf.nn.relu, batch_size = 32,
                 optimizer='adam', lr=0.001, max_ep = 20, mon_freq = 5000,  metric = 'donsker_varadhan'):

        self.dim_x = dx
        self.data_dim = data_train.shape[1]
        self.X = data_train[:, 0:dx]   
        self.Y = data_train[:, dx:]    
        self.train_size = len(data_train)

        self.X_eval = data_eval[:, 0:dx]
        self.Y_eval = data_eval[:, dx:]
        self.eval_size = len(data_eval)

        # Hyper-parameters of statistical network
        self.h_dim = h_dim
        self.actv = actv

        # Hyper-parameters of training process
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.max_iter = int(max_ep * self.train_size / batch_size)
        self.mon_freq = mon_freq
        self.metric = metric
        self.tol = 1e-4
        self.eps = 1e-8


        self.reg_coeff = 1e-3
    
    def sample_p_finite(self, batch_size):
        index = np.random.randint(low = 0, high = self.train_size, size=batch_size)   
        return np.hstack((self.X[index, :], self.Y[index, :]))  
    
    def shuffle(self, batch_data, dx):
        batch_x = batch_data[:, 0:dx]    
        batch_y = batch_data[:, dx:]     
        batch_y = np.random.permutation(batch_y)   
        return np.hstack((batch_x, batch_y))
    
    def log_mean_exp_numpy(self, fx_q, ax = 0):
        eps = 1e-8
        max_ele = np.max(fx_q, axis=ax, keepdims = True)
        return (max_ele + np.log(eps + np.mean(np.exp(fx_q-max_ele), axis = ax, keepdims=True))).squeeze()

    def classifier(self, inp, reuse = False):
        
        #tf.compat.v1.reset_default_graph()
        with tf.compat.v1.variable_scope('func_approx') as vs:
            if reuse:
                vs.reuse_variables()
            
            # tf.contrib.layers.l2_regularizer
            # actv = tf.nn.relu
            dense1 = tf.compat.v1.layers.dense(inp, units=self.h_dim, activation=self.actv, 
                                               kernel_regularizer=tf.keras.regularizers.l2(self.reg_coeff))
            dense2 = tf.compat.v1.layers.dense(dense1, units=self.h_dim, activation=self.actv,
                                               kernel_regularizer=tf.keras.regularizers.l2(self.reg_coeff))
            logit = tf.compat.v1.layers.dense(dense2, units=1, activation=None, 
                                              kernel_regularizer=tf.keras.regularizers.l2(self.reg_coeff))
            prob = tf.nn.sigmoid(logit)

            return logit, prob

    def train_classifier_MLP(self):

        # Define tensorflow nodes for classifier
        tf.compat.v1.disable_eager_execution()
        Inp = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.data_dim], name='Inp')
        label = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1], name='label')

        logit, y_prob = self.classifier(Inp)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
        l2_loss = tf.compat.v1.losses.get_regularization_loss()   
        cost = tf.reduce_mean(input_tensor=cross_entropy) + l2_loss

        y_hat = tf.round(y_prob)   
        
        correct_pred = tf.equal(y_hat, label)
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))
        
        # optimizer='adam'
        if self.optimizer == 'sgd':
            opt_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost)
        elif self.optimizer == 'adam':
            opt_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.allow_growth = True  
        
        with tf.compat.v1.Session(config = run_config) as sess:

            sess.run(tf.compat.v1.global_variables_initializer()) 

            eval_inp_p = np.hstack((self.X_eval, self.Y_eval))  
            eval_inp_q = self.shuffle(eval_inp_p, self.dim_x)   
            B = len(eval_inp_p)    
            
            for it in range(self.max_iter):     
    
                batch_inp_p = self.sample_p_finite(self.batch_size)     
                batch_inp_q = self.shuffle(batch_inp_p, self.dim_x)     
                 
                batch_inp = np.vstack((batch_inp_p, batch_inp_q))   
                by = np.vstack((np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))))    
                batch_index = np.random.permutation(2*self.batch_size)    
                batch_inp = batch_inp[batch_index]    
                by = by[batch_index]     

                L, _ = sess.run([cost, opt_step], feed_dict={Inp: batch_inp, label: by})

                if ((it + 1) % self.mon_freq == 0):     

                    eval_inp = np.vstack((eval_inp_p, eval_inp_q))
                    eval_y = np.vstack((np.ones((B, 1)), np.zeros((B, 1))))
                    eval_acc = sess.run(accuracy, feed_dict={Inp: eval_inp, label: eval_y})
                    print('Iteraion = {}, Test accuracy = {}'.format(it+1, eval_acc))
            
            pos_label_pred_p = sess.run(y_prob, feed_dict={Inp: eval_inp_p})
            rn_est_p = (pos_label_pred_p+self.eps)/(1-pos_label_pred_p-self.eps)
            finp_p = np.log(np.abs(rn_est_p))

            pos_label_pred_q = sess.run(y_prob, feed_dict={Inp: eval_inp_q})
            rn_est_q = (pos_label_pred_q + self.eps) / (1 - pos_label_pred_q - self.eps)
            finp_q = np.log(np.abs(rn_est_q))

            mi_est = np.mean(finp_p) - self.log_mean_exp_numpy(finp_q)

        return mi_est



class CCMI(object):


    def __init__(self, X, Y, Z, tester, metric, num_boot_iter, h_dim, max_ep):

        self.dim_x = X.shape[1]
        self.dim_y = Y.shape[1]
        self.dim_z = Z.shape[1]
        self.data_xyz = np.hstack((X, Y, Z))    
        self.data_xz = np.hstack((X, Z))      
        self.threshold = 1e-4

        self.tester = tester
        self.metric = metric
        self.num_boot_iter = num_boot_iter
        self.h_dim = h_dim
        self.max_ep = max_ep

    def split_train_test(self, data):
        total_size = data.shape[0]
        train_size = int(2*total_size/3)
        data_train = data[0:train_size,:]    
        data_test = data[train_size:, :]
        return data_train, data_test
    
    
    def gen_bootstrap(self, data):
        np.random.seed()
        random.seed()
        num_samp = data.shape[0]
        I = np.random.permutation(num_samp)    
        data_new = data[I, :]
        return data_new
    
    def get_cmi_est(self):
        if self.tester == 'Neural':
            print('Tester = {}, metric = {}'.format(self.tester, self.metric))
            
        elif self.tester == 'Classifier':
            #print('Tester = {}, metric = {}'.format(self.tester, self.metric))
            I_xyz_list = []
            for t in range(self.num_boot_iter):   
                tf.compat.v1.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xyz)   
                data_xyz_train, data_xyz_eval = self.split_train_test(data_t)    
                
                classMINE_xyz = Classifier_MI(data_xyz_train, data_xyz_eval, self.dim_x,
                                              h_dim = self.h_dim, max_ep = self.max_ep)
                I_xyz_t = classMINE_xyz.train_classifier_MLP()
                I_xyz_list.append(I_xyz_t)

            I_xyz_list = np.array(I_xyz_list)
            I_xyz = np.mean(I_xyz_list)

            I_xz_list = []
            for i in range(self.num_boot_iter):
                tf.compat.v1.reset_default_graph()
                data_t = self.gen_bootstrap(self.data_xz)
                data_xz_train, data_xz_eval = self.split_train_test(data_t)
                classMINE_xz = Classifier_MI(data_xz_train, data_xz_eval, self.dim_x,
                                              h_dim = self.h_dim, max_ep = self.max_ep)
                I_xz_t = classMINE_xz.train_classifier_MLP()
                I_xz_list.append(I_xz_t)

            I_xz_list = np.array(I_xz_list)
            I_xz = np.mean(I_xz_list)
            cmi_est = abs(I_xyz - I_xz)
        else:
            raise NotImplementedError

        return cmi_est



#
# python code for our NNSCIT
#
def GMCIT(x, y, z, normalize=False, verbose=False):
    
    if normalize:
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        z = (z - z.min()) / (z.max() - z.min())

    x_dim = y_dim = 1
    z_dim = z.shape[1]
    
    n = len(z[:, 0])
    x_train, y_train, z_train = x[:int(2*n/3),], y[:int(2*n/3),], z[:int(2*n/3),]
    x_test, y_test, z_test = x[int(2*n/3):,], y[int(2*n/3):,], z[int(2*n/3):,]
    
    
    data_xyz = np.hstack((x_train, y_train, z_train))
    
    def split_XYZ(data, dx, dy):
        X = data[:, 0:dx]
        Y = data[:, dx:dx+dy]
        Z = data[:, dx+dy:]
        return X, Y, Z

    def gen_bootstrap(data):
        np.random.seed()
        random.seed()
        num_samp = data.shape[0]
        I = np.random.permutation(num_samp)
        data_new = data[I, :]
        return data_new

    def split_train_test(data):
        total_size = data.shape[0]
        train_size = int(2*total_size/3)
        data_train = data[0:train_size,:]   
        data_test = data[train_size:, :]
        return data_train, data_test
    
    def mimic_knn(data_mimic, dx, dy, dz, Z_marginal):
        X_train, Y_train, Z_train  = split_XYZ(data_mimic, dx, dy)
        nbrs = NearestNeighbors(n_neighbors=1).fit(Z_train)
        indx = nbrs.kneighbors(Z_marginal, return_distance=False).flatten()
        X_marginal = X_train[indx, :]
        
        return X_marginal
    
    U1 = np.hstack((x_test, y_test, z_test))
    X_1, Y_1, Z_1 = split_XYZ(U1, x_dim, y_dim) 

    n_samples = 500
    rho = []
    for _ in range(n_samples):
        
        data = gen_bootstrap(data_xyz)  
        mimic_size = int(len(data)/2)
        data_mimic = data[0:mimic_size,:]  
        x_hat = mimic_knn(data_mimic, x_dim, y_dim, z_dim, Z_1)  

        if verbose:
            dist = pd.DataFrame(x_hat, columns=['knn generate samples'])
            fig, ax = plt.subplots()
            dist.plot.hist(density=True, ax=ax)
            ax.set_ylabel('')
            plt.show()
        
        # Using a nearest neighbors approach to compute I(X_hat;Y_test)
        da0 = np.concatenate([x_hat, y_test], axis=1)
        da0 = pd.DataFrame(da0)
        cmi_value0 = cmi([0],[1],[], 3, da0)
        rho.append(cmi_value0)
        
    # Using CCMI method to compute I(X_test;Y_test|Z_test)
    tf.compat.v1.reset_default_graph()
    model_indep = CCMI(x_test, y_test, z_test, tester='Classifier', metric='donsker_varadhan', num_boot_iter=5, h_dim=256, max_ep=24)
    cmi_value = model_indep.get_cmi_est()
    
    print(f'I(X;Y|Z) is {cmi_value}')
    
    p_value = (1 + sum(np.array(rho) > cmi_value)) / (1 + n_samples)
    
    return p_value