#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:18:38 2019

@author: apple
"""

# the goal is to try the l2 loss with scipy.optimize and see if the behavior
# would be different from the conjugate gradient method. No longer computing 
# (approximating) the mle of eta here (though could try using SGD or other methods)
# and instead is essentially (?) performing feature engineering with the \bar{phi}_d 's
# so could possibly use other methods (random forests, SVM, etc) on these bar_phi_d 's.


import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import timeit
from scipy import optimize
import sys

# define functions
# use labels to decide the topic identifibiality issue and output a dictionary
def topic_mapping(y, phi, K):
    """
    :param y: the true label(s), should be an integer np.array of shape (n,)
    :param phi: should be an np.array of shape (n, K), e.g. phi_sample[:,:,MCMC_iters-1]
    :returns: a dictionary with (key, value) = (topic, index), where topic is
    a nonnegative integer, and index is the corresponding entry for the topic
    in a row vector of phi (phi is indexed by document, this phi is the theta in our report)
    """
    # first get the column sum of observations for each topic
    # in order to use scipy.optimize.linear_sum_assignment, need to first
    # transform the cost matrix, so that the cost matrix is nonnegative
    # and the goal to find the smallest "cost" permutation
    #col_sum_dict = {}
    #cost_dict = {}
    cost_matrix = np.zeros((K,K))
    for topic in range(K):
        #col_sum_dict[topic] = np.sum(phi[y==topic,:], axis = 0)
        #cost_dict[topic] = sum(y == topic) - col_sum_dict[topic]
        cost_matrix[topic,:] = sum(y == topic) - np.sum(phi[y==topic,:], axis = 0)
    # now find the "optimal" topic mapping by minimizing the loss
    row_ind, col_ind = linear_sum_assignment(cost_matrix) # row_ind is not of interest
#    cost_matrix[row_ind, col_ind].sum()
    return dict(zip(row_ind, col_ind))

def bar_phi_d(d, prob_dict, K, doc_doc_term_dict, doc_term_dict_R31):
    """
    :param d: a string of a nonnegative integer standing for the document id
    :param prob_dict: should be a dictionary with
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
    :returns: an np.array of shape(K,)
    """
    # first find the keys existent in prob_dict that includes 'd'
    Nd = 0 # number of slots/words in doc d
    pks_d = np.zeros(K,)
    for (doc_id, term) in doc_doc_term_dict[str(d)]:
        count = np.sum(doc_term_dict_R31[(doc_id, term)])
        Nd = Nd + np.sum(doc_term_dict_R31[(doc_id, term)])
        pks_d = pks_d + prob_dict[(doc_id, term)] * count
    return pks_d/Nd


###################################################################
################ L2 loss function and its gradient ################
###################################################################

# prepare \bar{\phi}_d for all documents d
def bar_phi_all_d(prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31):
    """
    :param prob_dict: should be a dictionary with
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
    :param D: number of documents considered
    :param K: number of classes of classficiation (e.g. number of topics)
    :param doc_doc_term_dict: a dictionary with key as a string of an integer that
    represents the document ID, and the value as a list of a 2-tuple of strings, 
    where the first element is same as the key and the second element is a term. 
    The value contains and only contains the pair of terms that are existent in that 
    particular document.
    :param doc_term_dict_R31: a dictionary with a key as a 2-tuple of strings ('doc_id', term)
    where the doc_id is a nonnegative integer and the corresponding value as an 
    np array of K elements, where each element represents the frequency of the k-th
    topic assignment
    :returns: a np.array of shape (D, K) of floating numbers
    """
    rv = np.zeros((D, K))
    for d in range(D):
        rv[d,:] = bar_phi_d(d, prob_dict, K, doc_doc_term_dict, doc_term_dict_R31)
    return rv

def l2_loss_eta(eta, y, bar_phi, D, K, f, pr):
    """
    :param eta: should be an np.array of shape (K*K, ), e.g. eta_init
    :param y: the true label(s), should be an integer np.array of shape (n,)
    :param bar_phi: a D by K matrix where each row d equals bar_phi_d for doc d
    :param D: number of documents considered
    :param K: number of classes of classficiation (e.g. number of topics)
    :returns: a nonnegative floating number
    """
    # first convert y to one-hot encoded K-vector
    eta2 = eta.reshape((K, K))
    y_one_hot = np.zeros((D, K))
    y_one_hot[tuple(range(D)), y.astype(int)] = 1
    # now compute the L2 loss
    l2_loss = np.sum(pow((np.matmul(bar_phi, eta2) - y_one_hot).flatten(), 2)) / 2
    #for d in range(D):
    #    b_phi_d = bar_phi_d(d, prob_dict, K, doc_doc_term_dict, doc_term_dict_R31)
    #    l2_loss += pow(np.matmul(b_phi_d, eta) - y_one_hot[d], 2) / 2
    # for monitoring progress
    print('{}   {}   {}   {}   {}'.format(
        eta[0], eta[1], eta[2], eta[3], l2_loss), file = f)
    if pr == True:       
        print('{}   {}   {}   {}   {}'.format(
            eta[0], eta[1], eta[2], eta[3], l2_loss))
    return l2_loss

def l2_loss_eta_gradient(eta, y, bar_phi, D, K, f, pr):
    """
    :param eta: should be an np.array of shape (K*K, ), e.g. eta_init
    :param y: the true label(s), should be an integer np.array of shape (n,)
    :param bar_phi: a D by K matrix where each row d equals bar_phi_d for doc d
    :param D: number of documents considered
    :param K: number of classes of classficiation (e.g. number of topics)
    :returns: an floating number np.array of shape (K*K,)
    """
    # start with a K by K shape np.array for the gradient and flatten at the end
    eta2 = eta.reshape((K, K))
    rv = np.zeros((K, K))
    for idx1 in range(K):
        for idx2 in range(K):
            sum1 = np.sum(pow(bar_phi[:, idx2], 2) * eta2[idx1, idx2] * np.array(idx2 != y))
            sum2 = np.sum((np.matmul(bar_phi, eta2[:,idx2]) - 1) * bar_phi[:, idx2] * np.array(idx2 == y))
            rv[idx1, idx2] = sum1 + sum2
    return rv.flatten()

###################################################################
################## prediction accuracy function ###################
###################################################################

def pred_eta(eta, y, bar_phi):
    """
    :param eta: should be an np.array of shape (K*K, ), e.g. eta_init
    :param y: the true label(s), should be an integer np.array of shape (D,)
    :param bar_phi: a D by K matrix where each row d equals bar_phi_d for doc d
    :param D: number of documents considered
    :param K: number of classes of classficiation (e.g. number of topics)
    :returns: an floating number np.array of shape (K*K,)
    """
    K = bar_phi.shape[1]
    eta2 = eta.reshape((K, K))    
    tmp_prod = np.matmul(bar_phi, eta2)
    pred = np.argmax(tmp_prod, 1)
    return pred


###################################################################
#################### setup for scipy.optimize #####################
###################################################################

doc_term_prob_dict = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_term_prob_dict.npy').item()
doc_doc_term_dict = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_doc_term_dict.npy').item()
doc_term_dict_R31 = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_term_dict_R31.npy').item()
y = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/y.npy')

# T, D, K
T = 8411
D = 2542
K = 2

# bar_phi_d for all documents d
bar_phi = bar_phi_all_d(doc_term_prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31)

#eta_init = np.reshape(np.random.uniform(-1, 1, K * K), (K, K)).flatten() # array([ 0.04383143,  0.39881273, -0.46026206,  0.34896375])


###################################################################
################## try 100 random initial values ##################
###################################################################

#bnds = ((-1, 1), (-1, 1), (-1, 1), (-1, 1))
bnds = (-1, 1)
num_init_vals = 100

eta_init_vals = np.zeros((num_init_vals, K*K))
eta_final_vals = np.zeros((num_init_vals, K*K))
l2_final_vals = np.zeros(num_init_vals)
pred_accuracy = np.zeros(num_init_vals)


seed = 80
np.random.seed(seed)


for i in range(num_init_vals):

    eta_init_random = np.random.uniform(-1, 1, K * K)
    eta_init_vals[i] = eta_init_random
    
    f_cg = open("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_2_topic_initial_val_search_output/seed" + 
                str(seed) + "init_val" + str(i) + ".txt", "a")
    print('{}   {}   {}   {}   {}'.format(
        "eta[0]", "eta[1]", "eta[2]", "eta[3]", "l2_loss"), file = f_cg)
    opts = {'maxiter': None,  # default value.
        'disp': True,  # non-default value.
        'gtol': 1e-5,  # default value.
        'norm': np.inf,  # default value.
        'eps': 1.4901161193847656e-08}  # default value.
    res = optimize.minimize(l2_loss_eta, eta_init_random, jac=l2_loss_eta_gradient,
                         args=(y, bar_phi, D, K, f_cg, False), method='cg',
                         options=opts)
    eta_final_vals[i] = res.x
    l2_final_vals[i] = res.fun
    pred_accuracy[i] = np.mean(pred_eta(res.x, y, bar_phi) == y)
    print(str(i) + "th initial value: ")
    print('{}   {}   {}   {}   {}   {}   {}   {}   {}   {}'.format(
        pred_accuracy[i], res.fun, eta_init_random[0], eta_init_random[1], eta_init_random[2], 
        eta_init_random[3], res.x[0], res.x[1], res.x[2], res.x[3]))
    #min_test = minimize(f,[x_init, y_init], bounds = bnds)
    #print(res.x, res.fun)

# maximim prediction accuracy 0.9429583005507475
np.max(pred_accuracy)
np.argmax(pred_accuracy)
np.mean(pred_eta(eta_final_vals[np.argmax(pred_accuracy)], y, bar_phi) == y) # 0.9429583005507475








