#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:54:02 2019

@author: apple
"""

# the goal is to investigate the instable performance of eta (optimized using 
# conjugdate gradient)


import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import timeit
from scipy import optimize
import sys
sys.path.append('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/')


# define functions
# use labels to decide the topic identifibiality issue and output a dictionary
def topic_mapping(y, phi, K):
    """
    :param y: the true label(s), should be an np.array of shape (n,)
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

# this function may be completely unnecessary
def bar_phi_d_i(d, i, prob_dict, K, doc_doc_term_dict, doc_term_dict_R31):
    """
    :param d: a string of a nonnegative integer standing for the document id
    :param prob_dict: should be a dictionary with
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
    :param i: an index that is in {0, 1, 2, ..., K-1}
    :returns: a floating number
    """
    rv = bar_phi_d(d, prob_dict, K, doc_doc_term_dict, doc_term_dict_R31)
    return rv[i]

def kappa_d(d, prob_dict, eta, K, doc_doc_term_dict, doc_term_dict_R31):
    """
    :param d: a string of a nonnegative integer standing for the document id
    :param prob_dict: should be a dictionary with
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
    :param eta: a K by K np.array of the eta, c-th column representd \eta_c in the paper
    :returns: a floating number
    """
    # compute Nd [this is repetitive work, but do not how to optimize it yet]
    Nd = 0
    terms_d = []
    for (doc_id, term) in doc_doc_term_dict[str(d)]:
        Nd = Nd + np.sum(doc_term_dict_R31[(doc_id, term)])
        terms_d += [term]
    # create a matrix of pks_(doc, term), each row as pks for a term
    pks_mat = np.zeros((len(terms_d), K))
    for idx in range(len(terms_d)):
        pks_mat[idx, :] = prob_dict[(str(d), terms_d[idx])]
    # generate exp(\eta_c / N_d) as a matrix
    exp_eta_mat = np.exp(eta / Nd)
    # compute inner product between pks_(doc, term) and exp(\eta_c / Nd) for fixed t and c
    tmp = np.zeros(K) # will store the summands of \Sigma_{c=1}^C
    for c in range(K):
        tmpp = np.zeros(len(terms_d))
        for t in range(len(terms_d)):
            tmpp[t] = np.inner(pks_mat[t,:], exp_eta_mat[:,c])
        tmp[c] = np.prod(tmpp)
    return np.sum(tmp)

# changed the sign of log_lik_eta
def log_lik_eta(eta, y, prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31, f):
    """
    :param y: the true label(s), should be an np.array of shape (n,), and each element
    in y is an integer in {0, 1, ..., K-1}
    :param phi: should be an np.array of shape (n, K), e.g. phi_sample[:,:,MCMC_iters-1]
    :param eta: should be an np.array of shape (K, K), e.g. eta_init
    :param prob_dict: should be a dictionary with
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
    :param f: a file object to where the eta and negative log likelihood is stored
    :returns: a dictionary with (key, value) = (topic, index), where topic is
    a nonnegative integer, and index is the corresponding entry for the topic in a row of phi
    """
    # will call bar_phi_d() and kappa_d()
    eta = eta.reshape((K, K))
    tmp = 0
    sum_log_kappa_d = 0
    for d in range(D):
        # compute inner product of \eta_{c_d} and \bar{\phi}_d
        d_true_label = int(y[d])
        tmp += np.inner(eta[:, d_true_label], bar_phi_d(d, prob_dict, K, doc_doc_term_dict, doc_term_dict_R31))
        sum_log_kappa_d += np.log(kappa_d(d, prob_dict, eta, K, doc_doc_term_dict, doc_term_dict_R31))
    rv = tmp - sum_log_kappa_d
    # for monitoring progress
    eta_tmp = eta.flatten()
    print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(
        eta_tmp[0], eta_tmp[1], eta_tmp[2], eta_tmp[3], -rv), file = f)
    print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(
        eta_tmp[0], eta_tmp[1], eta_tmp[2], eta_tmp[3], -rv))
    return -rv


def log_lik_eta_grad_c_i(eta, y, c, i, prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31, f):
    """
    :param y: the true label(s), should be an np.array of shape (n,)
    :param phi: should be an np.array of shape (n, K), e.g. phi_sample[:,:,MCMC_iters-1]
    :param eta: should be an np.array of shape (K, K), e.g. eta_init
    :parap c: c for c-th column in eta, an integer in {0, 1, 2, ..., K-1 (= C-1)}
    :param i: i for i-th entry in \eta_c (\eta_c is in R^C, C = K in this particular problem)
    :param prob_dict: should be a dictionary with
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
    :param D: number of documents in the corpora
    :param f: a file object where the printout of the input eta and the negative log likelihood
    :returns: a dictionary with (key, value) = (topic, index), where topic is
    a nonnegative integer, and index is the corresponding entry for the topic in a row of phi
    """
    ### this function can be further optimizied, as \phi_{di} for some doc is computed more than once
    # compute the second sum over d = 1, ...., D
    # first find out all documents with true label c
    doc_label_c = np.linspace(0, len(y)-1, num = len(y))[y == c]
    doc_label_c = doc_label_c.astype(int)
    # compute \bar{\phi_{di}} for doc id in doc_label_c
    sum1 = 0
    for d in doc_label_c:
        sum1 += bar_phi_d_i(d, i, prob_dict, K, doc_doc_term_dict, doc_term_dict_R31)
    # compute the second sum over d = 1, ...., D
    sum2 = 0
    for d in range(D):
        Nd = 0 # compute Nd [this is repetitive work, but do not know how to optimize it yet]
        terms_d = [] # list of (unique) terms in doc d
        terms_d_freq = []
        for (doc_id, term) in doc_doc_term_dict[str(d)]:
            Nd = Nd + np.sum(doc_term_dict_R31[(doc_id, term)])
            terms_d += [term]
        # create a matrix of pks_(doc, term), each row as pks for a term
        pks_mat = np.zeros((len(terms_d), K))
        for idx in range(len(terms_d)):
            pks_mat[idx, :] = prob_dict[(str(d), terms_d[idx])]
        # generate exp(\eta_c / N_d) as a matrix
        exp_eta_mat = np.exp(eta / Nd).reshape((K, K))
        kappa_d_summands = np.zeros(K) # will store the summands of \Sigma_{c=1}^C in \kappa_d
        factor2_tmp1 = np.zeros(len(terms_d)) # will be useful for factor2
        factor2_tmp2 = np.zeros(len(terms_d)) # will be useful for factor2
        for cc in range(K):
            tmpp = np.zeros(len(terms_d))
            tmpp2 = np.zeros(len(terms_d))
            for t in range(len(terms_d)):
                tmpp[t] = np.inner(pks_mat[t, :], exp_eta_mat[:, cc])
                tmpp2[t] = pow(tmpp[t], np.sum(doc_term_dict_R31[(str(d), terms_d[t])]))
            kappa_d_summands[cc] = np.prod(tmpp2)
            if cc == c: # for computation of factor2
                factor2_tmp2 = tmpp
                for t in range(len(terms_d)):
                    factor2_tmp1[t] = pks_mat[t,c] * exp_eta_mat[i,c] / Nd
        # the first term to be multiplied in the second sum over d = 1, ..., D
        factor1 = kappa_d_summands[c] / np.sum(kappa_d_summands)
        # compute the second factor to be multiplied
        terms_d_freq = np.zeros(len(terms_d)) # this may be optimized as well
        for t in range(len(terms_d)):
            terms_d_freq[t] = np.sum(doc_term_dict_R31[(str(d), terms_d[t])])
        factor2 = np.inner(factor2_tmp1 / factor2_tmp2, terms_d_freq)
        sum2 += factor1 * factor2
    return sum1 - sum2


# a wrapper that returns the whole gradient (instead of a partial derivative)
def log_lik_eta_grad(eta, y, prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31, f):
    grad_arr = np.zeros((K, K))
    for c in range(K):
        for i in range(K):
            grad_arr[i, c] = log_lik_eta_grad_c_i(eta, y, c, i, prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31, f)
    return grad_arr.flatten()


#############################################################################
#############################################################################
#############################################################################

# load dictionaries
doc_term_prob_dict = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_term_prob_dict.npy').item()
doc_doc_term_dict = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_doc_term_dict.npy').item()
doc_term_dict_R31 = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_term_dict_R31.npy').item()
y = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/y.npy')

# T, D, K
T = 8411
D = 2542
K = 2


# random initialization of eta
seed = 80
random.seed(seed)
#eta_init = np.reshape(np.random.uniform(-1, 1, K * K), (K, K)).flatten()
eta_init = np.identity(K).flatten() + np.random.uniform(-1, 1, K * K)

# save the eta on each iteration to file
f = open("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/output_seed" + str(seed) + "id.txt", "a")
print('{}   {}   {}   {}   {}'.format(
        "eta[0]", "eta[1]", "eta[2]", "eta[3]", "negative_log_lik"), file = f)
print('{}   {}   {}   {}   {}'.format(
        "eta[0]", "eta[1]", "eta[2]", "eta[3]", "negative_log_lik"))
eta_opt_start = timeit.default_timer()
opts = {'maxiter': None,  # default value.
        'disp': True,  # non-default value.
        'gtol': 1e-5,  # default value.
        'norm': np.inf,  # default value.
        'eps': 1.4901161193847656e-08}  # default value.
res4 = optimize.minimize(log_lik_eta, eta_init, jac=log_lik_eta_grad,
                         args=(y, doc_term_prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31, f), method='CG',
                         options=opts)
eta_opt_stop = timeit.default_timer()
print('Time of eta optimization using conjugate gradient: ', eta_opt_stop - eta_opt_start, file = f)
print('Time of eta optimization using conjugate gradient: ', eta_opt_stop - eta_opt_start)
f.close()


eta_final_4 = res4.x.reshape((K,K))
np.save("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/eta_seed" + str(seed) + "id.npy", res4.x)

l4 = log_lik_eta(res4.x, y, doc_term_prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31, None)

# see prediction using the trained eta_final
eta_4_pred = np.zeros(D)
eta_4_prod = np.zeros((D, K))
bar_phi = np.zeros((D,K))
for idx in range(D):
    bar_phi[idx,:] = bar_phi_d(int(idx), doc_term_prob_dict, K, doc_doc_term_dict, doc_term_dict_R31)
    eta_4_prod[idx,:] = np.matmul(bar_phi[idx], eta_final_4)
    eta_4_pred[idx] = np.argmax(eta_4_prod[idx,:])

np.mean(eta_4_pred == y) # 0.6884343036978757 prediction accuracy



# this eta below used to give good result - 0.9287962234461055 prediction accuracy
eta_good = np.array([0.748099, -0.351764, 0.238562, 0.448838]).reshape((K,K))

eta_good_pred = np.zeros(D)
eta_good_prod = np.zeros((D, K))
bar_phi = np.zeros((D,K))
for idx in range(D):
    bar_phi[idx,:] = bar_phi_d(int(idx), doc_term_prob_dict, K, doc_doc_term_dict, doc_term_dict_R31)
    eta_good_prod[idx,:] = np.matmul(bar_phi[idx], eta_good)
    eta_good_pred[idx] = np.argmax(eta_good_prod[idx,:])

np.mean(eta_good_pred == y)



