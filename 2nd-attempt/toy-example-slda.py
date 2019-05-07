#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:03:11 2019

@author: apple
"""

import numpy as np
import timeit
from collections import Counter
from scipy.optimize import linear_sum_assignment
import random

random.seed(1)

T = 5
D = 3
K = 2

doc_term_dict = {('0', 'eat'): np.random.choice(K, 1),
                 ('0', 'fish'): np.random.choice(K, 1),
                 ('0', 'vegetable'): np.random.choice(K, 1),
                 ('1', 'fish'): np.random.choice(K, 1),
                 ('1', 'pet'): np.random.choice(K, 1),
                 ('2', 'kitten'): np.random.choice(K, 1),
                 ('2', 'eat'): np.random.choice(K, 1),
                 ('2', 'fish'): np.random.choice(K, 1)}
j_r = np.array([0, 1, 2, 1, 3, 4, 0, 1]) # ~ terms
i_r = np.array([0, 0, 0, 1, 1, 2, 2, 2]) # ~ docs
v_r = np.array([1, 1, 1, 1, 1, 1, 1, 1])


doc_term_dict_R31 = {}
doc_topic_mat = np.zeros((D, K))
term_topic_mat = np.zeros((T, K))
topic_mat = np.zeros((1, K))


for idx in range(8):
    k = [j for j in doc_term_dict.keys()][idx]
    freq = Counter(doc_term_dict[k])
    tmp_dict = dict(freq)
    tmp_array = np.array(list(tmp_dict.items()))
    tmpp = np.zeros((1, K))
    tmpp[0,tmp_array[:,0]] = tmp_array[:,1]
    doc_term_dict_R31[k] = tmpp    
    for existent_topic in list(freq.keys()):
        ## update n_{doc, topic}
        # (d, k)-th entry in n_{doc, topic} = 
        # number of times the k-th topic being assigned to terms in 
        # the d-th document (including replicates)
        doc_topic_mat[int(k[0]), existent_topic] += freq[existent_topic]
        ## update n_{term, topic}
        # (w,k)-th entry in n_{term, topic} = 
        # number of times the w-th term being assigned to the k-th topic
        # acorss all D documents
        term_topic_mat[j_r[idx], existent_topic] += freq[existent_topic]
        ## update n_{topic}
        # k-th entry in n_{topic} = number of times the k-th topic
        # is being assigned to a term across all D documents
        topic_mat[0, existent_topic] += freq[existent_topic]


term_id_dict = {'eat': 0, 'fish': 1, 'vegetable': 2, 'pet': 3, 'kitten': 4}
news_id_dict = {'0': 0, '1': 1, '2': 2}
doc_term_prob_dict = {k: np.zeros(K) for k in doc_term_dict_R31.keys()}

# save the original matrix
doc_topic_mat_orig = doc_topic_mat
term_topic_mat_orig = term_topic_mat
topic_mat_orig = topic_mat
doc_term_dict_R31_orig = dict(doc_term_dict_R31) # actual copy of the dictionary
doc_term_dict_orig = doc_term_dict

# initialize and will update the items in MCMC iterations
doc_term_prob_dict = {k: np.zeros(K) for k in doc_term_dict_R31.keys()}

# a dictionary that maps the doc id to the the existent pairs of doc id and term
doc_doc_term_dict = {}
#tmp = [k for k in doc_term_dict_R31.keys()]
#for doc_id in docs_txt_lines2:
#    doc_doc_term_dict[doc_id] = [t for t in filter(lambda x: x[0] == doc_id, tmp)]
doc_doc_term_dict['0'] = [('0', 'eat'), ('0', 'fish'), ('0', 'vegetable')]
doc_doc_term_dict['1'] = [('1', 'fish'), ('1', 'pet')]
doc_doc_term_dict['2'] = [('2', 'kitten'), ('2', 'eat'), ('2', 'fish')]

# true label (not provided in the original tutorial: https://algobeans.com/2015/06/21/laymans-explanation-of-topic-modeling-with-lda-2/)
y = np.array([0, 1, 0])



# Implementation of Figure 2 in http://proceedings.mlr.press/v13/xiao10a/xiao10a.pdf
# Gibbs sampler
MCMC_iters = 1000 # number of iterations
a = 0.1 # entry in alpha
b = 0.1 # entry in beta

# store the estimated \theta_{term, topic} and \phi_{doc, topic}
theta_sample = np.zeros((T, K ,MCMC_iters))
phi_sample = np.zeros((D, K, MCMC_iters))

start_1 = timeit.default_timer()
for m in range(MCMC_iters):
    start_loop1 = timeit.default_timer()
    print(m)
    for (key,value) in doc_term_dict.items():
        news_id = key[0]
        term = key[1]
        term_id = term_id_dict[term]
        Ndi = len(value)
        doc_topic_mat[int(news_id),:] = doc_topic_mat[int(news_id),:] - doc_term_dict_R31[(news_id, term)]
        term_topic_mat[term_id, :] = term_topic_mat[term_id, :] - doc_term_dict_R31[(news_id, term)]
        topic_mat[0,:] = topic_mat[0,:] - doc_term_dict_R31[(news_id, term)]
        # compute the multinomial probability
        pks = []
        for k in range(K):
            pks += [((doc_topic_mat[int(news_id), k] + 
                    a)*(term_topic_mat[term_id, k]+b))/(topic_mat[0,k]+b*T)]
        norm_cnst = sum(pks)
        doc_term_prob_dict[key] = np.array(pks/norm_cnst) #TODO: store pks
        #Ndi = len(doc_term_dict[(news_id, term)])
        k_Ndi_samples = np.random.multinomial(Ndi, pks/norm_cnst, size = 1)
        doc_term_dict_R31[(news_id, term)] = k_Ndi_samples
        # update n_{doc, topic}, n_{term, topic} and n_{topic}
        doc_topic_mat[int(news_id), :] = doc_topic_mat[int(news_id),:] + doc_term_dict_R31[(news_id, term)]
        term_topic_mat[term_id, :] = term_topic_mat[term_id, :] + doc_term_dict_R31[(news_id, term)]
        topic_mat[0,:] = topic_mat[0,:] + doc_term_dict_R31[(news_id, term)]
    stop_loop1 = timeit.default_timer()
    print(m, '-th loop, ', 'Time of d-loop: ', stop_loop1 - start_loop1)
    # update phi (see pg.73 of http://proceedings.mlr.press/v13/xiao10a/xiao10a.pdf)
    # this phi_{d,k} here looks like \theta_{d,z} in U Guleph tutorial
    start_loop2 = timeit.default_timer()
    for d1 in range(D):
        news_id1 = d1
        sum_C_d_k = sum(doc_topic_mat[int(news_id1),:])
        phi_sample[int(news_id1),:,m] = (doc_topic_mat[int(news_id1),:] + a)/(sum_C_d_k + K*a)  
    stop_loop2 = timeit.default_timer()
    print(m, '-th loop, ', 'Time of d1-loop: ', stop_loop2 - start_loop2)
    # update theta (see pg.73 of http://proceedings.mlr.press/v13/xiao10a/xiao10a.pdf)
    # this theta_{v,k} here looks like \phi_{w,z} (= \phi_{z,w}) in U Guleph tutorial
    start_loop3 = timeit.default_timer()
    for k1 in range(K):
        sum_C_v_k = sum(term_topic_mat[:,k1])
        theta_sample[:,k1,m] = (term_topic_mat[:,k1] + b)/(sum_C_v_k + T*b)
    stop_loop3 = timeit.default_timer()
    print(m, '-th loop, ', 'Time of k1-loop: ', stop_loop3 - start_loop3)
stop_1 = timeit.default_timer()    
print('Time: ', stop_1 - start_1) # 622.18273554


# generate prediction using value of phi (document x topic x iterations)
# the result oscilates at particular iterations
for i in np.linspace(99, MCMC_iters-1, num = (MCMC_iters-1-99)/100 + 1):
    print(phi_sample[:,:,int(i)])

# see if average of iterations excluding burn-in would provide better result
# no, it does not provide better result, bad margin
# certain burn_in even gives only 1/3 classification accuracy, e.g. 989
burn_in = 499
np.mean(phi_sample[:,:, burn_in:(MCMC_iters-1)], axis = 2)

# now try supervised LDA
# spent a bit of time to import the script
import sys
sys.path.append('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/')
import prototype_fns

## sanity check - check if the computed topic_mapping gives the best prediction accuracy
t_mapping = prototype_fns.topic_mapping(
        y, np.mean(phi_sample[:,:,burn_in:(MCMC_iters-1)], axis = 2), 2)
phi_mean = np.mean(phi_sample[:,:,199:499], axis = 2)
phi_pred = np.zeros(phi_mean.shape[0])
for i in range(phi_mean.shape[0]):
    phi_pred[i] = t_mapping[phi_mean[i,:].argmax(0)]
# prediction accuracy is
np.mean(1 - (y - phi_pred))

# now try slda
eta_init = np.reshape(np.random.uniform(-1, 1, K*K), (K, K))

from scipy import optimize

args = (y, )
opts = {'maxiter' : None,    # default value.
        'disp' : True,    # non-default value.
        'gtol' : 1e-5,    # default value.
        'norm' : np.inf,  # default value.
        'eps' : 1.4901161193847656e-08}  # default value.
res2 = optimize.minimize(prototype_fns.log_lik_eta, eta_init, jac=prototype_fns.log_lik_eta_grad, 
                         args=args, method='CG', options=opts)









