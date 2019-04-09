#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:45:13 2019

@author: apple
"""

# import packages
import numpy as np
import timeit
from collections import Counter
import random
import re
from scipy.optimize import linear_sum_assignment

# load R output
working_dir = "/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/unsupervised-2-topics/R_output/CRIME_EDUCATION/"
i_txt = open(working_dir + "i.txt", "r")
i_txt_lines = i_txt.readlines()
i_txt_lines2 = [line.rstrip('\n') for line in i_txt_lines] # remove newlines '\n'
i_r = list(map(int, i_txt_lines2)) # turn the list of strings into a list of ints
i_array = np.asarray(i_r)

j_txt = open(working_dir + "j.txt", "r")
j_txt_lines = j_txt.readlines()
j_txt_lines2 = [line.rstrip('\n') for line in j_txt_lines] # remove newlines '\n'
j_r = list(map(int, j_txt_lines2))
j_array = np.asarray(j_r)

v_txt = open(working_dir + "v.txt", "r")
v_txt_lines = v_txt.readlines()
v_txt_lines2 = [line.rstrip('\n') for line in v_txt_lines] # remove newlines '\n'
v_r = list(map(int, v_txt_lines2))
v_array = np.asarray(v_r)

terms_txt = open(working_dir + "terms.txt", "r")
terms_txt_lines = terms_txt.readlines()
terms_txt_lines2 = [line.rstrip('\n') for line in terms_txt_lines] # remove newlines '\n'
terms_txt_lines2 = [line.replace('"', '') for line in terms_txt_lines2] # remove double quotes

docs_txt = open(working_dir + "docs.txt", "r")
docs_txt_lines = docs_txt.readlines()
docs_txt_lines2 = [line.rstrip('\n') for line in docs_txt_lines] # remove newlines '\n' # a list of doc IDs
# ^^^ keep news_id as character instead of number
docs_r = list(map(int, docs_txt_lines2))
docs_array = np.asarray(docs_r)

r_output = [i_r, j_r, v_r, terms_txt_lines2, docs_r]

#for i in range(4):
#    print(r_output[i][0:10])

# initializing the \bar{Z} at random from a uniform multinomial from (1, ..., K = 31)
K = 2 # number of topics
doc_term_dict = dict() # in fact this dictionary is not used in the sampling part. could be removed with some change
doc_term_dict_R31 = dict() # add this dictionary to store the R^{31} form representation of doc_term_dict
doc_topic_mat = np.zeros((len(docs_r), K))
term_topic_mat = np.zeros((len(terms_txt_lines2), K))
topic_mat = np.zeros((1, K))

random.seed(9)

start = timeit.default_timer()
for idx in range(len(j_r)):
    news_id = docs_txt_lines2[i_r[idx]] # i_r[idx] as id for `doc`
    term = terms_txt_lines2[j_r[idx]] # j_r[idx] as id for `term`
    k = (news_id, term)
    val_len = v_r[idx]
    doc_term_dict[k] = np.random.choice(K, val_len)
    # store the n_{doc, topic}, n_{term, topic} and n_{topic}
    freq = Counter(doc_term_dict[k])
    tmp_dict=dict(freq)
    tmp_array = np.array(list(tmp_dict.items()))
    tmpp = np.zeros((1, K))
    tmpp[0,tmp_array[:,0]] = tmp_array[:,1]
    doc_term_dict_R31[k] = tmpp    
    for existent_topic in list(freq.keys()):
        ## update n_{doc, topic}
        # (d, k)-th entry in n_{doc, topic} = 
        # number of times the k-th topic being assigned to terms in 
        # the d-th document (including replicates)
        doc_topic_mat[int(news_id), existent_topic] += freq[existent_topic]
        ## update n_{term, topic}
        # (w,k)-th entry in n_{term, topic} = 
        # number of times the w-th term being assigned to the k-th topic
        # acorss all D documents
        term_topic_mat[j_r[idx], existent_topic] += freq[existent_topic]
        ## update n_{topic}
        # k-th entry in n_{topic} = number of times the k-th topic
        # is being assigned to a term across all D documents
        topic_mat[0,existent_topic] += freq[existent_topic]

stop = timeit.default_timer()    
print('Time: ', stop - start) # 1.2487086339999998 seconds

# define dictionaries for term_id and news_id
T = len(terms_txt_lines2)
D = len(docs_r)
term_id_dict = dict(zip(terms_txt_lines2, range(len(terms_txt_lines2))))
news_id_dict = dict(zip(docs_txt_lines2, range(len(docs_txt_lines2)))) # never used?

# save the original matrix
doc_topic_mat_orig = doc_topic_mat
term_topic_mat_orig = term_topic_mat
topic_mat_orig = topic_mat
doc_term_dict_R31_orig = dict(doc_term_dict_R31) # actual copy of the dictionary
doc_term_dict_orig = doc_term_dict

# newly added for slda - April 6, 2019
# a dictionary that stores the Gibbs sampling probabilities pks below
# initialzed with zero arrays of length K (K = 2 here)
# will replace these zero arrays with pks then
doc_term_prob_dict = {k: np.zeros(K) for k in doc_term_dict_R31.keys()}

# newly added for slda - April 6, 2019
# a dictionary that maps the doc id to the the existent pairs of doc id and term
doc_doc_term_dict = {}
tmp = [k for k in doc_term_dict_R31.keys()]
for doc_id in docs_txt_lines2:
    doc_doc_term_dict[doc_id] = [t for t in filter(lambda x: x[0] == doc_id, tmp)]


# TODO: need to randomly initialize some eta (K by K matrix) for later use
# in the comment below, use n instead of eta for simplicity, one example for K = 3:
# suppose we arrange it by [n_11, n_12, n_13, n_21, n_22, n_23, n_31, n_32, n_33]
# where n_i = [n_i1, n_i2, n_i3] for i = 1, 2, 3
eta_init = np.random.uniform(-1, 1, K*K)

# true label as an np.array
#y = np.array([0, 1, 0])
category_txt = open(working_dir + "category.txt", "r")
category_txt_lines = category_txt.readlines()
category_txt_lines2 = [line.rstrip('\n') for line in category_txt_lines] # remove newlines '\n'
category_txt_lines3 = [line[1:-1] for line in category_txt_lines2] # remove double quotes
# a dictionary mapping topic (e.g. CRIME) to {0, 1, ..., K-1}
topic_num_dict = dict(zip(set(category_txt_lines3), range(K)))
# create labels according to this dictionary
y = np.zeros(len(category_txt_lines3))
for topic in set(category_txt_lines3):
    tmp = [True if item == topic else False for item in category_txt_lines3]
    y[tmp] = topic_num_dict[topic]


# Implementation of Figure 2 in http://proceedings.mlr.press/v13/xiao10a/xiao10a.pdf
# Gibbs sampler
MCMC_iters = 500 # number of iterations
a = 0.1 # entry in alpha
b = 0.1 # entry in beta

# store the estimated \theta_{term, topic} and \phi_{doc, topic}
theta_sample = np.zeros((T, K ,MCMC_iters))
phi_sample = np.zeros((D, K, MCMC_iters))

start_1 = timeit.default_timer()
for m in range(MCMC_iters):
    start_loop1 = timeit.default_timer()
    print(m)
    for (key,value) in doc_term_dict_R31.items():
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
        news_id1 = docs_txt_lines2[d1]
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

## sanity check - check if the computed topic_mapping gives the best prediction accuracy
t_mapping = topic_mapping(y, np.mean(phi_sample[:,:,199:499], axis = 2))
phi_mean = np.mean(phi_sample[:,:,199:499], axis = 2)
phi_pred = np.zeros(phi_mean.shape[0])
for i in range(phi_mean.shape[0]):
    phi_pred[i] = t_mapping[phi_mean[i,:].argmax(0)]
# prediction accuracy for 'CRIME' class 0.9734848484848485
np.mean(phi_pred[[True if item == 1 else False for item in y]] == y[[True if item == 1 else False for item in y]])
# prediction accuracy for 'EDUCATION' class- 0.8897142857142857
np.mean(phi_pred[[True if item == 0 else False for item in y]] == y[[True if item == 0 else False for item in y]])
# it looks fine on this simplest 2-class sample


# use labels to decide the topic identifibiality issue and output a dictionary
def topic_mapping(y, phi):
    """
    :param y: the true label(s), should be an np.array of shape (n,)
    :param phi: should be an np.array of shape (n, K), e.g. phi_sample[:,:,MCMC_iters-1]
    :returns: a dictionary with (key, value) = (topic, index), where topic is 
    a nonnegative integer, and index is the corresponding entry for the topic in a row of phi
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


def log_lik_eta(y, phi, eta):
    """
    :param y: the true label(s), should be an np.array of shape (n,)
    :param phi: should be an np.array of shape (n, K), e.g. phi_sample[:,:,MCMC_iters-1]
    :param eta: should be an np.array of shape (K, K), e.g. eta_init
    :returns: a dictionary with (key, value) = (topic, index), where topic is 
    a nonnegative integer, and index is the corresponding entry for the topic in a row of phi
    """
    



def bar_phi_d(d, prob_dict):
    """
    :param d: a string of a nonnegative integer standing for the document id
    :param prob_dict: should be a dictionary with 
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
    :returns: an np.array of shape(K,)
    """
    # first find the keys existent in prob_dict that includes 'd'
    Nd = 0
    for (key, value) in doc_doc_term_dict[d]:
        Nd = Nd + len()
    
    
    

def kappa_d(d, prob_dic):
    



# TODO: a function that computes the class using the estimated theta
# April 8, 2019: maybe do not implement this function for now. try sampling as in the model
# on pg.2 of the image annotation paper
# NOTE: maybe later on we could try using the pks instead. try it if our method works...
def class_doc_by_phi(pks):
    """
    :param pks: should be an np.array of shape (K,), e.g. phi_sample[:,:,MCMC_iters-1]
    :returns: a tuple consisting 
    (1) one-hot encoded np.array of shape (K,) for the predicted topic/class
    (2) a number in {0, 1, ..., K-1} as the index for the nonzero entry as the predicted topic/class
    """
    # return the predicted class by selecting the one with largest value in PHI
    # PHI should be an np.array of shape (K,) and could be from phi_sample
    # returns a tuple consisting 
    # (1) one-hot encoded np.array of shape (K,)
    # (2) a number in {0, 1, ..., K-1} as the index for the nonzero entry
    





# store the files
save_dir = "/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/ecgs-2-topics/"
np.save(save_dir+"theta-unsupervised-ecgs-2-topics.npy", theta_sample)
np.save(save_dir+"phi-unsupervised-ecgs-2-topics.npy", phi_sample)

# save the estimates from last 10 iterations
#theta_sample = np.load(save_dir + "theta-unsupervised-crime-education.npy")
#phi_sample = np.load(save_dir + "phi-unsupervised-crime-education.npy")

np.save(save_dir+"theta-ecgs-2-topics-last-10-iters.npy", theta_sample[:,:,489:499])
np.save(save_dir+"phi-ecgs-2-topics-last-10-iters.npy", phi_sample[:,:,489:499])



