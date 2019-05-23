#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:37:53 2019

@author: apple
"""

###################################################################
######## MCMC sampling to provide mutinomial probabilities ########
###################################################################


import numpy as np
import timeit
from collections import Counter
import random
from scipy.optimize import linear_sum_assignment
from scipy import optimize
import pandas as pd


# load R output
K = 4
working_dir = "/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/unsupervised-2-topics/R_output/CRIME_EDUCATION_SPORTS_RELIGION/"
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
docs_txt_lines2 = [line.rstrip('\n') for line in docs_txt_lines] # remove newlines '\n'
# ^^^ keep news_id as character instead of number
docs_r = list(map(int, docs_txt_lines2))
docs_array = np.asarray(docs_r)

# true label as an np.array
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



# initializing the \bar{Z} at random from a uniform multinomial from (1, ..., K = 31)
doc_term_dict = dict()
doc_term_dict_R31 = dict() # add this dictionary to store the R^{31} form representation of doc_term_dict
doc_topic_mat = np.zeros((len(docs_r), K))
term_topic_mat = np.zeros((len(terms_txt_lines2), K))
topic_mat = np.zeros((1, K))

np.random.seed(9)

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
        doc_term_prob_dict[key] = np.array(pks/norm_cnst)
        
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
print('Time: ', stop_1 - start_1) # 1846.105528517 seconds

# save dictionaries and y
np.save('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/doc_term_prob_dict.npy', doc_term_prob_dict)
np.save('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/doc_doc_term_dict.npy', doc_doc_term_dict)
np.save('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/doc_term_dict_R31.npy', doc_term_dict_R31)
np.save('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/y.npy', y)



###################################################################
############ functions needed for fitting eta parameters ##########
###################################################################


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
################### determine the topic mapping ###################
###################################################################

doc_term_prob_dict = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/doc_term_prob_dict.npy').item()
doc_doc_term_dict = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/doc_doc_term_dict.npy').item()
doc_term_dict_R31 = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/doc_term_dict_R31.npy').item()
y = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/y.npy')

# T, D, K
working_dir = "/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/unsupervised-2-topics/R_output/CRIME_EDUCATION_SPORTS_RELIGION/"
terms_txt = open(working_dir + "terms.txt", "r")
terms_txt_lines = terms_txt.readlines()
terms_txt_lines2 = [line.rstrip('\n') for line in terms_txt_lines] # remove newlines '\n'
terms_txt_lines2 = [line.replace('"', '') for line in terms_txt_lines2]
T = len(terms_txt_lines2)

D = y.shape[0]
K = 4

# bar_phi_d for all documents d, needed for topic mapping
bar_phi = bar_phi_all_d(doc_term_prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31)

# this is the topic mapping
topic_mapping_dict = topic_mapping(y, bar_phi, K) # {0: 0, 1: 1, 2: 3, 3: 2} # 2-index in bar_phi[d] is prob for topic 3

np.save('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/obj/topic_mapping_dict.npy', topic_mapping_dict)


###################################################################
################### determine the topic mapping ###################
###################################################################

# now rearrange columns in bar_phi according to the topic mapping
# so that the rearranged bar_phi would have column i corresponding to topic i

def rearrange_bar_phi(bar_phi, y, K, topic_mapping_dict):
    if topic_mapping_dict == None:
        topic_mapping_dict = topic_mapping(y, bar_phi, K)
    lst = [topic_mapping_dict[idx] for idx in range(K)]
    rv = bar_phi[:,lst]
    return rv

bar_phi_rearranged = rearrange_bar_phi(bar_phi, y, K, topic_mapping_dict)


###################################################################
############### train/test & cross-validation set split ###########
###################################################################

# initial value is what is being cross-validated?
# no, simply use average of eta estimates trained on each fold as the model
# cv score is prediction accuracy (not l2 loss)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bar_phi_rearranged, y, test_size=0.2)

n_folds = 5
group_idx = np.array([random.randint(0, n_folds-1) for p in range(0, X_train.shape[0])])

# 100 random initial values
num_init_vals = 100
eta_init_vals = np.zeros((num_init_vals, K*K))
eta_final_vals = np.zeros((num_init_vals, K*K, n_folds))
eta_ests = np.zeros((num_init_vals, K*K))
eta_cv_vals = np.zeros((num_init_vals, K*K))
cv_scores = np.zeros(num_init_vals)
test_pred_accuracy = np.zeros(num_init_vals)

seed = 80
np.random.seed(seed)

for i in range(num_init_vals):

    eta_init_random = np.random.uniform(-1, 1, K * K)
    eta_init_vals[i] = eta_init_random
    eta_cv_est = np.zeros(K*K)
    eta_cv_score = 0
    
    for k in range(n_folds):
        
        X_train_k = X_train[group_idx != k]
        y_train_k = y_train[group_idx != k]
        X_valid_k = X_train[group_idx == k]
        y_valid_k = y_train[group_idx == k]
        
        f_cg = open("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/l2_loss_4_topic/initial_val_search_output/seed" + 
                    str(seed) + "init_val" + str(i) + "fold" + str(k) + ".txt", "a")
        print('{}   {}   {}   {}   {}'.format(
            "eta[0]", "eta[1]", "eta[2]", "eta[3]", "l2_loss"), file = f_cg)
        opts = {'maxiter': None,  # default value.
            'disp': True,  # non-default value.
            'gtol': 1e-5,  # default value.
            'norm': np.inf,  # default value.
            'eps': 1.4901161193847656e-08}  # default value.
        res = optimize.minimize(l2_loss_eta, eta_init_random, jac=l2_loss_eta_gradient,
                             args=(y_train_k, X_train_k, X_train_k.shape[0], K, f_cg, False), 
                             method='cg', options=opts)
        eta_final_vals[i,:,k] = res.x
        eta_cv_est += res.x
        eta_cv_score += np.mean(pred_eta(res.x, y_valid_k, X_valid_k) == y_valid_k)
        f_cg.close()
        
    # now compute the average of eta estimated on each fold
    eta_cv_est = eta_cv_est / n_folds
    eta_ests[i] = eta_cv_est
    eta_cv_score = eta_cv_score / n_folds # cv score on the "train" set
    eta_cv_vals[i] = eta_cv_est
    cv_scores[i] = eta_cv_score
    # prediction accuracy on the test set
    test_pred_accuracy[i] = np.mean(pred_eta(eta_cv_est, y_test, X_test) == y_test)
    print(str(i) + "th initial value: ")
    print('{}   {}   {}   {}   {}   {}   {}   {}   {}   {}'.format(
        test_pred_accuracy[i], cv_scores[i], eta_init_random[0], eta_init_random[1], eta_init_random[2], 
        eta_init_random[3], eta_cv_est[0], eta_cv_est[1], eta_cv_est[2], eta_cv_est[3]))



###################################################################
########## plot CV score against test prediction accuracy #########
###################################################################

max(cv_scores) # 0.7611340007216864
np.argmax(cv_scores) # 6
test_pred_accuracy[np.argmax(cv_scores)] # 0.7857142857142857
np.argmax(cv_scores) == np.argmax(test_pred_accuracy)

# and the model performing best in cross-validation is also the model performing
# the best on the test set


###################################################################
############## look into prediction accuracy by class #############
###################################################################

# class 0: 0.6738609112709832 test accuracy
np.mean(pred_eta(eta_ests[int(np.argmax(cv_scores))], y_test[y_test == 0], X_test[y_test == 0]) == y_test[y_test == 0])

# class 1: 0.7060931899641577 test accuracy
np.mean(pred_eta(eta_ests[int(np.argmax(cv_scores))], y_test[y_test == 1], X_test[y_test == 1]) == y_test[y_test == 1])

# class 2: 0.9772079772079773 test accuracy
np.mean(pred_eta(eta_ests[int(np.argmax(cv_scores))], y_test[y_test == 2], X_test[y_test == 2]) == y_test[y_test == 2])

# class 3: 0.7953216374269005 test accuracy
np.mean(pred_eta(eta_ests[int(np.argmax(cv_scores))], y_test[y_test == 3], X_test[y_test == 3]) == y_test[y_test == 3])

# todo: examine unbalancedness? and maybe modify the loss function accordingly if needed
for i in range(K):
    print(sum(y == i))


# which topic corresponds to which class?
actual_topic_dict = {} # {0: 'SPORTS', 1: 'RELIGION', 2: 'CRIME', 3: 'EDUCATION'}

for i in range(K):
    actual_topic_dict[int(i)] = (np.array(category_txt_lines3)[y == i])[0]
    
# so: good prediction on CRIME, OK on EDUCATION, and worst on RELIGION / SPORTS


###################################################################
############## look into misclassification  by class ##############
###################################################################

pred_full = pred_eta(eta_ests[int(np.argmax(cv_scores))], y, bar_phi_rearranged)

four_class_df = pd.DataFrame(data = bar_phi_rearranged)
four_class_df['y'] = y.astype('int')
four_class_df['pred'] = pred_full


###################################################################
################# compare with simple argmax method ###############
###################################################################

simple_pred = np.argmax(bar_phi_rearranged, axis = 1)
np.mean(simple_pred == y) # 0.8220506079526783

# when there are 4 topics, the simple method starts to outperform the linear method
# so probably there is non linearity affecting the result.


