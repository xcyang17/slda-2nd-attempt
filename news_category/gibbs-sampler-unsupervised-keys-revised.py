#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:09:23 2018

@author: CuiCan
"""

import numpy as np
import timeit
from collections import Counter
from operator import itemgetter
import random
import math
import pandas as pd

# load R output
working_dir = "/Users/CuiCan/Desktop/Slides/NCState/ST 740/Final Project/ST740-FA18-Final/news_category/R_output/"
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

r_output = [i_r, j_r, v_r, terms_txt_lines2, docs_r]


# initializing the \bar{Z} at random from a uniform multinomial from (1, ..., K = 31)
K = 31 # number of topics
doc_term_dict = dict()
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
print('Time: ', stop - start) # 45 seconds

# define dictionaries for term_id and news_id
T = len(terms_txt_lines2)
D = len(docs_r)
term_id_dict = dict(zip(terms_txt_lines2, range(len(terms_txt_lines2))))
news_id_dict = dict(zip(docs_txt_lines2, range(len(docs_txt_lines2))))

a = 0.1 # entry in alpha
b = 0.1 # entry in beta

MCMC_iters = 500

# store the estimated \theta_{term, topic} and \phi_{doc, topic}
theta_sample = np.zeros((T, K, MCMC_iters))
phi_sample = np.zeros((D, K, MCMC_iters))

random.seed(1111)
start_3 = timeit.default_timer()
for m in range(MCMC_iters):
    l=0
    for (key,value) in doc_term_dict.items(): # doc level
        l+=1
        if l%10000 == 0:
            print(l)
        news_id = key[0]
        term = key[1]
        term_id = term_id_dict[term]
        Idi = len(value)
        for j in range(Idi): # replicates of the term
            k_hat = doc_term_dict[(news_id, term)][j]
            doc_topic_mat[int(news_id), k_hat] -= 1 # C_{d, \hat{k}} -= 1
            term_topic_mat[term_id, k_hat] -= 1 # C_{v, \hat{k}} -= 1
            pks = []            
            for k in range(K):
                pks += [((doc_topic_mat[int(news_id), k] + 
                                        alpha[0,k])*(term_topic_mat[term_id, k]+beta[0,k]))/(topic_mat[0,k]+beta[0,k]*T)]
            # then normalize
            norm_cnst = sum(pks)
            k_sampled = np.random.choice(a = K, size = 1, p = pks/norm_cnst)[0]
            doc_topic_mat[int(news_id), k_sampled] += 1
            term_topic_mat[term_id, k_sampled] += 1
            doc_term_dict[(news_id, term)][j] = k_sampled
stop_3 = timeit.default_timer()    
print('Time: ', stop_3 - start_3) # 197.33248764699965
for d1 in range(D):
    news_id1 = docs_txt_lines2[d1]
    sum_C_d_k = sum(doc_topic_mat[int(news_id1),:])
    phi_sample[int(news_id1),:,0] = (doc_topic_mat[int(news_id1),:] + a)/(sum_C_d_k + K*a)        
# update theta (see pg.73 of http://proceedings.mlr.press/v13/xiao10a/xiao10a.pdf)
# this theta_{v,k} here looks like \phi_{w,z} (= \phi_{z,w}) in U Guleph tutorial
for k1 in range(K):
    sum_C_v_k = sum(term_topic_mat[:,k1])
    theta_sample[:,k1,0] = (term_topic_mat[:,k1] + b)/(sum_C_v_k + T*b)
stop_3 = timeit.default_timer()    
print('Time: ', stop_3 - start_3) # 199.98975172999997