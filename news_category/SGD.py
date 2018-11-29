#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:44:05 2018
@author: CuiCan
"""

from tables import *
import numpy as np
import timeit
from collections import Counter
from operator import itemgetter
import random
import math
import pandas as pd

# load R output
working_dir = "/Users/CuiCan/Desktop/Slides/NCState/ST 740/FinalProject/ST740-FA18-Final/news_category/R_output/"
#working_dir = "/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/news_category/R_output/"
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

terms_txt = open(working_dir + "Terms.txt", "r")
terms_txt_lines = terms_txt.readlines()
terms_txt_lines2 = [line.rstrip('\n') for line in terms_txt_lines] # remove newlines '\n'
terms_txt_lines2 = [line.replace('"', '') for line in terms_txt_lines2] # remove double quotes

docs_txt = open(working_dir + "Docs.txt", "r")
docs_txt_lines = docs_txt.readlines()
docs_txt_lines2 = [line.rstrip('\n') for line in docs_txt_lines] # remove newlines '\n'
# ^^^ keep news_id as character instead of number
docs_r = list(map(int, docs_txt_lines2))
docs_array = np.asarray(docs_r)

label_txt = open(working_dir + "label.txt", "r")
label_txt_lines = label_txt.readlines()
label_txt_lines2 = [line.rstrip('\n') for line in label_txt_lines]
label_txt_lines2 = [line.replace('"','') for line in label_txt_lines2]

r_output = [i_r, j_r, v_r, terms_txt_lines2, docs_r, label_txt_lines2]

for i in range(6):
    print(r_output[i][0:10])

#for i in range(4):
#    print(r_output[i][0:10])

# initializing the \bar{Z} at random from a uniform multinomial from (1, ..., K = 31)
K = 31 # number of topics
doc_term_dict = dict()
doc_topic_mat = np.zeros((len(docs_r), K))
term_topic_mat = np.zeros((len(terms_txt_lines2), K))
topic_mat = np.zeros((1, K))

doc_label_dict = dict()
for i in range(len(docs_txt_lines2)):
    doc_label_dict[docs_txt_lines2[i]] = label_txt_lines2[i]

freq = Counter(label_txt_lines2)
label_id_dict = dict(zip(freq.keys(),range(31)))

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
alpha = np.ones((1, K))
beta = np.ones((1, T))
## TODO: np.where returns weird result

# Gibbs sampler
# One iteration test time cost
a = 1 # entry in alpha
b = 1 # entry in beta
N = 1 # Number of iterations
# store the estimated \theta_{term, topic} and \phi_{doc, topic}
theta_sample = np.zeros((T, K ,N))
phi_sample = np.zeros((D, K, N))

# Use keys from dict instead
L = len(doc_term_dict.keys())  # 1516837
keys = list(doc_term_dict.keys()) # 1516837

start_3 = timeit.default_timer()
for l in range(L): # doc level
    if l%10000 == 0:
        print(l)
    k = keys[l]
    news_id = k[0]
    term = k[1]
    term_id = term_id_dict[term]
    Idi = len(doc_term_dict[(news_id, term)])
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
        #TODO: ???is topic_mat updated? or is n_{topic} not used in the sampling algo at all?
        #TODO: what if we sampled a k that is previously 
        doc_term_dict[(news_id, term)][j] = k_sampled
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

eta = np.zeros([K,K]) + 1/K
lam = 0.5
step = 1
def eta_gradient(k,n_sample=10,s=1111):
    gradient = np.zeros(K)
    random.seed(s)
    subsample = np.random.choice(range(D),n_sample,replace=False)
    for d in subsample:
        news_id = docs_txt_lines2[d]
        zbar = doc_topic_mat[int(news_id),:]/sum(doc_topic_mat[int(news_id),:])
        Pd = np.zeros((K,1))
        Yd = np.zeros((K,1))
        label = doc_label_dict[news_id]
        Yd[label_id_dict[label],0] = 1
        Pd = np.exp(np.dot(eta,zbar))
        Pd = Pd / sum(Pd)
        gradient = gradient + zbar*(Pd[k] - Yd[k,0])
    const = lam * eta[k,:]
    const = const.reshape((const.shape[0],))
    gradient = gradient/len(subsample) + const
    eta[k,:] = eta[k,:] + gradient * step
    return [gradient,subsample, eta]

start_4 = timeit.default_timer()
result = eta_gradient(k=0)
stop_4 = timeit.default_timer()    
print('Time: ', stop_4 - start_4) # 11.02097644999958

result1 = eta_gradient(k=1)
    
def loss_f(times):
    loss = []
    for t in range(times):
        for j in range(31):
            subsample = eta_gradient(j)[1]
            Ld = np.zeros(1)
            for d1 in subsample:
                news_id = docs_txt_lines2[d1]
                zbar = doc_topic_mat[int(news_id),:]/sum(doc_topic_mat[int(news_id),:])
                Pd = np.zeros((K,1))
                Yd = np.zeros((K,1))
                label = doc_label_dict[news_id]
                Yd[label_id_dict[label],0] = 1
                Pd = np.exp(np.dot(eta,zbar))
                Pd = Pd / sum(Pd)
                Ld = Ld - math.log(Pd[label_id_dict[label]])
            tmp = sum(np.linalg.norm(eta,axis=1)**2)/2
            loss.extend (Ld/len(subsample) + lam * tmp)
    return loss


start_5 = timeit.default_timer()
result1 = loss_f(times=1)
stop_5 = timeit.default_timer()    
print('Time: ', stop_5 - start_5) # 13.434937202000583
result1[1]
# [10.39993332731592,20.277826295450527,42.85497998431295,93.47237347320201,209.48449658610386,471.1744511618956,1062.5875130268887,2391.1916699264057,5386.369525821632,12137.328570608777]
# Increasing