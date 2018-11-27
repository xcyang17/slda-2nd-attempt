# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:14:31 2018

@author: apple
"""
# import packages
from tables import *
import numpy as np
import timeit
from collections import Counter
from operator import itemgetter
import random

# load R output
working_dir = "/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/news_category/R_output/"
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

for i in range(4):
    print(r_output[i][0:10])

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
print('Time: ', stop - start) # 38.389838336006505

# define dictionaries for term_id and news_id
T = len(terms_txt_lines2)
D = len(docs_r)
term_id_dict = dict(zip(terms_txt_lines2, range(len(terms_txt_lines2))))
news_id_dict = dict(zip(docs_txt_lines2, range(len(docs_txt_lines2))))
alpha = np.ones((1, K))
beta = np.ones((1, T))
## TODO: np.where returns weird result

# Gibbs sampler
L = 5000 # number of iterations
L = 5

# store the estimated \theta_{term, topic} and \phi_{doc, topic}
theta_sample = np.zeros((T, K ,L))


start_1 = timeit.default_timer()
for d in range(D):
    news_id = docs_txt_lines2[d]
    tmp = j_array[np.where(i_array == news_id_dict[news_id])] # as `lst` in original pseudo code
    Wd = list(itemgetter(*tmp.tolist())(terms_txt_lines2))
    Nd = len(Wd)
    for i in range(Nd):
        term = Wd[i]
        term_id = term_id_dict[term]
        Idi = len(doc_term_dict[(news_id, term)])
        for j in range(Idi):
            k_hat = doc_term_dict[(news_id, term)][j]
            doc_topic_mat[int(news_id), k_hat] -= 1 # C_{d, \hat{k}} -= 1
            term_topic_mat[term_id, k_hat] -= 1 # C_{v, \hat{k}} -= 1
            pks = []            
            for k in range(K):
                pks += [((doc_topic_mat[int(news_id), k] + 
                        alpha[0,k])*(term_topic_mat[term_id, k]+beta[0,k]))/(topic_mat[0,k]+beta[0,k]*T)]
            # then normalize
            norm_cnst = sum(pks)
            k_sampled = np.random.choice(K, 1, p = pks/norm_cnst)[0]
            doc_topic_mat[int(news_id), k_sampled] += 1
            term_topic_mat[term_id, k_sampled] += 1
            doc_term_dict[(news_id, term)][j] = k_sampled

stop_1 = timeit.default_timer()    
print('Time: ', stop_1 - start_1)


# understanding 3d array in numpy
np.asarray(list(range(48))).reshape((6,4,2))



rv = 0
for idx in range(len(Wd)):
    print(Wd[idx], len(doc_term_dict[('82341', Wd[idx])]))
    rv += len(doc_term_dict[('82341', Wd[idx])])















