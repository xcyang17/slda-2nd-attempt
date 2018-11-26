# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:14:31 2018

@author: apple
"""
# import packages
from tables import *
import numpy

# load R output
working_dir = "/Files/documents/ncsu/fa18/ST740/final/news_category/R_output/"
i_txt = open(working_dir + "i.txt", "r")
i_txt_lines = i_txt.readlines()
i_txt_lines2 = [line.rstrip('\n') for line in i_txt_lines] # remove newlines '\n'

j_txt = open(working_dir + "j.txt", "r")
j_txt_lines = j_txt.readlines()
j_txt_lines2 = [line.rstrip('\n') for line in j_txt_lines] # remove newlines '\n'

v_txt = open(working_dir + "v.txt", "r")
v_txt_lines = v_txt.readlines()
v_txt_lines2 = [line.rstrip('\n') for line in v_txt_lines] # remove newlines '\n'

terms_txt = open(working_dir + "terms.txt", "r")
terms_txt_lines = terms_txt.readlines()
terms_txt_lines2 = [line.rstrip('\n') for line in terms_txt_lines] # remove newlines '\n'

docs_txt = open(working_dir + "docs.txt", "r")
docs_txt_lines = docs_txt.readlines()
docs_txt_lines2 = [line.rstrip('\n') for line in docs_txt_lines] # remove newlines '\n'

r_output = [i_txt_lines2, j_txt_lines2, v_txt_lines2, terms_txt_lines2, 
            docs_txt_lines2]

for i in range(4):
    print(r_output[i][0:10])


# Gibbs sampler
L = 5000 # number of iterations
K = 31 # number of topics


# need word count matrix











