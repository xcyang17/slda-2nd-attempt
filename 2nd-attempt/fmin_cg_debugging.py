
import numpy as np
import timeit
from collections import Counter
from scipy.optimize import linear_sum_assignment
import random
from scipy import optimize
import sys
sys.path.append('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/')

# load dictionaries
doc_term_prob_dict = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_term_prob_dict.npy').item()
doc_doc_term_dict = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_doc_term_dict.npy').item()
doc_term_dict_R31 = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/doc_term_dict_R31.npy').item()
y = np.load('/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/2nd-attempt/2topic_obj/y.npy')

# T, D, K
T = 8411
D = 2542
K = 2

# define functions
import numpy as np
from scipy.optimize import linear_sum_assignment

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
    print(d)
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

def log_lik_eta(eta, y, prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31):
    """
    :param y: the true label(s), should be an np.array of shape (n,), and each element
    in y is an integer in {0, 1, ..., K-1}
    :param phi: should be an np.array of shape (n, K), e.g. phi_sample[:,:,MCMC_iters-1]
    :param eta: should be an np.array of shape (K, K), e.g. eta_init
    :param prob_dict: should be a dictionary with
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
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
    return rv


def log_lik_eta_grad_c_i(eta, y, c, i, prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31):
    """
    :param y: the true label(s), should be an np.array of shape (n,)
    :param phi: should be an np.array of shape (n, K), e.g. phi_sample[:,:,MCMC_iters-1]
    :param eta: should be an np.array of shape (K, K), e.g. eta_init
    :parap c: c for c-th column in eta, an integer in {0, 1, 2, ..., K-1 (= C-1)}
    :param i: i for i-th entry in \eta_c (\eta_c is in R^C, C = K in this particular problem)
    :param prob_dict: should be a dictionary with
    (key,value) = (('doc_id', 'term'), np.array(K,)), e.g. doc_term_prob_dict
    :param D: number of documents in the corpora
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
def log_lik_eta_grad(eta, y, prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31):
    grad_arr = np.zeros((K, K))
    for c in range(K):
        for i in range(K):
            grad_arr[i, c] = log_lik_eta_grad_c_i(eta, y, c, i, prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31)
    return grad_arr.flatten()


# random initialization of eta
eta_init = np.reshape(np.random.uniform(-1, 1, K * K), (K, K)).flatten()

# debugging 1 (done)
#log_lik_eta_grad_c_i(eta_init, y, 0, 0, doc_term_prob_dict, D, K,
#                     doc_doc_term_dict, doc_term_dict_R31)

# debugging 2 (done)
log_lik_eta(eta_init, y, doc_term_prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31)


# options of fmin_cg
opts = {'maxiter': None,  # default value.
        'disp': True,  # non-default value.
        'gtol': 1e-5,  # default value.
        'norm': np.inf,  # default value.
        'eps': 1.4901161193847656e-08}  # default value.
#res2 = optimize.minimize(prototype_fns.log_lik_eta, eta_init, jac=prototype_fns.log_lik_eta_grad,
#                         args=(y, doc_term_prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31), method='CG',
#                         options=opts)

res2 = optimize.minimize(log_lik_eta, eta_init, jac=log_lik_eta_grad,
                         args=(y, doc_term_prob_dict, D, K, doc_doc_term_dict, doc_term_dict_R31), method='CG',
                         options=opts)

