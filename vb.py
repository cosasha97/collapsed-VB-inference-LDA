import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import time
from numba import jit
import random
import scipy.special as sc
from scipy.special import gammaln
from scipy.special import logsumexp
from sklearn.decomposition._online_lda_fast import (mean_change, _dirichlet_expectation_1d, _dirichlet_expectation_2d)

def compute_variational_bound(document_word_matrix, phi, log_phi, gamma, lambda_, alpha, eta, K, W, D, W_array):
    """
    Compute the variational bound
    """
    score = 0
    for d in range(D):

        mask = document_word_matrix[d] > 0
        N_count = document_word_matrix[d][mask]
        W_list = W_array[mask]

        phi_reduced = phi[d, W_list-1, :]
        log_phi_reduced = log_phi[d, W_list-1, :]
        log_theta = np.tile(sc.digamma(gamma[d, :]), (W_list.shape[0], 1)) - sc.digamma(np.sum(gamma[d, :]))
        log_beta = sc.digamma(lambda_[:, W_list-1]).T - np.tile(sc.digamma(np.sum(lambda_, axis = 1)), (W_list.shape[0], 1))
        score += np.sum(N_count * np.sum(phi_reduced * (log_theta + log_beta - log_phi_reduced), axis = 1))

    score -= np.sum(sc.gammaln(np.sum(gamma, axis = 1)))
    score += np.sum((alpha - gamma) * (sc.digamma(gamma) - sc.digamma(gamma.sum(axis = 1).reshape(-1, 1))))
    score += np.sum(sc.gammaln(gamma))
    score += -np.sum(sc.gammaln(lambda_.sum(axis=1)))
    score += np.sum((eta - lambda_) * (sc.digamma(lambda_) - sc.digamma(lambda_.sum(axis=1)).reshape(-1, 1)))
    score += np.sum(sc.gammaln(lambda_))
    score += D*sc.gammaln(K*alpha)
    score -= D*K*sc.gammaln(alpha)
    score += sc.gammaln(W*eta)
    score -= W*sc.gammaln(eta)
    return score

def log_pba_approx(X, doc_topic_distr, lambda_, D, W, phi, alpha, eta):
        """
        Log pba with variational bound
        """
        # sample = document
        # features = word
        # components = topic K

        def _loglikelihood(prior, distr, dirichlet_distr, size):
            # calculate log-likelihood
            score = np.sum((prior - distr) * dirichlet_distr)
            score += np.sum(gammaln(distr) - gammaln(prior))
            score += np.sum(gammaln(prior * size) - gammaln(np.sum(distr, 1)))
            return score

        n_samples, n_components = doc_topic_distr.shape
        n_features = W
        score = 0

        components_ = lambda_ #np.sum(phi, axis = 0).T # K*W
        dirichlet_doc_topic = _dirichlet_expectation_2d(doc_topic_distr) # D*K
        dirichlet_component_ = _dirichlet_expectation_2d(components_) # K*W
        doc_topic_prior = alpha
        topic_word_prior = eta

        # E[log p(docs | theta, beta)]
        for idx_d in range(0, n_samples):

            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]
            temp = (dirichlet_doc_topic[idx_d, :, np.newaxis]
                    + dirichlet_component_[:, ids])
            norm_phi = logsumexp(temp, axis=0)
            score += np.dot(cnts, norm_phi)

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        score += _loglikelihood(doc_topic_prior, doc_topic_distr,
                                dirichlet_doc_topic, n_components)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score += _loglikelihood(topic_word_prior, components_,
                                dirichlet_component_, n_features)

        return score

def LDA_VB(document_word_matrix, document_word_matrix_test, n_iter_doc, n_iter, K, alpha, eta):
    """
    Fit a LDA model with EM algorithm
    Return the evolution of variational bound, log pba approximative
    and the parameters, phi, gamma and lambda
    """
    D = document_word_matrix.shape[0]
    W = document_word_matrix.shape[1]
    W_array = np.arange(0, W, 1) + 1

    bound_list = []
    bound_train = []
    bound_test = []
    #D = corpus['document'].max()
    #W = vocab.shape[0]
    phi_n = np.zeros((D, W, K))
    phi = np.zeros((D, W, K))
    log_phi = np.zeros((D, W, K))
    gamma = np.ones((D, K))
    lambda_ = np.random.rand(K, W)

    #N_w_k_j_test, N_k_j_test, N_w_k_test, N_k_test = compute_occurence_matrix(document_word_matrix_test, W_array, W, K, D)
    for iter in range(n_iter_doc):
        t_beginning = time.time()
        for d in range(D):

            # keeping only the considered document and useful infos
            mask = document_word_matrix[d] > 0
            N_count = document_word_matrix[d][mask]
            W_list = W_array[mask]

            #doc = corpus[corpus.document == d+1]
            #W_list = np.array(doc['word'])
            last_gamma = gamma[d, :].copy()
            gamma[d, :] = np.ones(K)
            #N_count = np.array(doc['count'])

            for i in range(n_iter):
                t0 = time.time()

                L_test = np.tile(sc.digamma(gamma[d, :]), (W_list.shape[0], 1)) + sc.digamma(lambda_[:, W_list-1]).T - sc.digamma(np.sum(lambda_[:, W_list -1], axis = 0)).reshape(-1, 1)

                phi_temp = sc.softmax(L_test, axis = 1)
                log_phi_temp = sc.log_softmax(L_test, axis = 1)

                # update gamma
                last_gamma = gamma[d, :].copy()
                gamma[d, :] = alpha + np.sum(phi_temp * N_count.reshape(-1, 1), axis =0)

                # check inner convergence
                if np.mean(np.abs(gamma[d, :] - last_gamma)) < 0.001:
                  break

            # update phi
            phi_n[d, W_list-1, :] = phi_temp * N_count.reshape(-1, 1)
            phi[d, W_list-1, :] = phi_temp
            log_phi[d, W_list-1, :] = log_phi_temp

        # M-step, update lambda_
        lambda_ = eta + np.sum(phi_n, axis = 0).T
        t_end = time.time()
        b = compute_variational_bound(document_word_matrix, phi, log_phi, gamma, lambda_, alpha, eta, K, W, D, W_array)
        b_train = log_pba_approx(document_word_matrix, gamma, lambda_, D, W, phi, alpha, eta)

        # Compute Test pbas
        #b_test = compute_pba_test(document_word_matrix_test, W_array, phi, D, K, W, alpha, eta)
        b_test = log_pba_approx(document_word_matrix_test, gamma, lambda_, D, W, phi, alpha, eta)

        t_end_bound = time.time()
        print('iter n°{} - iter time = {} - bound time = {} - bound value = {} - bound_test {}'.format(iter, t_end - t_beginning, t_end_bound - t_end, b, b_test))
        VB_bound.append(b)
        log_pba_train.append(b_test)
        log_pba_train.append(b_train)

    return VB_bound, log_pba_train, log_pba_test, phi, gamma, lambda_
