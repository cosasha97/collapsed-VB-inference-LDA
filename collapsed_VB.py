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

from vb import log_pba_approx

def LDA_collapsed(document_word_matrix, document_word_matrix_test, n_iter_doc, n_iter, K, alpha, eta):
    """
    Collapsed Variational Bayesian Inference for LDA.
    """
    np.random.seed(0)
    D = document_word_matrix.shape[0]
    W = document_word_matrix.shape[1]
    W_array = np.arange(0, W, 1)

    bound_list = []
    bound_test = []

    phi_n = np.zeros((D, W, K))
    phi = np.random.rand(D, W, K)
    for d in range(D):
        phi[d,:,:] = phi[d,:,:]/phi[d,:,:].sum(axis=1)[:,None]
    log_phi = np.zeros((D, W, K))
    gamma = np.ones((D, K))
    lambda_ = np.random.rand(K, W)

    norm_train = document_word_matrix_train.sum()
    norm_test = document_word_matrix_test.sum()

    for iter in range(n_iter_doc):
        t_beginning = time.time()
        for d in range(D):
            # if d%10 == 0:
            #   print(d)

            # keeping only the considered document and useful infos
            mask = document_word_matrix[d] > 0
            # N_count = document_word_matrix[d][mask]
            W_list = W_array[mask]
            # useful matrices
            M = document_word_matrix[d,W_list] # size W

            last_gamma = gamma[d, :].copy()
            gamma[d, :] = np.ones(K)

            for i in range(n_iter):
                # t0 = time.time()
                # if i%10 == 0:
                #   print(i)
                # phi reduced to the right number of words
                # phi_temp = np.zeros((W_list.shape[0], K))
                # log_phi_temp = np.zeros((W_list.shape[0], K))

                # update phi

                # useful matrices
                phi_list = phi[d,W_list,:] # size W*K
                phi_list_n = M.reshape((-1,1)) * phi_list # size W*K
                PHI_N = np.tile(document_word_matrix[:, W_list, np.newaxis], (1, 1, 8)) * phi[:,W_list,:] # D*W*K
                PHI_N_0 = PHI_N.sum(axis=0)
                var = PHI_N * (1 - phi[:,W_list,:])
                var_0 = var.sum(axis=0)

                #### collapsed VB: formula 18 of original article ####
                # 1st term
                K_ = phi_list_n.sum(axis=0) # size K
                esp1 = K_[None,:] - phi_list_n
                term1 = alpha + esp1 # W*K

                # 2nd term
                esp2 = PHI_N_0 - PHI_N[d,:,:]
                term2 = eta + esp2

                # 3rd term
                # esp3 = np.sum(PHI_N,axis=(0,1)).reshape((1,-1)) - PHI_N[d,:,:]
                esp3 = np.sum(PHI_N_0,axis=0).reshape((1,-1)) - PHI_N[d,:,:]
                term3 = W * eta + esp3 # W*K

                # 4th term
                var1 = phi_list_n * (1-phi_list)
                var1 = (var1).sum(axis=0).reshape((1,-1)) - var1 # W*K
                term4 = -var1 / (2*term1**2)

                # 5th term
                var2 = var_0 - var[d,:,:]
                term5 = - var2 / (2*term2**2)

                # 6th term
                # var3 = var.sum(axis=(0,1)).reshape((1,-1)) - var[d,:,:]
                var3 = var_0.sum(axis=0).reshape((1,-1)) - var[d,:,:]
                term6 = var3 / ( 2 * term3**2 )

                log_phi_temp = np.log(term1) + np.log(term2) - np.log(term3) \
                        + term4 + term5 + term6 # W*K

                # print(np.mean(log_phi_temp))
                # log_phi_temp = np.random.rand(len(W_list),K)


                phi_temp = sc.softmax(log_phi_temp, axis = 1)
                # log_phi_temp = sc.log_softmax(L_test, axis = 1)

                # update gamma
                last_gamma = gamma[d, :].copy()
                gamma[d, :] = alpha + np.sum(phi_temp * document_word_matrix[d,W_list].reshape(-1, 1), axis =0)

                # check inner convergence
                if np.mean(np.abs(gamma[d, :] - last_gamma)) < 0.001:
                  break

                # update phi
                phi_n[d, W_list, :] = phi_temp * document_word_matrix[d,W_list].reshape(-1, 1)
                phi[d, W_list, :] = phi_temp
                log_phi[d, W_list, :] = sc.log_softmax(log_phi_temp,axis=1)
                # print(time.time() - t0)

        # M-step, update lambda_
        lambda_ = eta + np.sum(phi_n, axis = 0).T
        t_end = time.time()
        #b = compute_bound_3(document_word_matrix, phi, log_phi, gamma, lambda_, alpha, eta, K, W, D, W_array)

        # Compute Test pbas
        #b_test = compute_pba_test(document_word_matrix_test, W_array, phi, D, K, W, alpha, eta)
        b_test = _approx_bound(document_word_matrix_test, gamma, lambda_, D, W, phi, alpha, eta)
        b =  _approx_bound(document_word_matrix, gamma, lambda_, D, W, phi, alpha, eta)
        t_end_bound = time.time()
        print('iter n°{} - iter time = {} - bound time = {} - bound value = {} - bound_test {}'.format(iter, t_end - t_beginning, t_end_bound - t_end, b, b_test))
        bound_list.append(b)
        bound_test.append(b_test)

    return bound_list, bound_test, phi, gamma, lambda_
