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

@jit(nopython=True)
def choice_jit(a, p):
    """
    Draw an element of an array a with probability p
    """
    u = np.random.random()
    for i in range(p.shape[0]):
        if u <= p[i]:
            return i

@jit(nopython=True)
def compute_log_pba_jit(document_word_matrix, phi, theta, z, D, K, W, W_array):
    """
    Compute log pba for Gibbs sampling
    """
    score = 0.
    # P(W|Z)
    for d in range(D):
        mask = document_word_matrix[d] > 0
        N_count = document_word_matrix[d][mask]
        W_list = W_array[mask]

        # observed z
        for w_idx in range(W_list.shape[0]):
            # given topic
            z_i_j = z[W_list[w_idx]-1, d]
            # add score
            score += np.float64(N_count[w_idx]) * (np.log(phi[W_list[w_idx]-1, z_i_j]))

    # P(Z)
    for d in range(D):
        mask = document_word_matrix[d] > 0
        W_list = W_array[mask]
        for w_idx in range(W_list.shape[0]):
            score += np.float64(np.log(theta[z[W_list[w_idx] - 1, d], d]))

    return score

@jit(nopython=True)
def collapsed_gibbs_sampling_jit(document_word_matrix, n_iter, K, alpha, eta):
    """
    Return the evolution of the log pba with collapsed Gibbs sampling
    """

    D = document_word_matrix.shape[0]
    W = document_word_matrix.shape[1]
    W_array = np.arange(0, W, 1) + 1
    bound_list = []
    #bound_list2 = []

    phi = np.zeros((D, W, K))
    K_array = np.arange(0, K, 1)

    #z = np.random.randint(low = 0, high = K, size=(W, D))
    z = np.zeros((W, D)).astype(np.int32)
    for i in range(W):
        for j in range(D):
            z[i, j] = np.random.randint(low=0, high=K)

    # Compute array N
    N_w_k_j = np.zeros((W, K, D))
    for d in range(D):
        mask = document_word_matrix[d] > 0
        N_count = document_word_matrix[d][mask]
        W_list = W_array[mask]

        for w_idx in range(W_list.shape[0]):
            k_value = z[W_list[w_idx]-1, d] == K_array
            N_w_k_j[W_list[w_idx]-1, :, d] = N_count[w_idx] * k_value

    N_k_j = np.sum(N_w_k_j, axis = 0)
    N_w_k = np.sum(N_w_k_j, axis = 2)
    N_k = np.sum(np.sum(N_w_k_j, axis = 0), axis = 1)

    for i in range(n_iter):
    #t_beginning = time.time()
        for d in range(D):
            mask = document_word_matrix[d] > 0
            N_count = document_word_matrix[d][mask]
            W_list = W_array[mask]

            for w_idx in range(W_list.shape[0]):
                P = np.zeros(K)
                s=0
                for k in range(0, K):

                    alpha_k_j = (N_k_j[k, d] - N_w_k_j[W_list[w_idx]-1, k, d])
                    alpha_k_j = np.float64(alpha_k_j)
                    beta_w_k = (N_w_k[W_list[w_idx]-1, k] - N_w_k_j[W_list[w_idx]-1, k, d] + eta) / (N_k[k] - N_w_k_j[W_list[w_idx]-1, k, d] + W * eta)
                    s += np.float64(alpha_k_j) * np.float64(beta_w_k)
                    P[k] = s

                P = P / P[K-1]
                #z_ij = np.random.choice(a=K_array, p=P)
                z_ij = choice_jit(K_array, P)

                # update N
                new_value = N_count[w_idx] * (z_ij == K_array)  # N_w_k_j[W_list[w_idx]-1, :, d]

                N_k_j[:, d] = N_k_j[:, d] - N_w_k_j[W_list[w_idx]-1, :, d] + new_value
                N_w_k[W_list[w_idx]-1, :] = N_w_k[W_list[w_idx]-1, :] - N_w_k_j[W_list[w_idx]-1, :, d] + new_value
                N_k = N_k - N_w_k_j[W_list[w_idx]-1, :, d] + new_value

                N_w_k_j[W_list[w_idx]-1, :, d] = new_value

                z[W_list[w_idx]-1, d] = z_ij

        phi = (N_w_k + eta) / (np.sum(N_w_k, axis = 0).reshape(1, -1) + W * eta)  # W * K
        theta = (N_k_j + alpha) / (np.sum(N_k_j, axis =0).reshape(1, -1) + K * alpha) # K * D

        #t_end = time.time()
        b = compute_log_pba_jit(document_word_matrix, phi, theta, z, D, K, W, W_array)
        bound_list.append(b)

    return bound_list, z


# @jit(nopython=True)
# def compute_bound(document_word_matrix, phi, theta, D, K, W, W_array):
#   score = 0.
#
#   for d in range(D):
#     mask = document_word_matrix[d] > 0
#     N_count = document_word_matrix[d][mask]
#     W_list = W_array[mask]
#
#     for w_idx in range(W_list.shape[0]):
#       score += np.float64(np.log(np.sum(theta[:, d] * phi[W_list[w_idx]-1, :])))
#   return score
