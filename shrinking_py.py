# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 23:58:36 2022

@author: unque
"""

import numpy as np
import pandas as pd
import pulp
from precoli.learn.Diploma.munkres import Munkres, DISALLOWED
import copy



def distance(df, t1, t2):
    return np.sqrt(np.power(df[t1] - df[t2], 2).sum()).astype(np.int32)

def create_matrix(name: str):
    df = pd.read_csv(name, sep = '\s+', header=None, skiprows=6, engine='python', index_col = 0, skipfooter=1)
    df = df.to_numpy()
    
    N = df.shape[0]
    C = np.zeros((N, N), dtype = np.int32)
    
    #save pair sities as variable: the vector (N * N); save a distances as variable: the vector (N * N)
    for t1 in range(N):
        for t2 in range(t1, N):
            #distance 
            dist = distance(df, t1, t2)
            C[t1, t2] = dist
            C[t2, t1] = dist
    
    np.fill_diagonal(C, DISALLOWED)  
    C = C.tolist()
    return C

def compute_optimal(m, C, dist_sum):
    indexes = m.compute(C)
    total_cost = 0
    for r, c in indexes:
        x = C[r][c]
        total_cost += x
        #print(('(%d, %d) -> %d' % (r, c, x)))
    dist_sum += total_cost
    print(('lowest cost=%d' % total_cost))
    print('total distance=%d' % dist_sum)
    return indexes, dist_sum

def getSets(df, indexes):
    subsets = []
    isub = 0
    N = len(df)
    idx = np.zeros(N, dtype=int)
    for r, c in indexes:
        idx[r] = c

    count = 0
    while count < N:
        for i in range(N):
            if idx[i] >= 0:
                break
        subsets.append([])
        while idx[i] >= 0:
            subsets[isub].append(i)
            iold = i
            i = idx[i]
            idx[iold] = -1
            count += 1
        isub += 1
    return subsets

def min_distance(matrix_prev, loop_1, loop_2):
    min_dist = matrix_prev[loop_1[0]][loop_2[0]]
    for vertex_1 in loop_1:
        for vertex_2 in loop_2:
            min_dist = min(min_dist, matrix_prev[vertex_1][vertex_2])
    return min_dist

def compute_new_matrix(Sets0, C):
    n1 = len(Sets0)
    C1 = np.zeros((n1, n1), dtype=int).tolist()
    for i in range(n1):
        for j in range(n1):
            if i == j:
                C1[i][j] = DISALLOWED
                continue
            C1[i][j] = min_distance(C, Sets0[i], Sets0[j])
            
    return C1

def mindist(dist_matrix, dim, i, j):
    cur_min = np.inf
    for k in range(dim):
        if k != j and k != i:
            cur_min = min(cur_min, dist_matrix[i][k] + dist_matrix[k][j])
    return cur_min

# вариант с компресссией за один проход
def compression(dist_matrix, dim):
    flag = False
    for i in range(dim):
        for j in range(dim):
            if i != j:
                cur_mindist = mindist(dist_matrix, dim, i, j)
                if dist_matrix[i][j] > cur_mindist:
                    dist_matrix[i][j] = cur_mindist
                    flag = True #, dist_matrix

    return flag, dist_matrix

def compute_compression(C1, n1):
    cmprs_status = True
    #before_cmprs = C1.copy()
    while (cmprs_status):
        print("compression", cmprs_status)
        cmprs_status, C1 = compression(C1, n1)
    return C1



if __name__ == "__main__":
    m = Munkres()
    dist_sum = 0
    C = create_matrix("precoli/learn/kommi/a280.tsp")
    C1 = copy.deepcopy(C)
    while (len(C1) > 2):
        indexes, dist_sum = compute_optimal(m, C1, dist_sum)
        Sets0 = getSets(C1, indexes)
        C1 = compute_new_matrix(Sets0, C1)
        C1 = compute_compression(C1, len(Sets0))
        
        