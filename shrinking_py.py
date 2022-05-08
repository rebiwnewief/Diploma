# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 23:58:36 2022

@author: unque
"""

import os
import glob
import json
import numpy as np
import pandas as pd
#import pulp
#from precoli.learn.Diploma.munkres import Munkres, DISALLOWED
from munkres import Munkres, DISALLOWED
import copy
#from precoli.learn.Diploma.branchNbound import BranchNbound
import pulp
from branchNbound import BranchNbound
import time

from simple_tsp.main import precomute_route
from simple_tsp.prep import load_problem, load_solution
from simple_tsp.helpers import route2cost

MUNKRES_FILLER = DISALLOWED
PULP_FILLER = 100000000

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def distance(df, t1, t2):
    return np.sqrt(np.power(df[t1] - df[t2], 2).sum()).astype(np.int32)

def create_matrix(name: str, filler):
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
    
    np.fill_diagonal(C, filler)  
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
    return m.C, indexes, dist_sum

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

def compute_new_matrix(Sets0, C, filler):
    n1 = len(Sets0)
    C1 = np.zeros((n1, n1), dtype=int).tolist()
    for i in range(n1):
        for j in range(n1):
            if i == j:
                C1[i][j] = filler
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

def block_route(C):
    pass

def create_test(filler):
    optimum = 216
    df = np.zeros((10, 10), dtype = int)
    df[1, 0] = 32
    df[2, :2] = np.array([41, 22])
    df[3, :3] = np.array([22, 50, 63])
    df[4, :4] = np.array([20, 42, 41, 36])
    df[5, :5] = np.array([57, 51, 30, 78, 45])
    df[6, :6] = np.array([54, 61, 45, 72, 36, 22])
    df[7, :7] = np.array([32, 20, 10, 54, 32, 32, 41])
    df[8, :8] = np.array([22, 54, 60, 20, 22, 67, 57, 50])
    df[9, :9] = np.array([45, 51, 36, 64, 28, 20, 10, 32, 50])
    
    df = df + df.T
    np.fill_diagonal(df, filler)
    return df, optimum
    #df = df.tolist()

def warm_start(problem, lk_route):
    for i in range(len(lk_route) - 1):
        problem.x[problem.N * lk_route[i] + lk_route[i + 1]].setInitialValue(1)
    # for k, v in solution.items():
    #     x[k].setInitialValue(v)


if __name__ == "__main__":
    m = Munkres()
    dist_sum = 0
    step_count = 5
    pulp_problem = True
    test_problem = create_test(PULP_FILLER)
    solved = []
    TSP_problem = [("ALL_tsp/eil76.tsp/eil76.tsp", 538), ("ALL_tsp/d493.tsp/d493.tsp", 35002),
                   ("ALL_tsp/a280.tsp/a280.tsp", 2579), ("ALL_tsp/d657.tsp/d657.tsp", 48912),
                   ("ALL_tsp/bier127.tsp/bier127.tsp", 118282), ("ALL_tsp/d1291.tsp/d1291.tsp", 50801),
                   ("ALL_tsp/nrw1379.tsp/nrw1379.tsp", 56638)]
    
    TSP_all_problem = open('simple_tsp/valid_files.txt', 'r').readlines()
    with open("simple_tsp/optimal.json", "r") as js:
        solutions = json.load(js)
    #TSP_all_problem = #glob.glob("simple_tsp/data/tsplib/*.tsp")
    #for problem, optimal in TSP_problem[:]:
    for problem in TSP_all_problem:
        problem = problem.strip()
        problem_name = problem.split("/")[-1][:-4]
        print("current problem is %s" %(problem_name))
        #C = create_matrix(problem, PULP_FILLER) #  "precoli/learn/kommi/a280.tsp" : "../kommi/a280.tsp"
        C, nodes = load_problem(problem)
        
        if problem_name in solutions:
            optimal = solutions[problem_name]
        else:
            optimal = -1
            
        np.fill_diagonal(C, PULP_FILLER)
        
        lk_route, lk_bestcost = precomute_route(problem)
        print("lk_route", lk_route)
        
        C1 = np.array(copy.deepcopy(C))
        # C1, optimum = test_problem
        dist_sum = 0
        
        for step in range(step_count):
            print("current step %d" %step)
            print("problem dimmension", C1.shape)
            
            if C1.shape == (1, 1):
                break
            #  pulp solution
            BnB = BranchNbound(C1)      
            if step == 0:
                if os.path.exists("saved_model/%s.json" %problem_name):
                    loaded_vars, loaded_model = pulp.LpProblem.from_json("saved_model/%s.json" %problem_name)
                    BnB.prob = loaded_vars
                    BnB.solver = loaded_model
                else:
                    BnB.create_task(block_route = False, filler = PULP_FILLER)
                    warm_start(BnB, lk_route)
                    status = BnB.prob.solve(BnB.solver)
                    print("stattus", status)
                    BnB.prob.to_json("saved_model/%s.json" %problem_name , cls=NpEncoder)
            else:
                BnB.create_task(block_route = True, filler = PULP_FILLER, max_block=2*nodes)
                status = BnB.prob.solve(BnB.solver)
                print("stattus", status)
            #print(pulp.LpStatus[self.prob.status])
            cur_sum, Sets0 = BnB.getSets()
            #print(BnB.C.reshape(C1.shape))
            print("cur_sum %d" %cur_sum)
            dist_sum += cur_sum
            
            if len(C1) <= 2:
                break
            
            if pulp_problem:
                #print(Sets0)
                C1 = compute_new_matrix(Sets0, BnB.C.reshape(BnB.N, BnB.N), filler = PULP_FILLER)
                C1 = compute_compression(C1, len(Sets0))
                C1 = np.array(C1)
                #print(C1)
                continue
            
            
            # этап стягивания графа
            C_, indexes, dist_sum = compute_optimal(m, C1, dist_sum)
            Sets0 = getSets(C1, indexes)
            C1 = compute_new_matrix(Sets0, C_)
            C1 = compute_compression(C1, len(Sets0))
            
            # C_, indexes, dist_sum = compute_optimal(m, C1, dist_sum)
            # Sets0 = getSets(C1, indexes)
            # C1 = compute_new_matrix(Sets0, C_)
            # C1 = compute_compression(C1, len(Sets0))
            
        
        print("current LB", dist_sum)
        if not pulp_problem:
            C1 = np.array(C1)
            # np.fill_diagonal(C1, 100000000)
            BnB = BranchNbound(C1)
            print(np.array(C1).shape)
            start = time.time()
            BnB.solve()
            end = time.time()
            print("solve time", end - start)
        solved.append(dist_sum)
        print("optimal solution is", dist_sum + BnB.S)
        print("global optimum is %d" %optimal)
        