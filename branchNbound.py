# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 01:17:40 2022

@author: unque
"""
import pandas as pd
import numpy as np
import time
import pulp
#from munkres import Munkres, DISALLOWED
#from shrinking_py import getSets, compute_new_matrix, compute_compression

class BranchNbound():
    
    def __init__(self, df):
        self.df = df
        self.N = self.df.shape[0]
        self.x = np.zeros(self.N*self.N, dtype=pulp.LpVariable)
        self.C = np.zeros(self.N*self.N, dtype = np.int32)
        self.prob = pulp.LpProblem("TSPminimize", pulp.LpMinimize)
        self.solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True, threads=4, timeLimit=300)
        self.S = 0
        
    def distance(self, t1, t2):
        return np.sqrt(np.power(self.df[t1] - self.df[t2], 2).sum()).astype(np.int32)


    def create_task(self, block_route: bool, filler: int, max_block: int = 5000):
        # self.N = self.df.shape[1]
        # self.x = np.zeros(self.N*self.N, dtype=pulp.LpVariable)
        # self.C = np.zeros(self.N*self.N)
        
        for t1 in range(self.N):
            for t2 in range(self.N):
                self.x[self.N*t1+t2] = pulp.LpVariable("x_"+ str(t1)+"_"+str(t2), 0, 1, cat='Binary')
                dist = self.df[t1, t2] # self.distance(t1, t2)
                self.C[self.N*t1+t2] = dist
        np.fill_diagonal(self.C.reshape(self.N, self.N), filler)
        # self.prob = pulp.LpProblem("TSPminimize", pulp.LpMinimize)
        self.prob += pulp.lpDot(self.x, self.C)
        for t1 in range(self.N):
            self.prob += pulp.lpSum(self.x[self.N*t1+t2] for t2 in range(self.N)) == 1
            self.prob += pulp.lpSum(self.x[self.N*t2+t1] for t2 in range(self.N)) == 1
        if block_route:
            self.block_route(max_block)
        
        # status = self.prob.solve(self.solver)
        # print(pulp.LpStatus[self.prob.status])
        # return self.N, self.prob, self.x, self.C
        
    #create sets of ways
    def getSets(self):
        subsets = []
        isub = 0
        indexes = np.zeros(self.N, dtype=int)
        S = 0
        for t1 in range(self.N):
            for t2 in range(self.N):
                #if edge in the row / active 
                if self.x[self.N*t1+t2].value() == 1:
                    #set this row as second city number; from t1 to t2
                    indexes[t1] = t2
    
        #while dont take all cities
        count = 0
        while count < self.N:
            for i in range(self.N):
                if indexes[i] >= 0:
                    break
            subsets.append([])
            while indexes[i] >= 0:
                subsets[isub].append(i)
                iold = i
                i = indexes[i]
                if i >= 0:
                    S += self.C[self.N*iold+i]
                else:
                    S += self.C[self.N*i+subsets[isub][0]]
                indexes[iold] = -1
                count += 1
            isub += 1
        self.update_costmatrix()
        return S, subsets
            
    def getMinSet(self, Sets):
        minSet = []
        L = 99999999
        for one_set in Sets:
            if len(one_set) < L:
                minSet = one_set
                L = len(one_set)
        return minSet

    def block_route(self, max_block: int):
        double_dict = {}
        for t1 in range(self.N):
            for t2 in range(t1 + 1, self.N):
                double_dict[(t1, t2)] = np.sum((self.C[self.N*t1 + t2], self.C[self.N*t2 + t1]))
        #сортировка по стоимости маршрута
        double_dict = dict(sorted(double_dict.items(), key=lambda item: item[1]))
        max_block = min(max_block, len(double_dict))
        #for j in range(0, len(double_dict), step):
        print("count doubles %d" %len(double_dict))
        n_items = list(double_dict.keys())[:max_block]
        for key in n_items:
            t1, t2 = key
            self.prob.add(pulp.lpSum((self.x[self.N*t1+t2], self.x[self.N*t2 + t1])) <= 1, \
                     "Pair%d_%d"%(t1, t2))
        # for t1 in range(self.N):
        #     for t2 in range(t1 + 1, self.N):
        #         self.prob.add(pulp.lpSum((self.x[self.N*t1 + t2], self.x[self.N*t2 + t1]))  <= 1, \
        #                       "Pair%d_%d"%(t1, t2))
                
    def update_costmatrix(self):
        for t1 in range(self.N):
            for t2 in range(self.N):
                #if edge in the row / active 
                if self.x[self.N*t1+t2].value() == 1:
                    self.C[self.N*t1:min(self.N*(t1+1), self.C.size)] -= self.C[self.N*t1+t2]
        self.C = np.clip(self.C, 0, None)
                    
    
    def solve_subtask(self):
        #solution TCP
        now = time.time()
        S, Sets = self.getSets()
        i = 0
        count_same = 0
        status = 0
        while len(Sets) > 1:
            print(len(Sets), S)
            for set_one in Sets:
                L = len(set_one)
                self.prob.add(pulp.lpSum(self.x[self.N*set_one[i]+set_one[(i+1)%L]] for i in range(L)) <= L-1, "Name"+str(i))
                i += 1
            status = self.prob.solve(self.solver)
            old_S = S
            S, Sets = self.getSets()
            if old_S == S:
                count_same = count_same + 1
                if count_same == 5:
                    break
        print("solve time", time.time() - now)
        print(len(Sets), S)
        self.S = S
        
    def solve(self, block_route):
        self.create_task(block_route)
        if block_route:
            _, Sets0 = self.getSets()
            C1 = compute_new_matrix(Sets0, self.C)
            C1 = compute_compression(C1, len(Sets0))
            return
        self.solve_subtask()



if __name__ == "__main__":
    df = pd.read_csv("../kommi/a280.tsp", sep = '\s+', header=None, skiprows=6, engine='python', index_col = 0, skipfooter=1)
    #df = pd.read_csv("../kommi/ftv33.atsp", header=None, skiprows=2, engine='python')
    df = df.to_numpy()
    #np.fill_diagonal(df, DISALLOWED)
    #df = df.values
    BnB = BranchNbound(df)
    BnB.solve(block_route = True)
    