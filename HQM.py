import numpy as np
import random
from reward_HQM import getReward1
import copy
from tqdm import tqdm
import time

def updateP(Q):
    sx = np.sum(Q, axis=1).reshape(-1, 1)
    qs = np.tile(sx, Q.shape[1])
    P = Q / qs
    return P

def getNeighborhood(x, y, m, n, nvars, r):
    km = round(m*r)
    kn = round(n*r)
    xt = np.random.randint(-km, km, nvars)
    x_new = x + xt
    x_new = np.minimum(x_new, m-1)
    x_new = np.maximum(x_new, 0)

    temp = np.random.permutation(n).reshape(-1)
    sel1 = temp[0:kn]
    sel2 = temp[kn:2*kn]
    sel = [[sel1[i], sel2[i]]for i in range(len(sel1))]
    y_new = copy.copy(y)
    for se in sel:
        i1 = se[0]
        i2 = se[1]
        c1 = y_new[i1]
        temp = c1
        y_new[i1] = y_new[i2]
        y_new[i2] = temp

    return x_new, y_new

def Qlearning(ncar, npoints, maxspeed, capacity, gene, popsize, policy, data):

    np.random.seed(int(time.time() % 2**32))
    m = ncar
    n = npoints
    kr = np.exp(0.2*np.arange(0, popsize)) / np.max(np.exp(0.2*np.arange(0, popsize))) #reward coefficient
    gama = 0.9 #discount rate
    Qx = [] #Q-Matrix for allocation
    Qy = [] #Q-Matrix for sorting
    Px = [] #Possibility matrix
    Py = [] #Possibility matrix
    '''Initialise Q-Matrix and Possibility Matrix'''
    for i in range(n):
        if i == 0:
            Qx.append(1e-4*np.ones((1, m)))
            Qy.append(1e-4*np.ones((1, n)))
            Px.append(np.zeros((1, m)))
            Py.append(np.zeros((1, n)))
        else:
            Qx.append(1e-4*np.ones((m, m)))
            Qy.append(1e-4*np.ones((n, n)))
            Px.append(np.zeros((m, m)))
            Py.append(np.zeros((n, n)))
    '''Initialise agent'''
    chrom = []
    reward = []
    for i in range(popsize):
        x = np.random.randint(0, m, n)
        y = np.random.permutation(n)
        chrom.append([x, y])
        reward.append(getReward1([x, y], ncar, data, maxspeed, capacity, policy, 0))
    '''record best solution'''
    bestfit = []
    bestx = []
    index = np.argmax(reward)
    bestfit.append(reward[index])
    bestx = chrom[index]
    initial_result_q = bestfit[0]
    '''convergence judegement'''
    Qnorm = []
    maxcount = 100
    error = 1e-8
    count = 0
    Qnorm.append(0)
    '''start iteration'''
    for ks in tqdm(range(gene)):
        '''update learning rate'''
        afa = 0.9*np.exp(-ks / gene) #动态学习率
        '''sorting'''
        index = np.argsort(np.array(reward)).reshape(-1) #由小到大排序
        reward = [reward[i] for i in index]
        chrom = [chrom[i] for i in index]
        '''record'''
        bestfit.append(reward[-1])
        bestx = chrom[-1]
        '''update possibility matrix'''
        for i in range(n):
            Px[i] = updateP(Qx[i])
            Py[i] = updateP(Qy[i])
        '''global search based on Q-Matrix'''
        for i in range(popsize):
            x = []
            y = []
            pre_x = 0
            pre_y = 0
            list_y =[]
            for j in range(n):
                temp = []
                temp = copy.copy(Px[j][pre_x])
                spx = np.concatenate(([0], np.cumsum([temp])))
                temp = []
                temp = copy.copy(Py[j][pre_y])
                if j > 0:
                    for h in list_y:
                        temp[h] = 0
                    temp = temp / (np.sum(temp) + 1e-9)
                spy = np.concatenate(([0], np.cumsum([temp])))
                sel_x = np.searchsorted(spx, np.random.random()) - 1
                sel_y = np.searchsorted(spy, np.random.random()) - 1
                x.append(sel_x)
                y.append(sel_y)
                pre_x = sel_x
                pre_y = sel_y
                list_y.append(sel_y)
            x = np.array(x)
            y = np.array(y)
            zz = np.unique(y)
            new_reward = getReward1([x, y], ncar, data, maxspeed, capacity, policy, flag=0)
            if new_reward > reward[i]:
                reward[i] = new_reward
                chrom[i] = [x, y]
        '''neighbor search'''
        for i in range(popsize):
            x, y = chrom[i][0], chrom[i][1]
            x_new, y_new = getNeighborhood(x, y, m, n, nvars=n, r=0.2)
            new_reward = getReward1([x_new, y_new], ncar, data, maxspeed, capacity, policy, flag=0)
            if new_reward > reward[i]:
                reward[i] = new_reward
                chrom[i] = [x_new, y_new]
        '''update Q-Matrix'''
        Qx_old = copy.copy(Qx)
        Qy_old = copy.copy(Qy)
        re = kr*np.array(reward)
        for i in range(popsize):
            x = chrom[i][0]
            y = chrom[i][1]
            pre_x = 0
            pre_y = 0
            for j in range(n):
                cur_x = x[j]
                cur_y = y[j]
                qx = copy.copy(Qx[j])
                qy = copy.copy(Qy[j])
                maxqx = np.max(qx[pre_x, :])
                maxqy = np.max(qy[pre_y, :])
                qx[pre_x, cur_x] = (1 - afa) * qx[pre_x, cur_x] + afa * (re[i] + gama * maxqx)
                qy[pre_y, cur_y] = (1 - afa) * qy[pre_y, cur_y] + afa * (re[i] + gama * maxqy)
                Qx[j] = copy.copy(qx)
                Qy[j] = copy.copy(qy)
        '''Convergence judgement'''
        snorm = 0
        for i in range(n):
            qx = Qx[i]
            qy = Qy[i]
            qx_old = Qx_old[i]
            qy_old = Qy_old[i]
            snorm += np.linalg.norm(qx-qx_old) + np.linalg.norm(qy-qy_old)
        Qnorm.append(snorm)
        if np.abs(Qnorm[-1] - Qnorm[-2]) <= error:
            count += 1
        else:
            count = 0
        if count >= maxcount:
            print(count)
            break
    '''output'''
    Qnorm[0] = Qnorm[1]
    return bestfit, bestx, Qnorm, initial_result_q




    
    
    

















