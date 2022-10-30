import numpy as np
from reward_GA import getReward
import copy
from tqdm import tqdm
import time


def select(chrom, fit):
    chrom = copy.copy(chrom)
    fit = copy.copy(np.array(fit))
    popsize = len(chrom)

    index = np.argsort(fit)
    newchrom = []
    for i in index:
        newchrom.append(chrom[i])
    chrom = newchrom
    fit = fit[index]

    eli = np.int(np.fix(0.05 * popsize))
    newchrom = copy.copy(chrom)
    for i in range(eli):
        newchrom[-(i + 1)] = chrom[i]
    chrom = newchrom
    fit[-np.arange(1, eli + 1)] = fit[0:eli]

    return chrom, fit

#crossover
def corros(c1, c2, m, n):
    x1, y1 = c1[0], c1[1]
    x2, y2 = c2[0], c2[1]

    # operation for allocation scheme
    sel = np.argwhere(np.random.rand(n) <= 0.5).reshape(-1)
    t1 = x1 + 0.3*(x1 - x2)
    t2 = x2 + 0.3*(x2 - x1)
    t1 = np.fix(np.clip(t1, np.random.choice(m), m-1)).astype(int)
    t2 = np.fix(np.clip(t2, np.random.choice(m), m-1)).astype(int)
    x1[sel] = t1[sel]
    x2[sel] = t2[sel]

    #operation for task execution sequence
    w = np.random.randint(np.round(0.1*n), np.round(0.9*n))
    p = np.random.randint(0, n-w-1)
    for i in range(w):
        h = p + i
        h1 = np.argwhere(y1 == y2[h]).reshape(-1)
        h2 = np.argwhere(y2 == y1[h]).reshape(-1)
        temp = y1[h]
        y1[h] = y2[h]
        y2[h] = temp
        temp = y1[h1]
        y1[h1] = y2[h2]
        y2[h2] = temp

    new_c1 = [x1, y1]
    new_c2 = [x2, y2]

    return new_c1, new_c2

#mutation
def mutate(c, m, n):
    x, y = c[0], c[1]

    #operation for allocation scheme
    sel = np.random.randint(0, n, size=np.fix(0.05*n).astype(int))
    x[sel] = np.random.randint(0, m, size=np.fix(0.05*n).astype(int))

    #operation for task execution sequence 
    sel = np.random.randint(0, n, size=np.fix(0.05*n).astype(int))
    for i in range(0, np.size(sel)-1, 2):
        h1, h2 = sel[i], sel[i+1]
        temp = y[h1]
        y[h1] = y[h2]
        y[h2] = temp

    new_c = [x, y]

    return new_c

def ga(ncar, npoints, gene, popsize, maxspeed, capacity, policy, data):

    np.random.seed(int(time.time() % 2 ** 32))
    m = ncar
    n = npoints
    fc = 1
    pc = 1
    '''initialisation'''
    chrom = []
    reward = []
    for i in range(popsize):
        x = np.random.randint(0, m, n)
        y = np.random.permutation(n)
        chrom.append([x, y])
        reward.append(getReward([x, y], ncar, data, maxspeed, capacity, policy, 0))
    '''record the best result'''
    bestfit = []
    bestx = []
    min_index = np.argmin(reward)
    bestfit.append(reward[min_index])
    bestx = chrom[min_index]
    '''covergence setting'''
    maxcount = 100
    error = 1e-8
    count = 0
    '''start iteration'''
    for ks in tqdm(range(gene)):
        '''sorting'''
        index = np.argsort(np.array(reward)).reshape(-1) #sorted from low to high
        reward = [reward[i] for i in index]
        chrom = [chrom[i] for i in index]
        '''record'''
        bestfit.append(reward[0])
        bestx = chrom[0]
        '''selection'''
        chrom, reward = select(chrom, reward)
        '''crossover'''
        for i in range(0, popsize-1, 2):
            if np.random.random() > pc:  # possibility for crossover
                continue
            c1, c2 = chrom[i], chrom[i+1]
            f1, f2 = reward[i], reward[i+1]
            new_c1, new_c2 = corros(c1, c2, m, n)
            new_f1 = getReward(new_c1, ncar, data, maxspeed, capacity, policy, flag=0)
            new_f2 = getReward(new_c2, ncar, data, maxspeed, capacity, policy, flag=0)
            if new_f1 < f1:
                chrom[i] = new_c1
                reward[i] = new_f1
            if new_f2 < f2:
                chrom[i+1] = new_c2
                reward[i+1] = new_f2
        '''mutation'''
        for i in range(popsize):
            if np.random.random() > fc:  # possibility for mutation
                continue
            c = chrom[i]
            f = reward[i]
            new_c = mutate(c, m, n)
            new_f = getReward(new_c, ncar, data, maxspeed, capacity, policy, flag=0)
            if new_f < f:
                chrom[i] = new_c
                reward[i] = new_f
        '''covergence judgement'''
        if np.abs(bestfit[-1] - bestfit[-2]) <= error:
            count += 1
        else:
            count = 0
        if count >= maxcount:
            break
    for i in range(0, gene - len(bestfit)):
        bestfit.append(bestfit[-1])
   
    '''output'''
    return bestfit, bestx





