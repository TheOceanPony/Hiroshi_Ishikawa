import copy
import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from cv2 import imread, resize, cvtColor, COLOR_BGR2GRAY
from time import time, ctime

@njit
def initial_labeling(img, C):
    
    k_init = np.zeros(img.size, dtype=np.int32)
    
    img = img.flatten()

    for p in range(0, img.size):
        best_k = None
        min_score = np.inf

        for k in range(0, C.size):
            score = np.abs(img[p] - C[k])

            if score < min_score:
                min_score = score
                best_k = k

        k_init[p] = C[int(best_k)]

    return k_init


@njit
def indicator(k, k_, scale):
    if k != k_:
        return scale
    else:
        return 0


@njit
def distance(k, k_, scale):
    return np.abs(k - k_)


@njit
def init_g(img, labeling, a_i, scale):

    h, w = img.shape[:2]
    img = img.flatten()

    g = np.zeros((h*w + 2, h*w + 2), dtype = np.int32)

    # unary penalty k_i
    for p in range(0, img.size):
        g[0, p+1] = np.abs( labeling[p] - img[p])
        #print(labeling[p], " | ", img[p])

    # unary penalty a_i
    for p in range(0, img.size):
        g[p+1, -1] = np.abs( a_i - img[p]) 

    for i in range(h):
        for j in range(w):
            index = j + i*w + 1
            # left
            if j > 0:
                left = j - 1 + i*w + 1
                g[index, left] = indicator( labeling[index], labeling[left], scale )
            # right
            if j < w - 1:
                right = j + 1 + i*w + 1
                g[index, right] = indicator( labeling[index], labeling[right], scale )
            # up
            if i > 0:
                up = j + (i-1)*w + 1
                g[index, up] = indicator( labeling[index], labeling[up], scale )
            # down
            if i < h - 1:
                down = j + (i+1)*w + 1
                g[index, down] = indicator( labeling[index], labeling[down], scale )

    return g


@njit
def iteration(FBin, WBin, NBin, Map, Vsize):
    
    # BFS
    Q = [0] # Queue that start with S
    Prev = np.full(Vsize, -1) # Prev node
    cursor = 0 # instead of popping elements of of the Q
    
    while True:
        v_curr = Q[cursor]
        for i, neighbour in enumerate(NBin[Map[v_curr,0]:Map[v_curr,1]]):
            #print(f"{V[v_curr]} | neighbour: {neighbour} | Weights: {Weights[v_curr][i]} | {F[v_curr]} ? {Weights[v_curr][i]}")

            if FBin[Map[v_curr,0]:Map[v_curr,1]][i] < WBin[Map[v_curr,0]:Map[v_curr,1]][i] and Prev[neighbour] == -1:
                Q.append(neighbour)
                Prev[neighbour] = v_curr
        cursor += 1
        if cursor >= len(Q):
            break
    
    
    # path reconstruction
    v_curr = Vsize - 1
    path = [v_curr]
    while True:
        v_prev = Prev[v_curr]
        v_curr = v_prev
        if v_curr == -1 : break
        path.append(v_curr)

    path = path[::-1]
    
    
    # path bottleneck
    bottleneck = np.inf
    for i in range(len(path) - 1):
        v_curr = path[i]
        v_next = path[i+1]

        ind = np.where( NBin[Map[v_curr,0]:Map[v_curr,1]] == v_next)[0][0]
        if WBin[Map[v_curr,0]:Map[v_curr,1]][ind] < bottleneck: 
            bottleneck = WBin[Map[v_curr,0]:Map[v_curr,1]][ind]
    
    
    # End break
    if bottleneck == np.inf : return FBin, True
    
    # path flow update
    for i in range(len(path) - 1):
        v_curr = path[i]
        v_next = path[i+1]

        ind = np.where( NBin[Map[v_curr,0]:Map[v_curr,1]] == v_next)[0][0]
        FBin[Map[v_curr,0]:Map[v_curr,1]][ind] += bottleneck
    
    #print('')
    #print('----- New iteration -----')
    #print('Bottleneck is: ', bottleneck)
    
    return FBin, False


#@njit
def Ford_Falkerson(Tsize, g):


    ### MAX FLOW ########################
    # Init
    Vsize = Tsize + 2

    bin_size = g[g > 0].shape[0]
    
    NBin = np.full((bin_size), -1)
    WBin = np.full((bin_size), -1)
    FBin = np.full((bin_size), -1)

    Map = np.full((g.shape[0],2), -1)

    start = 0
    for i in range(g.shape[0]):
        end = start + g[i][g[i] > 0].shape[0]

        Map[i] = [start, end]
        
        NBin[start:end] = np.array(np.where(g[i] > 0)[0])
        WBin[start:end] = np.array(g[i][g[i] > 0])
        FBin[start:end] = 0
        
        start = end

    # Main
    while True:
        FBin, end = iteration(FBin, WBin, NBin, Map, Vsize)
        if end : break



    ### MIN CUT #########################
    # BFS
    Q = [0] # Queue that start with S
    Prev = np.full(Vsize, -1) # Prev node
    cursor = 0 # instead of popping elements of of the Q

    while True:
        v_curr = Q[cursor]
        for i, neighbour in enumerate(NBin[Map[v_curr,0]:Map[v_curr,1]]):
            #print(f"{V[v_curr]} | neighbour: {neighbour} | Weights: {Weights[v_curr][i]} | {F[v_curr]} ? {Weights[v_curr][i]}")

            if FBin[Map[v_curr,0]:Map[v_curr,1]][i] < WBin[Map[v_curr,0]:Map[v_curr,1]][i] and Prev[neighbour] == -1:
                Q.append(neighbour)
                Prev[neighbour] = v_curr
        cursor += 1
        if cursor >= len(Q):
            break
            
    min_cut = np.array(Q, dtype=np.int32)
    
    res = np.zeros((Tsize), dtype=np.uint8)
    for v_ind in min_cut:
        res[v_ind] = 1
    

    return res


@njit
def translate_to_labeling(res, k_init, a_i):
    for i in range(0, res.size):
        if res[i] == 0:
            res[i] = k_init[i]
        else:
            res[i] = a_i

    return res
