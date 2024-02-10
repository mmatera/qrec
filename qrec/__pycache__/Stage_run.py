import numpy as np
from .utils import *

def updates(indb, n, g, r, q0, q1, n0, n1, lr=0.01):
    q1[indb, n, g]+= (1/n1[indb,n,g])*(r - q1[indb, n, g])
    q0[indb]+= (1/n0[indb])*np.max([q1[indb, n, g] for g in [0,1]] - q0[indb])
    n0[indb]+= 1
    n1[indb,n,g]+= 1
    return q0, q1, n0, n1

def Update_reload(n0, n1, epsilon, change, rate):
    #epsilon += (1-epsilon) * np.abs(change) * rate
    for i in range(0, len(n1)):
        n0[i] /= 1 + rate * np.abs(change)
        for j in range(0, len(n1[i])):
            for k in range(0, len(n1[i][j])):
                n1[i, j, k] /= 1 + rate * np.abs(change)
    return n0, n1, epsilon

"""
Hiperparameters:
[0] Epsilon0.
[1] how fast epsilon varies.
[2] How fast the learning rate decreases when the avr reward has a negative derivative.
[3] Dispersion of random Gaussian.
[4] Linear relation between temperature and epsilon.
"""
def Experiment_run(details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam = [], N0=0, delta1= 50):
    start = time.time()
    mean_rew = 0
    points = [0, 0]
    epsilon0 = hiperparam[0]
    means = []
    epsilon = 1
    for experiment in range(N0, N0 + N):
        #print(epsilon)
        if experiment%int(N/10)==0:
            print(experiment)
        hidden_phase = np.random.choice([0,1])
        indb, b = ep_greedy(q0, betas_grid, hiperparam[3], hiperparam[4], ep=epsilon)
        n = give_outcome(hidden_phase, b, alpha=alpha)
        indg, g = ep_greedy(q1[indb,n,:], [0,1], hiperparam[3], hiperparam[4], ep=epsilon)
        r = give_reward(g,hidden_phase)

        means.append(r)
        if len(means) > delta1:
            means.pop(0)
        if experiment % delta1 == 0:
            points[0] = points[1]
            mean_rew = 0
            for i in range(len(means)):
                mean_rew += means[i] / delta1 
            points[1] = mean_rew
            mean_deriv = (points[1] - points[0]) / delta1
            if mean_deriv < 0:
                n0, n1, epsilon = Update_reload(n0, n1, epsilon, mean_deriv, hiperparam[2])
            else:
                epsilon -= epsilon0
                epsilon /= hiperparam[1]
                epsilon += epsilon0



        q0, q1, n0, n1 = updates(indb, n, g, r, q0, q1, n0, n1)
        details["experience"].append([b,n,g,r])
        details["Ps_greedy"].append(Psq(q0,q1,betas_grid, hiperparam[3], hiperparam[4],alpha=alpha))
    details["tables"] = [q0,q1,n0,n1]
    end = time.time() - start
    details["total_time"] = end
    return details