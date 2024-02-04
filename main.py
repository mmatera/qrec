import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import time
import pickle
import os

def p(alpha,n):
    """
    p(n|alpha), born rule
    """
    pr = np.exp(-(alpha)**2)
    return [pr, 1-pr][n]

def Perr(beta,alpha=0.4):
    ps=0
    for n in range(2):
        ps+=np.max([p(sgn*alpha + beta,n) for sgn in [-1,1]])
    return 1-ps/2


bmin,bmax=-2.,0.
betas_grid = np.linspace(bmin,bmax,25)
alpha = 0.4

#### Landscape inspection

mmin = minimize(Perr, x0=-alpha, bounds = [(bmin, bmax)])
p_star = mmin.fun
beta_star = mmin.x

plt.figure()
ax=plt.subplot(111)
ax.axvline(beta_star,color="black")
ax.axhline(p_star,color="blue")
ax.plot(betas_grid,[Perr(b) for b in betas_grid])


####   Q-Learning approach
def define_q(nbetas=10):
    betas_grid = np.linspace(-2, 0, nbetas)
    q0 = np.zeros(betas_grid.shape[0])  #Q(beta)
    q1 = np.zeros((betas_grid.shape[0],2,2)) # Q(beta,n; g)
    n0 = np.ones(betas_grid.shape[0])  #Q(beta)
    n1 = np.ones((betas_grid.shape[0],2,2)) # Q(beta,n; g)
    return betas_grid, q0, q1,n0,n1

def greedy(arr):
    return np.random.choice(np.where( arr == np.max(arr))[0])

def policy(qvals, actions, ep=1.):
    """
    policy(q1, betas_grid)
    policy(q1[1,0,:], [0,1])
    """
    if np.random.random() < ep:
        inda = np.random.choice(range(len(actions)))
    else:
        inda = greedy(qvals)
    return inda,actions[inda]

def give_outcome(hidden_phase, beta, alpha=0.4):
    """
    hidden_phase in {0,1}
    """
    return np.random.choice([0,1], p= [p(alpha*(-1)**hidden_phase + beta,n) for n in [0,1]])

def give_reward(g, hidden_phase):
    if int(g) == int(hidden_phase):
        return 1.
    else:
        return 0.

def updates(indb, n, g, r, lr=0.01):
    q1[indb, n, g]+= (1/n1[indb,n,g])*(r - q1[indb, n, g])
    q0[indb]+= (1/n0[indb])*np.max([q1[indb, n, g] for g in [0,1]] - q0[indb])
    n0[indb]+=1
    n1[indb,n,g]+=1
    return

experiment_index = 0
seed = 0
np.random.seed(seed)
start = time.time()
betas_grid, q0, q1,n0,n1 = define_q()
history=[]
for experiment in range(int(1e6)):
    if experiment%int(1e5)==0:
        print(experiment)
    hidden_phase = np.random.choice([0,1])
    indb, b = policy(q0, betas_grid, ep=1.)
    n = give_outcome(hidden_phase, b)
    indg, g = policy(q1[indb,n,:], [0,1], ep=1.)
    r = give_reward(g,hidden_phase)
    updates(indb, n, g, r)
    history.append([b,n,g,r])
end = time.time() - start

details = {"index":experiment_index, "alpha":alpha, "ep":"1.", "betas":betas_grid,"history":history, "seed":seed,"total_time":end}
directory = "kennedy/{}/".format(experiment_index)
os.makedirs(directory,exist_ok=True)
with open(directory+"details.pickle","wb") as f:
    pickle.dump(details,f, protocol=pickle.HIGHEST_PROTOCOL)


plt.figure()
ax=plt.subplot(111)
ax.plot(betas_grid,q0,label=r'$Q(\beta)$')
ax.set_xlabel(r'$\beta$')
ax.plot(betas_grid,[1-Perr(b) for b in betas_grid],label=r'$P_s(\beta)$')
ax.legend(prop={"size":20})
plt.savefig(directory+"q0.png")


stacked_history = np.stack(details["history"])

plt.figure()
ax=plt.subplot(111)
ax.plot(np.cumsum(stacked_history[:,-1])/np.arange(1,len(stacked_history[:,-1])+1),label=r'$R_t/t$')
ax.set_xlabel(r'$experiment$')
ax.set_xscale("log")
ax.legend(prop={"size":20})
plt.savefig(directory+"learning_rates.png")

#####
