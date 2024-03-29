import numpy as np
import qutip as qp
from qutip.measurement import measure, measurement_statistics
import matplotlib.pyplot as plt
import sys
import os

path = "experiments/Qubit"
sys.path.insert(0, os.getcwd())
from qrec.utils import *

def Circuit(state, params, noise_params):
    state = qp.rx(params[0] + noise_params[0]) * state
    return state

def Experiment(state, params, noise_params):
   state = Circuit(state, params, noise_params)
   value, new_state = measure(state, qp.sigmaz())
   return round(value)

def Score_function(state):
   eigenvalues, eigenstates, probabilities = measurement_statistics(state, qp.sigmaz())
   return probabilities[0]


def updates(beta_indx, reward, qlearn, lr=0.005):
    q0 = qlearn.q0
    n0 = qlearn.n0
    q0[beta_indx] += (1/n0[beta_indx])*(reward - q0[beta_indx])
    n0[beta_indx] += 1
    return q0, n0

def rewards(n):
    if int(n) == -1:
        return 1
    else:
        return 0

def Training(qlearn, betas_grid, hiperparam, epsilon, noise_params):

    indb, b = ep_greedy(qlearn.q0, betas_grid, hiperparam[2], nr_prob=epsilon)
    state = qp.basis(2, 0)
    n = Experiment(state, [b], noise_params)
    r = rewards(n)
    return indb, b, n, r

seed = 0
experiment_index = 1
alpha = 1.5
epsilon = 0.01
tables = define_q(beta_steps=10, range=[0, np.pi * 2])
betas_grid = tables.betas_grid

details = {"index":experiment_index, "alpha":alpha, "ep":epsilon, "betas":betas_grid,"experience":[],
 "Ps_greedy":[], "seed":seed, "tables":tables, "mean_rewards":[0, 0], "means":[]}
np.random.seed(seed)

### run q-learning
model = True
N=int(5e2)
np.random.seed(seed)
qlearn = define_q(beta_steps=10, range=[0, np.pi * 2])
hiperparam = [0.01, 15000, 0, 1]

noise_params = [0]

for i in range(N):
    if i % (N // 10) == 0:
        print(i)
    beta_indx, beta, outcome, reward = Training(qlearn, betas_grid, hiperparam, epsilon=0.05, noise_params=noise_params)
    q0, n0 = updates(beta_indx, reward, qlearn)

plt.plot(betas_grid, qlearn.q0, label="Q-learning")



scores = []
thetas = np.linspace(0, np.pi * 2, 10)
for i in range(len(thetas)):
    state = qp.basis(2, 0)

    state = Circuit(state, [thetas[i]], noise_params)
    scores.append(Score_function(state))

plt.plot(thetas, scores, label="Theoretical Reward")
plt.xlabel(r"$\theta$")
plt.ylabel("P(-1)")
plt.legend()
plt.show()