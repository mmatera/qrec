import numpy as np
from .utils import *
from Model_Semi_aware.find_unobservables import Find_optimal_intensity
from Model_Semi_aware.intensity_guess import guess_intensity
import time
"""
Hiperparameters:
[0] Epsilon0.
[1] how fast epsilon varies.
[2] Dispersion of random Gaussian.
[3] Learning rate reset.
"""

# How the q-learning parameters update.
def updates(indb, n, g, r, q0, q1, n0, n1, lr=0.001):
    q1[indb, n, g] += (1/n1[indb,n,g])*(r - q1[indb, n, g])
    q0[indb] += (1/n0[indb])*np.max([q1[indb, n, g] for g in [0,1]] - q0[indb])
    n0[indb] += 1
    n1[indb,n,g] += 1
    return q0, q1, n0, n1

# How the learning rate changes when the enviroment changes.
# It could be interesting to change the reward with the mean_rew.
def Update_reload(n0, n1, mean_rew, restart_point, restart_epsilon):
    epsilon = restart_epsilon
    for i in range(0, len(n1)):
        n0[i] = restart_point
        if n0[i] < 1:
            n0[i] = 1
        for j in range(0, len(n1[i])):
            for k in range(0, len(n1[i][j])):
                n1[i, j, k] = restart_point
                if n1[i, j, k] < 1:
                    n1[i, j, k] = 1
    return n0, n1, epsilon

# Reload the Q-function with the model.
def Reset_with_model(alpha, beta_grid, q0, q1):
    for i in range(len(beta_grid)):
        q0[i] = 1 - Perr_model(beta_grid[i], alpha)

    for i in range(len(q1)):
        for j in range(len(q1[i])):
            for k in range(len(q1[i, j])):
                q1[i,j,k] = p_model((-1)**(k+1) * alpha, - beta_grid[i], j) / (p_model(-alpha, - beta_grid[i], j) + p_model(alpha, - beta_grid[i], j))

    return q0, q1

# Change in beta.
def Experiment_noise_1(q0, q1, betas_grid, hiperparam, epsilon, alpha, lambd):
    hidden_phase = np.random.choice([0,1])

    indb, b = ep_greedy(q0, betas_grid, hiperparam[2], ep=epsilon)
    n = give_outcome(hidden_phase, b, alpha=alpha, lambd=lambd)
    indg, g = ep_greedy(q1[indb,n,:], [0,1], hiperparam[2], ep=epsilon)
    r = give_reward(g,hidden_phase)
    return indb, b, n, indg, g, r

# Change in the priors
def Experiment_noise_2(q0, q1, betas_grid, hiperparam, epsilon, alpha, lambd):
    hidden_phase = np.random.choice([0,1], p=[0.5 - lambd, 0.5 + lambd]) 
    indb, b = ep_greedy(q0, betas_grid, hiperparam[2], ep=epsilon)
    n = give_outcome(hidden_phase, b, alpha=alpha, lambd=0)
    indg, g = ep_greedy(q1[indb,n,:], [0,1], hiperparam[2], ep=epsilon)
    r = give_reward(g,hidden_phase)
    return indb, b, n, indg, g, r


# Exactly the same but checks with model to update initial parameters.
def Run_Experiment(details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam = [], delta1= 1000, current=1.5, lambd=0.0, model=True, noise_type=None):
    start = time.time()
    mean_rew = float(details["mean_rewards"][-1])
    points = [mean_rew, float(details["mean_rewards"][-2])]
    means = details["means"]
    
    guessed_intensity = 0  # Could be initialized with the previous value but there is no reason...
    epsilon = float(details["ep"])
    Checked = False

    for experiment in range(0, N):
        if epsilon > 0.05:
            epsilon -= hiperparam[1]
        else:
            epsilon = 0.05
        if experiment % (N // 10) == 0:
            print(experiment)

        if noise_type == 0:
            pass
        if noise_type == 1:
            indb, b, n, indg, g, r = Experiment_noise_1(q0, q1, betas_grid, hiperparam, epsilon, alpha, lambd)
        if noise_type == 2:
            indb, b, n, indg, g, r = Experiment_noise_2(q0, q1, betas_grid, hiperparam, epsilon, alpha, lambd)

        means, mean_rew = calculate_mean_rew(means, mean_rew, r, delta1)

        
        # Check if the reward is smaller
        if experiment % delta1 == 0:
            points[0] = points[1]
            points[1] = mean_rew
            mean_deriv = points[1] - points[0]
            if mean_deriv <= -0.2 and n0[indb] > 300:
                n0, n1, epsilon = Update_reload(n0, n1, mean_rew, hiperparam[3], hiperparam[0])
                Checked = True

        # If it looks like changed, guess a new intensity to verify.
        if model and Checked:
            guessed_intensity = guess_intensity(alpha, delta1 * 10, lambd=lambd)
            print(guessed_intensity)
            if np.abs(guessed_intensity - current) > 5 / np.sqrt(delta1 * 10):
                current = guessed_intensity
                q0, q1 = Reset_with_model(current, betas_grid, q0, q1)
            Checked = False

        q0, q1, n0, n1 = updates(indb, n, g, r, q0, q1, n0, n1)

        min, pstar, bstar = model_aware_optimal(betas_grid, alpha=alpha, lambd=lambd)

        details["mean_rewards"].append(mean_rew)
        details["means"] = means
        details["experience"].append([b,n,g,r/(1-pstar)])
        details["Ps_greedy"].append(Psq(q0,q1,betas_grid, hiperparam[2], alpha=alpha, lambd=lambd))
    details["tables"] = [q0,q1,n0,n1]
    end = time.time() - start
    details["total_time"] = end
    details["ep"] = f"{epsilon}"
    print(n0)
    return details
