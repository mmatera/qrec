import numpy as np
from .utils import *
from Model_Semi_aware.find_unobservables import Find_optimal_intensity

"""
Hiperparameters:
[0] Epsilon0.
[1] how fast epsilon varies.
[2] Dispersion of random Gaussian.
[3] Linear relation between temperature and epsilon.
"""

# How the q-learning parameters update.
def updates(indb, n, g, r, q0, q1, n0, n1, lr=0.001):
    q1[indb, n, g]+= (1/n1[indb,n,g])*(r - q1[indb, n, g])
    q0[indb]+= (1/n0[indb])*np.max([q1[indb, n, g] for g in [0,1]] - q0[indb])
    #q1[indb, n, g]+= lr*(r - q1[indb, n, g])
    #q0[indb]+= lr*np.max([q1[indb, n, g] for g in [0,1]] - q0[indb])
    n0[indb]+= 1
    n1[indb,n,g]+= 1
    return q0, q1, n0, n1

# How the learning rate changes when the enviroment changes.
def Update_reload(n0, n1, epsilon, mean_rew, restart_point):
    epsilon = 0.15  # Example, but important to see the maximum reward and current reward.
    for i in range(0, len(n1)):
        n0[i] = restart_point * mean_rew
        if n0[i] < 1:
            n0[i] = 1
        for j in range(0, len(n1[i])):
            for k in range(0, len(n1[i][j])):
                n1[i, j, k] = restart_point * mean_rew
                if n1[i, j, k] < 1:
                    n1[i, j, k] = 1
    return n0, n1, epsilon

# How to reload if you have a model.
def Reset_with_model(alpha, beta_grid, q0, q1, n0, n1, restart_value):
    for i in range(len(beta_grid)):
        vals = 0.5 * np.exp(-np.abs(-alpha - beta_grid[i])**2) +  0.5 * (1 - np.exp(-np.abs(alpha - beta_grid[i])**2))
        q0[i] = vals

    for i in range(len(q1)):
        for j in range(len(q1[i])):
            for k in range(len(q1[i, j])):
                q1[i,j,k] = p((-1)**(k+1) * alpha - beta_grid[i], j) / (p(-alpha - beta_grid[i], j) + p(alpha - beta_grid[i], j))

    epsilon = 0.01
    for i in range(len(n1)):
        n0[i] = restart_value
        for j in range(len(n1[i])):
            for k in range(len(n1[i,j])):
                n1[i,j,k] = restart_value

    return q0, epsilon, n0

# All the experiment and update of parameters.
def Experiment_run(details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam = [], delta1=500):
    start = time.time()
    mean_rew = 0
    points = [1, 1]
    means = []
    mean_rew = 0
    epsilon = float(details["ep"])

    for experiment in range(0, N):
        epsilon -= hiperparam[0]
        epsilon /= hiperparam[1]
        epsilon += hiperparam[0]

        if experiment%int(N/10)==0:
            print(experiment)
        hidden_phase = np.random.choice([0,1])
        indb, b = ep_greedy(q0, betas_grid, hiperparam[2], hiperparam[3], ep=epsilon)
        n = give_outcome(hidden_phase, b, alpha=alpha)
        indg, g = ep_greedy(q1[indb,n,:], [0,1], hiperparam[2], hiperparam[3], ep=epsilon)
        r = give_reward(g,hidden_phase)
        means, mean_rew = calculate_mean_rew(means, mean_rew, r, delta1)


        # Estimate how much the mean reward is changing.   
        if experiment % delta1 == 0:
            points[0] = points[1]
            
            points[1] = mean_rew
            mean_deriv = (points[1] - points[0]) / delta1

            if mean_deriv < -0.001 and n0[indb] > 300:  # if the change is big enough change the learning rate and epsilon.
                n0, n1, epsilon = Update_reload(n0, n1, epsilon, mean_rew, hiperparam[4])

        # update the values of the q-learning.
        q0, q1, n0, n1 = updates(indb, n, g, r, q0, q1, n0, n1)

        # Save all the data
        details["mean_rewards"].append(mean_rew)
        details["experience"].append([b,n,g,r])
        details["Ps_greedy"].append(Psq(q0,q1,betas_grid, hiperparam[2], hiperparam[3],alpha=alpha))
    details["tables"] = [q0,q1,n0,n1]
    end = time.time() - start
    details["total_time"] = end
    details["ep"] = f"{epsilon}"

    return details

# Exactly the same but checks with model to update initial parameters.
def Model_experiment(details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam = [], N0=0, delta1= 10000, current=1.5):
    start = time.time()
    mean_rew = 0
    points = [1, 1]
    means = []
    mean_rew = 0

    model_tries = 1000
    delta_guess = 0.4
    guessed_intensity = 0
    exp_vals = {"Phase":[], "Betas":[], "Observations":[]}
    epsilon = float(details["ep"])
    Checked = False

    for experiment in range(N0, N0 + N):
        if experiment%int(N/10)==0:
            print(experiment)
        hidden_phase = np.random.choice([0,1])
        indb, b = ep_greedy(q0, betas_grid, hiperparam[2], hiperparam[3], ep=epsilon)
        n = give_outcome(hidden_phase, b, alpha=alpha)
        indg, g = ep_greedy(q1[indb,n,:], [0,1], hiperparam[2], hiperparam[3], ep=epsilon)
        r = give_reward(g,hidden_phase)
        means, mean_rew = calculate_mean_rew(means, mean_rew, r, delta1)
        

        # If the reward is decreasing then reset in a fixed reward.
        if experiment % delta1 == 0:
            points[0] = points[1]
            points[1] = mean_rew
            mean_deriv = (points[1] - points[0]) / delta1
            if mean_deriv <= -0.4/delta1 and n0[indb] > 300:
                Checked = True

        if Checked:
            exp_vals["Phase"].append((-1) ** hidden_phase)
            exp_vals["Betas"].append(b)
            exp_vals["Observations"].append(n)
            if len(exp_vals["Phase"]) >= model_tries:
                #guessed_intensity = Find_optimal_intensity(exp_vals["Phase"], exp_vals["Betas"], exp_vals["Observations"])
                guessed_intensity = 0.3
                print(guessed_intensity)
                if np.abs(guessed_intensity - current) > 0.3:
                    current = guessed_intensity
                    q0, epsilon, n0 = Reset_with_model(current, betas_grid, q0, q1, n0, n1, hiperparam[4])
                Checked = False
                exp_vals["Phase"] = []
                exp_vals["Betas"] = []
                exp_vals["Observations"] = []

        q0, q1, n0, n1 = updates(indb, n, g, r, q0, q1, n0, n1)

        details["mean_rewards"].append(mean_rew)
        details["experience"].append([b,n,g,r])
        details["Ps_greedy"].append(Psq(q0,q1,betas_grid, hiperparam[2], hiperparam[3],alpha=alpha))
    details["tables"] = [q0,q1,n0,n1]
    end = time.time() - start
    details["total_time"] = end
    details["ep"] = f"{epsilon}"
    print(n0)
    return details