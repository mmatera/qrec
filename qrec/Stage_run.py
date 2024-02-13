import numpy as np
from .utils import *
from Model_Semi_aware.find_unobservables import Find_optimal_intensity

def updates(indb, n, g, r, q0, q1, n0, n1, lr=0.01):
    q1[indb, n, g]+= (1/n1[indb,n,g])*(r - q1[indb, n, g])
    q0[indb]+= (1/n0[indb])*np.max([q1[indb, n, g] for g in [0,1]] - q0[indb])
    n0[indb]+= 1
    n1[indb,n,g]+= 1
    return q0, q1, n0, n1

def Update_reload(n0, n1, epsilon, mean_rew):
    #epsilon = (1-epsilon) * mean_rew
    for i in range(0, len(n1)):
        n0[i] = 500 * mean_rew
        if n0[i] < 1:
            n0[i] = 1
        for j in range(0, len(n1[i])):
            for k in range(0, len(n1[i][j])):
                n1[i, j, k] *= mean_rew
                if n1[i, j, k] < 1:
                    n1[i, j, k] = 1
    return n0, n1, epsilon

"""
Hiperparameters:
[0] Epsilon0.
[1] how fast epsilon varies.
[2] How fast the learning rate decreases when the avr reward has a negative derivative.
[3] Dispersion of random Gaussian.
[4] Linear relation between temperature and epsilon.
"""
def Experiment_run(details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam = [], N0=0, delta1= 100):
    start = time.time()
    mean_rew = 0
    points = [0, 0]
    means = []
    mean_rew = 0

    epsilon = details["epsilon"]
    for experiment in range(N0, N0 + N):
        epsilon -= hiperparam[0]
        epsilon /= hiperparam[1]
        epsilon += hiperparam[0]

        if experiment%int(N/10)==0:
            print(experiment)
        hidden_phase = np.random.choice([0,1])
        indb, b = ep_greedy(q0, betas_grid, hiperparam[3], hiperparam[4], ep=epsilon)
        n = give_outcome(hidden_phase, b, alpha=alpha)
        indg, g = ep_greedy(q1[indb,n,:], [0,1], hiperparam[3], hiperparam[4], ep=epsilon)
        r = give_reward(g,hidden_phase)

        means.append(r)
        mean_rew += r / delta1
        
        if len(means) > delta1:
            mean_rew -= means[0] / delta1
            means.pop(0)
            

        details["mean_rewards"].append(mean_rew)

        if experiment % delta1 == 0:
            points[0] = points[1]
            
            points[1] = mean_rew
            mean_deriv = (points[1] - points[0]) / delta1
            if mean_deriv < -0.001 and n0[indb] > 300:  # Set a better value...
                n0, n1, epsilon = Update_reload(n0, n1, epsilon, mean_rew)


        q0, q1, n0, n1 = updates(indb, n, g, r, q0, q1, n0, n1)
        details["experience"].append([b,n,g,r])
        details["Ps_greedy"].append(Psq(q0,q1,betas_grid, hiperparam[3], hiperparam[4],alpha=alpha))
    details["tables"] = [q0,q1,n0,n1]
    end = time.time() - start
    details["total_time"] = end
    details["epsilon"] = epsilon
    return details

def Reset_with_model(alpha, beta_grid, q0, n0, n1):
    for i in range(len(beta_grid)):
        vals = 0.5 * np.exp(-np.abs(-alpha - beta_grid[i])**2) +  0.5 * (1 - np.exp(-np.abs(alpha - beta_grid[i])**2))
        q0[i] = vals
    
    epsilon = 0.01
    for i in range(len(n1)):
        n0[i] = 1000
        for j in range(len(n1[i])):
            for k in range(len(n1[i,j])):
                n1[i,j,k] = 1000

    return q0, epsilon, n0

def Model_experiment(details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam = [], N0=0, delta1= 1000, current=1.5):
    start = time.time()
    mean_rew = 0
    points = [0, 0]
    epsilon0 = hiperparam[0]
    means = []
    mean_rew = 0

    model_tries = 200
    delta_guess = 0.4
    guessed_intensity = 0
    exp_vals = {"Phase":[], "Betas":[], "Observations":[]}
    epsilon = details["epsilon"]
    Checked = False

    for experiment in range(N0, N0 + N):
        if experiment%int(N/10)==0:
            print(experiment)
        hidden_phase = np.random.choice([0,1])
        indb, b = ep_greedy(q0, betas_grid, hiperparam[3], hiperparam[4], ep=epsilon)
        n = give_outcome(hidden_phase, b, alpha=alpha)
        indg, g = ep_greedy(q1[indb,n,:], [0,1], hiperparam[3], hiperparam[4], ep=epsilon)
        r = give_reward(g,hidden_phase)


        # Hay que optimizar esto si o si...
        exp_vals["Phase"].append((-1) ** hidden_phase) # Hay que cambiar la convención de signos.
        exp_vals["Betas"].append(b)
        exp_vals["Observations"].append(n)
        if len(exp_vals) > model_tries:
            exp_vals['Phase'].pop(0)
            exp_vals["Betas"].pop(0)
            exp_vals["Observations"].pop(0)
        
        
        means.append(r)
        mean_rew += r / delta1
        
        if len(means) > delta1:
            mean_rew -= means[0] / delta1
            means.pop(0)
            

        details["mean_rewards"].append(mean_rew)

        if experiment % delta1 == 0:
            points[0] = points[1]
            
            points[1] = mean_rew
            mean_deriv = (points[1] - points[0]) / delta1
            if mean_deriv < 0:
                #print(mean_deriv)
                n0, n1, epsilon = Update_reload(n0, n1, epsilon, mean_rew)
                Checked = True
            #else:
            #    epsilon -= epsilon0
            #    epsilon /= hiperparam[1]
            #    epsilon += epsilon0

        if Checked:
            #guessed_intensity = Find_optimal_intensity(exp_vals["Phase"], exp_vals["Betas"], exp_vals["Observations"])
            guessed_intensity = 0.3
            #print(guessed_intensity)
            if np.abs(guessed_intensity - current) > 0.3:
                current = guessed_intensity
                q0, epsilon, n0 = Reset_with_model(current, betas_grid, q0, n0, n1)
            Checked = False

        q0, q1, n0, n1 = updates(indb, n, g, r, q0, q1, n0, n1)
        details["experience"].append([b,n,g,r])
        details["Ps_greedy"].append(Psq(q0,q1,betas_grid, hiperparam[3], hiperparam[4],alpha=alpha))
    details["tables"] = [q0,q1,n0,n1]
    end = time.time() - start
    details["total_time"] = end
    details["epsilon"] = epsilon
    return details