experiment_index = 3
import os
import sys

path = "experiments/{}/".format(experiment_index)
sys.path.insert(0, os.getcwd())
import time

from qrec.utils import *


def updates(indb, n, g, r, lr=0.005):
    q1[indb, n, g] += lr * (r - q1[indb, n, g])
    q0[indb] += lr * np.max([q1[indb, n, g] for g in [0, 1]] - q0[indb])
    n0[indb] += 1
    n1[indb, n, g] += 1
    return


seed = 0
experiment_index = 3
alpha = 1.5
epsilon = 0.01
betas_grid, tables = define_q(nbetas=10)
details = {
    "index": experiment_index,
    "alpha": alpha,
    "ep": epsilon,
    "betas": betas_grid,
    "experience": [],
    "Ps_greedy": [],
    "seed": seed,
    "tables": tables,
}
np.random.seed(seed)


### run q-learning
N = int(1e5)
np.random.seed(seed)
start = time.time()
betas_grid, [q0, q1, n0, n1] = define_q()
details = {
    "index": experiment_index,
    "alpha": alpha,
    "ep": epsilon,
    "betas": betas_grid,
    "experience": [],
    "Ps_greedy": [],
    "seed": seed,
}
for experiment in range(N):
    if experiment % int(N / 10) == 0:
        print(experiment)
    hidden_phase = np.random.choice([0, 1])
    indb, b = ep_greedy(q0, betas_grid, ep=epsilon)
    n = give_outcome(hidden_phase, b, alpha=alpha)
    indg, g = ep_greedy(q1[indb, n, :], [0, 1], ep=epsilon)
    r = give_reward(g, hidden_phase)
    updates(indb, n, g, r)
    details["experience"].append([b, n, g, r])
    details["Ps_greedy"].append(Psq(q0, q1, betas_grid, alpha=alpha))
details["tables"] = [q0, q1, n0, n1]
end = time.time() - start
details["total_time"] = end

os.makedirs("../data_rec/experiments/{}/".format(experiment_index), exist_ok=True)
with open(
    "../data_rec/experiments/{}/details.pickle".format(experiment_index), "wb"
) as f:
    pickle.dump(details, f, protocol=pickle.HIGHEST_PROTOCOL)

details.keys()
betas_grid = details["betas"]
stacked_history = np.stack(details["experience"])

plt.figure()
ax = plt.subplot(111)
ax.plot(betas_grid, q0, label=r"$Q(\beta)$")
ax.set_xlabel(r"$\beta$")
ax.plot(
    betas_grid, [1 - Perr(b, alpha=alpha) for b in betas_grid], label=r"$P_s(\beta)$"
)
ax.legend(prop={"size": 20})
plt.savefig(path + "q0.png")


min, pstar, bstar = model_aware_optimal(betas_grid, alpha=alpha)
stacked_history = np.stack(details["experience"])
counts, bins = np.histogram(stacked_history[:, 0], bins=len(betas_grid))
x_bins = np.linspace(np.min(bins), np.max(bins), len(bins) - 1)

plt.figure(figsize=(20, 20))
ax = plt.subplot(311)
ax.bar(x_bins, counts, label=r"$\beta_t$", width=np.std(bins[:2]))
ax.axvline(bstar, color="black")
ax.set_yscale("log")
ax.legend(prop={"size": 20})
ax = plt.subplot(312)
ax.plot(
    np.cumsum(stacked_history[:, -1]) / np.arange(1, len(stacked_history[:, -1]) + 1),
    label=r"$R_t/t$",
)
ax.legend(prop={"size": 20})
ax.set_xscale("log")
ax.axhline(1.0 - pstar, color="black", label=r"$P_s^*$")
ax = plt.subplot(313)
ax.plot(details["Ps_greedy"], label=r"$P_t$")
ax.axhline(1.0 - pstar, color="black", label=r"$P_s^*$")
ax.set_xlabel(r"$experiment$")
ax.set_xscale("log")
ax.legend(prop={"size": 20})
plt.savefig(path + "learning_curve.png")

#####
