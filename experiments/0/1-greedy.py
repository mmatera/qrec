
experiment_index=0
import os
import sys
path = "experiments/{}/".format(experiment_index)
sys.path.insert(0, os.getcwd())
from qrec.utils import *
import time


def updates(indb, n, g, r, lr=0.01):
    q1[indb, n, g]+= (1/n1[indb,n,g])*(r - q1[indb, n, g])
    q0[indb]+= (1/n0[indb])*np.max([q1[indb, n, g] for g in [0,1]] - q0[indb])
    n0[indb]+=1
    n1[indb,n,g]+=1
    return

seed = 0
experiment_index = 0
alpha = 0.4
epsilon = 1.
betas_grid, tables = define_q(nbetas=10)
details = {"index":experiment_index, "alpha":alpha, "ep":epsilon, "betas":betas_grid,"experience":[],
 "Ps_greedy":[], "seed":seed, "tables":tables}
np.random.seed(seed)


### run q-learning
N=int(1e5)
np.random.seed(seed)
start = time.time()
betas_grid, [q0, q1,n0,n1] = define_q()
details = {"index":experiment_index, "alpha":alpha, "ep":"1.", "betas":betas_grid,"experience":[], "Ps_greedy":[], "seed":seed}
for experiment in range(N):
    if experiment%int(N/10)==0:
        print(experiment)
    hidden_phase = np.random.choice([0,1])
    indb, b = ep_greedy(q0, betas_grid, ep=1.)
    n = give_outcome(hidden_phase, b)
    indg, g = ep_greedy(q1[indb,n,:], [0,1], ep=1.)
    r = give_reward(g,hidden_phase)
    updates(indb, n, g, r)
    details["experience"].append([b,n,g,r])
    details["Ps_greedy"].append(Psq(q0,q1,betas_grid))
end = time.time() - start

details["total_time"] = end
#path is current directory, override by sys.path.insert...os.get
with open(path+"details.pickle","wb") as f:
    pickle.dump(details,f, protocol=pickle.HIGHEST_PROTOCOL)


plt.figure()
ax=plt.subplot(111)
ax.plot(betas_grid,q0,label=r'$Q(\beta)$')
ax.set_xlabel(r'$\beta$')
ax.plot(betas_grid,[1-Perr(b) for b in betas_grid],label=r'$P_s(\beta)$')
ax.legend(prop={"size":20})
plt.savefig(path+"q0.png")

stacked_history = np.stack(details["experience"])
min, pstar,bstar = model_aware_optimal(betas_grid)


plt.figure(figsize=(20,20))
ax=plt.subplot(311)
ax.plot(stacked_history[:,0],linewidth=.5,label=r'$\beta_t$')
ax.axhline(bstar,color="red",label=r'$\beta^*$')
ax.legend(prop={"size":20})
ax=plt.subplot(312)
ax.plot(np.cumsum(stacked_history[:,-1])/np.arange(1,len(stacked_history[:,-1])+1),label=r'$R_t/t$')
ax.legend(prop={"size":20})
ax.axhline(1.-pstar,color="black",label=r'$P_s^*$')
ax=plt.subplot(313)
ax.plot(details["Ps_greedy"],label=r'$P_t$')
ax.axhline(1.-pstar,color="black",label=r'$P_s^*$')
ax.set_xlabel(r'$experiment$')
ax.set_xscale("log")
ax.legend(prop={"size":20})
plt.savefig(path+"learning_rates.png")

#####
