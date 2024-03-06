from qrec.utils import *

betas_grid, [q0, q1, n0, n1] = define_q(nbetas=25)
min, pstar, bstar = model_aware_optimal(betas_grid)

plt.figure()
ax = plt.subplot(111)
ax.axvline(bstar, color="black")
ax.axhline(pstar, color="blue")
ax.plot(betas_grid, [Perr(b) for b in betas_grid])
