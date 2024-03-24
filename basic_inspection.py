import matplotlib.pyplot as plt

from qrec.utils import bayes_decision_error_probability, define_q, model_aware_optimal

qlearning = define_q(beta_steps=25)
min, pstar, bstar = model_aware_optimal(qlearning.betas_grid)

plt.figure()
ax = plt.subplot(111)
ax.axvline(bstar, color="black")
ax.axhline(pstar, color="blue")
ax.plot(
    qlearning.betas_grid,
    [bayes_decision_error_probability(b) for b in qlearning.betas_grid],
)
plt.show()
