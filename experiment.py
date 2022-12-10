import matplotlib.pyplot as plt
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from notearsdiy import Notears
# data simulation, simulate true causal dag and train_data.
weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=10,
                                      weight_range=(0.5, 2.0), seed=1)
dataset = IIDSimulation(W=weighted_random_dag, n=2000, method='linear',
                        sem_type='uniform', noise_scale=1)
true_causal_matrix, X = dataset.B, dataset.X

notears = Notears()
notears.learn(X)
GraphDAG(notears.causal_matrix, true_causal_matrix, 'result')

# calculate metrics
mt = MetricsDAG(notears.causal_matrix, true_causal_matrix)
print(mt.metrics)

x = [i for i in range(len(notears.w_loss))]

plt.bar(x, notears.w_loss,color="#f89588",
        edgecolor='black', linewidth=1, zorder=10)
plt.title('loss in iterations')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(['loss'])
plt.show()


plt.bar(x, notears.h_loss,color="#7cd6cf",
        edgecolor='black', linewidth=1, zorder=10)
plt.title('h_loss in iterations')
plt.xlabel('iteration')
plt.ylabel('h_loss')
plt.legend(['h_loss'])
plt.show()

plt.bar(x, notears.rou,color="#f8cb7f",
        edgecolor='black', linewidth=1, zorder=10)
plt.title('rho(Lagrange) in iterations')
plt.xlabel('iteration')
plt.ylabel('rou')
plt.legend(['rou'])
plt.show()
