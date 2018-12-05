# n is the number of clienst
# N is set of clients, with N = {1,2,3, .... n}
# V is set of vehicles, with V = {0} U N
# A is ste of arcs, with A = {i,}


import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model

rnd = np.random
rnd.seed(0)

n = 10
Q = 15
N = [i for i in range(1, n + 1)]
V = [0] + N
q = {i: rnd.randint(1, 10) for i in N}
loc_x = rnd.rand(len(V)) * 200
loc_y = rnd.rand(len(V)) * 100

plt.scatter(loc_x[1:], loc_y[1:], c='b')
for i in N:
    plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))

plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')
# plt.show()

A = [(i, j) for i in V for j in V if i != j]
c = {(i, j): np.hypot(loc_x[i] - loc_x[j], loc_y[i] - loc_y[j]) for i, j in A}

md1 = Model('CVRP')
x = md1.binary_var_dict(A, name = 'x')
u = md1.continuous_var_dict(N, ub=Q, name='u')

md1.minimize(md1.sum(c[i,j]*x[i,j] for i, j in A))
md1.add_constraints(md1.sum(x[i,j] for j in V if j!=i)==1 for i in N)
md1.add_constraints(md1.sum(x[i,j] for i in V if i!=j)==1 for j in N)

md1.add_indicator_constraints(md1.indicator_constraint(x[i,j],u[i]+q[j]==u[j]) for i,j in A if i!=0 and j!=0)
md1.add_constraints(u[i]>=q[i] for i in N)

solution = md1.solve(log_output=True)
#print(solution)


active_arc ={a for a in A if x[a].solution_value>0.9}

plt.scatter(loc_x[1:], loc_y[1:], c='b')
for i in N:
    plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))
for i,j in active_arc:
    plt.plot([loc_x[i],loc_x[j]], [loc_y[i],loc_y[j]], c='g', alpha=0.3)
plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')
#plt.show()

print(solution.solve_status)