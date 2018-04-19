# Copyright (C) 2017, Sigvald Marholm and Diako Darian
#
# This file is part of ConstantBC.
#
# ConstantBC is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# ConstantBC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# ConstantBC.  If not, see <http://www.gnu.org/licenses/>.

from dolfin import *
from mshr import *
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from ConstantBC import *
from itertools import count

iterative_solver    = True
monitor_convergence = False
order               = 2
resolution          = 4

rho = Constant(0.0)
rho = Expression("100*x[1]", degree=2)

print("Loading external mesh")
fname = "mesh/circle_and_square_in_rectangle_res"+str(resolution)
mesh, bnd, num_objects = load_mesh(fname)

print("Making spaces")
V = FunctionSpace(mesh, "Lagrange", order)

phi = TrialFunction(V)
psi = TestFunction(V)

vsources = [[1,0,-1]]
# vsources = [[-1,0,1],[-1,1,2]]

bc_e = DirichletBC(V, Constant(0), bnd, 1)
objects = [ObjectBC(V, bnd, 2+i) for i in range(num_objects)]
circuit = Circuit(V, bnd, objects, vsources)

objects[0].charge = 7.
objects[1].charge = -3.

print("Creating variational form")

lhs = dot(grad(phi), grad(psi)) * dx
rhs = rho*psi * dx

print("Assembling matrix")
# Do not use assemble_system()
A = assemble(lhs)
b = assemble(rhs)

print("Applying boundary conditions")
bc_e.apply(A, b)
A, b = circuit.apply(A, b)
for o in objects:
    o.apply(A, b)

phi = Function(V)

if iterative_solver:
    print("Solving equation using iterative solver")
    solver = PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10 #e-12
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = monitor_convergence

    solver.set_operator(A)
    solver.solve(phi.vector(), b)
else:
    print("Solving equation using direct solver")
    solve(A, phi.vector(), b)

for o in objects:
    o.correct_charge(phi)

Qs = [o.charge for o in objects]
Vs = [o.get_potential(phi) for o in objects]

for i,Q,V in zip(count(),Qs,Vs):
    print("Object {}: Q={}, V={}".format(i,Q,V))

print("Tot/diff: Q={}, V={}".format(sum(Qs),Vs[1]-Vs[0]))

# ri = 0.2
# ro = 1.0
# print("Making plots")
# line = np.linspace(ri,ro,10000, endpoint=False)
# uh_line = np.array([phi(x,0,0) for x in line])
# ue_line = (Q.values()[0]/(4*np.pi))*(line**(-1)-ro**(-1))

# plt.plot(line, uh_line, label='Numerical')
# plt.plot(line, ue_line, '--', label='Exact')
# plt.legend(loc='lower left')
# plt.show()

print("Storing to file")
File("phi.pvd") << phi
