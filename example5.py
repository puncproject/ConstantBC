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

iterative_solver    = True
monitor_convergence = False
order               = 2
resolution          = 4

rho = Constant(0.0)
rho = Expression("100*x[1]", degree=2)

Q = [Constant(7.), Constant(-3.)]

print("Loading external mesh")
fname = "mesh/circle_and_square_in_rectangle_res"+str(resolution)
mesh, bnd, num_objects = load_mesh(fname)

print("Making spaces")
V = FunctionSpace(mesh, "Lagrange", order)

phi = TrialFunction(V)
psi = TestFunction(V)

bc_e = DirichletBC(V, Constant(0), bnd, 1)
objects = [ConstantBC(V, bnd, 2+i) for i in range(num_objects)]
# constraints = CircuitConstraints(objects, vsources)

print("Creating variational form")

lhs = dot(grad(phi), grad(psi)) * dx
rhs = rho*psi * dx

print("Assembling matrix")
# Do not use assemble_system()
A = assemble(lhs)
b = assemble(rhs)

print("Applying boundary conditions")
bc_e.apply(A, b)

for o in objects:
    o.apply(A,b)

# apply_circuit_constraints(A, b, objects, vsources)

vsources = [[1,0,-1]]

groups = get_charge_sharing_sets(vsources, 2)

R  = FunctionSpace(mesh, "Real", 0)
mu = TestFunction(R)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
n = FacetNormal(mesh)
code = open('addrow2.cpp', 'r').read()
compiled = compile_extension_module(code=code)

for group in groups:

    ds_group = np.sum([dss(i+2) for i in group])
    Q_group  = np.sum([Q[i](0)  for i in group])
    S = assemble(1.*ds_group)

    a0 = inner(mu, dot(grad(phi), n))*ds_group
    A0 = assemble(a0)
    cols, vals = A0.getrow(0)

    obj_id = group[0]
    obj = objects[obj_id]
    row = list(obj.get_boundary_values().keys())[0]

    B = Matrix()
    compiled.addrow(A, B, cols, vals, row, V)
    A = B

    b[row] = Q_group

used_object_rows   = [g[0] for g in groups]
unused_object_rows = list(set(range(num_objects))-set(used_object_rows))
unused_rows = [list(objects[i].get_boundary_values().keys())[0] for i in unused_object_rows]

for vsource, row in zip(vsources, unused_rows):

        obj_a_id = vsource[0]
        obj_b_id = vsource[1]
        V_ab     = vsource[2]

        cols = []
        vals = []

        if obj_a_id != -1:
            obj_a = objects[obj_a_id]
            dof_a = list(obj_a.get_boundary_values().keys())[0]
            cols.append(dof_a)
            vals.append(-1.0)

        if obj_b_id != -1:
            obj_b = objects[obj_b_id]
            dof_b = list(obj_b.get_boundary_values().keys())[0]
            cols.append(dof_b)
            vals.append(+1.0)

        cols = np.array(cols, dtype=np.uintp)
        vals = np.array(vals)

        print(cols, vals, V_ab)

        B = Matrix()
        compiled.addrow(A, B, cols, vals, row, V)
        A = B

        b[row] = V_ab


#     for obj_id in group[1:]:

#         obj = objects[obj_id]
#         bnd_dof = list(obj.get_boundary_values().keys())[0]

#         vsource = vsources[vs_id]
#         obj_a_id = vsource[0]
#         obj_b_id = vsource[1]
#         V_ab = vsource[2]

#         cols = []
#         vals = []

#         if obj_a_id != -1:
#             obj_a = objects[obj_a_id]
#             dof_a = list(obj_a.get_boundary_values().keys())[0]
#             cols.append(dof_a)
#             vals.append(-1.0)

#         if obj_b_id != -1:
#             obj_b = objects[obj_b_id]
#             dof_b = list(obj_b.get_boundary_values().keys())[0]
#             cols.append(dof_b)
#             vals.append(+1.0)

#         cols = np.array(cols, dtype=np.uintp)
#         vals = np.array(vals)

#         # obj_a = objects[obj_a_id]
#         # obj_b = objects[obj_b_id]
#         # dof_a = list(obj_a.get_boundary_values().keys())[0]
#         # dof_b = list(obj_b.get_boundary_values().keys())[0]

#         # cols = np.array([dof_a, dof_b],dtype=np.uintp)
#         # vals = np.array([-1.0, 1.0])

#         B = Matrix()
#         compiled.addrow(A, B, cols, vals, bnd_dof, V)
#         A = B

#         b[bnd_dof] = V_ab

#         vs_id += 1


# for i, o in enumerate(objects):
#     print("Setting final dof on boundary to enforce inner(dot(grad(u), n))*dsi = Q")

#     ds_i = dss(2+i)
#     S = assemble(1.*ds_i)

#     a0 = inner(mu, dot(grad(phi), n))*ds_i
#     A0 = assemble(a0)

#     bnd_dof = list(o.get_boundary_values().keys())[0]
#     row, col = A0.getrow(0)
#     code = open('addrow.cpp', 'r').read()
#     compiled = compile_extension_module(code=code)
#     B = Matrix()
#     compiled.addrow(A, B, A0, bnd_dof, V)
#     A = B
#     b[bnd_dof] = Q[i](0)

phi = Function(V)

if iterative_solver:
    print("Solving equation using iterative solver")
    solver = PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10 #e-12
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = monitor_convergence

    solver.set_operator(B)
    solver.solve(phi.vector(), b)
else:
    print("Solving equation using direct solver")
    solve(B, phi.vector(), b)

print("Computing actual object charge")
Qm = assemble(dot(grad(phi), n) * dss(2))
print("Object 1 charge: ", Qm)
Qm = assemble(dot(grad(phi), n) * dss(3))
print("Object 2 charge: ", Qm)
Qm = assemble(dot(grad(phi), n) * (dss(2)+dss(3)))
print("Total charge: ", Qm)
d = 0.3
R = 0.15
V1 = phi(-d-R,0)
V2 = phi(d-R,0)
print("Object 1 voltage: ", V1)
print("Object 2 voltage: ", V2)
print("Voltage difference: ", V2-V1)

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
