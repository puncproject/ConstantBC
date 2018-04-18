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
from ConstantBC import ConstantBC

iterative_solver    = True
monitor_convergence = True
monitor_bc          = False
compiled_apply      = True
external_mesh       = True # NB: There's some trouble with mshr.
store_to_file       = False
order               = 1
resolution          = 4

rho = Constant(0.0)
# rho = Expression("100*x[0]", degree=2)

gamma_e_id = 58
gamma_i_id = 59
ri = 0.2
ro = 1.0
Q = Constant(7.)
EPS = 1e-3

if external_mesh:
    print("Loading external mesh")
    fname = "mesh/sphere_in_sphere_res"+str(resolution)
    mesh = Mesh(fname+".xml")
    bnd = MeshFunction("size_t", mesh, fname+"_facet_region.xml")

else:
    print("Generating mesh using mshr")
    domain = Sphere(Point(0,0,0), ro, 50) - Sphere(Point(0,0,0), ri, 20)
    mesh = generate_mesh(domain, 40)

    gamma_e = AutoSubDomain(lambda x, on_bnd: norm(x)>=ro-EPS and on_bnd)
    gamma_i = AutoSubDomain(lambda x, on_bnd: norm(x)<=ri+EPS and on_bnd)

    bnd = MeshFunction('size_t', mesh, 2)
    bnd.set_all(0)
    gamma_e.mark(bnd, gamma_e_id)
    gamma_i.mark(bnd, gamma_i_id)

print("Making spaces")
cell = mesh.ufl_cell()
VE = FiniteElement("Lagrange", cell, order)
RE = FiniteElement("Real", cell, 0)
W = FunctionSpace(mesh, VE)
R = FunctionSpace(mesh, RE)

phi = TrialFunction(W)
lamb = TrialFunction(R)
psi = TestFunction(W)
mu = TestFunction(R)

bc_e = DirichletBC(W, Constant(0), bnd, gamma_e_id)
bc_i = ConstantBC(W, bnd, gamma_i_id, compiled_apply=compiled_apply)
bc_i.monitor(monitor_bc)

dss = Measure("ds", domain=mesh, subdomain_data=bnd)
ds_i = dss(gamma_i_id)

print("Creating variational form")

S = assemble(1.*ds_i)
n = FacetNormal(mesh)

lhs = dot(grad(phi), grad(psi)) * dx
rhs = rho*psi * dx

a0 = inner(mu, dot(grad(phi), n))*ds_i

A0 = assemble(a0)

print("Assembling matrix")
# Do not use assemble_system()
A = assemble(lhs)
b = assemble(rhs)

print("Applying boundary conditions")
bc_e.apply(A, b)
bc_i.apply(A, b)

wh = Function(W)

print("Setting final dof on boundary to enforce inner(dot(grad(u), n))*dsi = Q")
int_bc = DirichletBC(W, 2, bnd, gamma_i_id)
ww = Function(W)
int_bc.apply(ww.vector())
bnd_dof = list(bc_i.get_boundary_values().keys())[0]
row, col = A0.getrow(0)
code = open('addrow.cpp', 'r').read()
compiled = compile_extension_module(code=code)
B = Matrix()
compiled.addrow(A, B, A0, bnd_dof, W)
b[bnd_dof] = Q(0)

if iterative_solver:
    print("Solving equation using iterative solver")
    solver = PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10 #e-12
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = monitor_convergence

    solver.set_operator(B)
    solver.solve(wh.vector(), b)
else:
    print("Solving equation using direct solver")
    solve(B, wh.vector(), b)

uh = wh

print("Computing actual object charge")
Qm = assemble(dot(grad(uh), n) * ds_i)
print("Object charge: ", Qm)

print("Making plots")
line = np.linspace(ri,ro,10000, endpoint=False)
uh_line = np.array([uh(x,0,0) for x in line])
ue_line = (Q.values()[0]/(4*np.pi))*(line**(-1)-ro**(-1))

dr = line[1]-line[0]
e_abs = np.sqrt(dr*np.sum((uh_line-ue_line)**2))
e_rel1 = e_abs/np.sqrt(dr*np.sum(ue_line**2))
e_rel2 = np.sqrt(dr*np.sum(((uh_line-ue_line)/ue_line)**2))
sum1 = np.sum(uh_line**2)
sum2 = np.sum(ue_line**2)
sum3 = np.sum(uh_line*ue_line)
sum4 = np.sum((uh_line/ue_line)**2)
sum5 = np.sum(uh_line/ue_line)
print(e_abs)
hmin = mesh.hmin()
hmax = mesh.hmax()

if store_to_file:
    with open("convergence.txt", "a") as myfile:
        myfile.write("%d %d %d %g %g %g %g %g %g %g %g %g %g %g\n"%(
                    resolution, order, len(line), e_abs, e_rel1, e_rel2,
                    hmin, hmax, dr, sum1, sum2, sum3, sum4, sum5))

print(resolution, order, e_abs, e_rel1, e_rel2, hmin, hmax, dr, len(line),
      sum1, sum2, sum3, sum4, sum5)

plt.plot(line, uh_line, label='Numerical')
plt.plot(line, ue_line, '--', label='Exact')
plt.legend(loc='lower left')
plt.show()

print("Storing to file")
File("phi.pvd") << uh
