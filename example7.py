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

monitor_convergence = False
orders              = [1,2]
resolutions         = [1,2,3,4,5]#,6]#,7,8]
allow_extrapolation = False

# rho = Expression("100*x[1]", degree=1)
rho = Constant(0)

fname = "mesh/pentagon_and_square_in_rectangle_res1"
# mesh, bnd, num_objects = load_mesh(fname)

domain = Circle(Point(0,0), 1.0, 100) \
       - Circle(Point(0,0), 0.2, 100)
mesh = generate_mesh(domain, 10)
num_objects = 1

class OuterBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return norm(x)>0.3 and on_bnd

class InnerBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return norm(x)<0.3 and on_bnd

outer_bnd  = OuterBoundary()
inner_bnd  = InnerBoundary()

hmins = []
errors = []
for resolution in resolutions:

    order = 5
    print("Order: {}, resolution: {} (reference)".format(order,resolution))

    bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
    bnd.set_all(0)
    outer_bnd.mark(bnd, 1)
    inner_bnd.mark(bnd, 2)

    W = FunctionSpace(mesh, "Lagrange", order)

    phi = TrialFunction(W)
    psi = TestFunction(W)

    bcs = [DirichletBC(W, Constant(i), bnd, i+1) for i in range(num_objects+1)]

    lhs = dot(grad(phi), grad(psi)) * dx
    rhs = rho*psi * dx

    A = assemble(lhs)
    b = assemble(rhs)

    for bc in bcs:
        bc.apply(A, b)

    phi_ref = Function(W)

    solver = PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10 #e-12
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = monitor_convergence

    solver.set_operator(A)
    solver.solve(phi_ref.vector(), b)

    # n = FacetNormal(mesh)
    # dss = Measure("ds", domain=mesh, subdomain_data=bnd)

    # charge1 = assemble(dot(grad(phi), n) * dss(2, degree=1))
    # charge2 = assemble(dot(grad(phi), n) * dss(3, degree=1))
    # total_charge = charge1 + charge2

    # vsources = [[0,1,1]]
    # vsources = [[-1,0,1],[-1,1,2]]
    # vsources = []

    File('phi.pvd') << phi_ref

    hmins.append(mesh.hmin())
    errors.append([])
    for order in orders:

        print("Order: {}, resolution: {}".format(order,resolution))

        V = FunctionSpace(mesh, "Lagrange", order)

        phi = TrialFunction(V)
        psi = TestFunction(V)

        # bc_e = DirichletBC(V, Constant(0), bnd, 1)
        # objects = [ObjectBC(V, bnd, 2+i) for i in range(num_objects)]
        # circuit = Circuit(V, bnd, objects, vsources)
        objects = [DirichletBC(V, Constant(i), bnd, i+1) for i in range(num_objects+1)]

        # objects[0].charge = charge1
        # objects[1].charge = charge2

        lhs = dot(grad(phi), grad(psi)) * dx
        rhs = rho*psi * dx

        # Do not use assemble_system()
        A = assemble(lhs)
        b = assemble(rhs)

        # bc_e.apply(A, b)
        # A, b = circuit.apply(A, b)
        for o in objects:
            o.apply(A, b)

        phi = Function(V)
        # phi.set_allow_extrapolation(allow_extrapolation)

        solver = PETScKrylovSolver('gmres','hypre_amg')
        solver.parameters['absolute_tolerance'] = 1e-14
        solver.parameters['relative_tolerance'] = 1e-10 #e-12
        solver.parameters['maximum_iterations'] = 100000
        solver.parameters['monitor_convergence'] = monitor_convergence

        solver.set_operator(A)
        solver.solve(phi.vector(), b)

        error = errornorm(phi_ref, phi, degree_rise=1)
        # error = np.sqrt(assemble((project(phi, W, bcs=objects)-phi_ref)**2*dx(mesh)))
        # error = np.sqrt(assemble((phi-project(phi_ref, V, bcs=objects))**2*dx(mesh)))
        errors[-1].append(error)

    mesh = refine(mesh)

for order in orders:
    x = np.array(hmins)
    y = np.array(errors)[:,order-1]
    p = plt.loglog(x, y, '-o', label=order)

    color = p[0].get_color()
    plt.loglog(x, y[0]*(x/x[0])**order, ':', color=color)

plt.grid()
plt.xlabel('hmin')
plt.ylabel('L_2 norm error')
plt.title('Convergence')
plt.legend(loc='lower right')
plt.show()
