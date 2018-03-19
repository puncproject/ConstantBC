from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ConstantBC import ConstantBC

mesh = Mesh("sphere_in_sphere.xml")
bnd = MeshFunction("size_t", mesh, "sphere_in_sphere_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59
ri = 0.2
ro = 1.0

Q = Constant(10.)

print("Making spaces")
cell = mesh.ufl_cell()
VE = FiniteElement("Lagrange", cell, 1)
RE = FiniteElement("Real", cell, 0)

V = FunctionSpace(mesh, VE)
W = FunctionSpace(mesh, MixedElement([VE, RE]))
u, c = TrialFunctions(W)
v, d = TestFunctions(W)

ext_bc = DirichletBC(W.sub(0), Constant(0), bnd, ext_bnd_id)
int_bc = ConstantBC(W.sub(0), bnd, int_bnd_id)

rho = Constant(0.0)
rho = Expression("100*x[1]", degree=2)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
dsi = dss(int_bnd_id)

print("Computing area");
S = assemble(Constant(1.)*dsi)

print("Creating variational form")
a = inner(grad(u), grad(v)) * dx -\
    inner(v, dot(grad(u), n)) * dsi +\
    inner(c, dot(grad(v), n)) * dsi +\
    inner(d, dot(grad(u), n)) * dsi

L = inner(rho, v) * dx +\
    inner(Q/S, d) * dsi

wh = Function(W)

print("Assembling matrix")
A = assemble(a)
b = assemble(L)

print("Applying boundary conditions")
ext_bc.apply(A)
ext_bc.apply(b)
int_bc.apply(A)
int_bc.apply(b)

print("Solving equation")

# solve(A, wh.vector(), b)

solver = PETScKrylovSolver('bicgstab','ilu')
solver.parameters['absolute_tolerance'] = 1e-14
solver.parameters['relative_tolerance'] = 1e-10 #e-12
solver.parameters['maximum_iterations'] = 100000
solver.parameters['monitor_convergence'] = True

solver.set_operator(A)
solver.solve(wh.vector(), b)

uh, ph = wh.split(deepcopy=True)

line = np.linspace(ri,ro,100)
uh_line = [uh(x,0,0) for x in line]
ue_line = (Q.values()[0]/(4*np.pi))*(line**(-1)-ro**(-1))
plt.plot(line, uh_line, label='Numerical')
plt.plot(line, ue_line, '--', label='Exact')
plt.legend(loc='lower left')
plt.show()

Qm = assemble(dot(grad(uh), n) * dsi)
print("Object charge: ", Qm)

print("Storing to file")
File("phi.pvd") << uh
