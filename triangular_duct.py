from fenics import *
from mshr import *
from math import log as ln
from math import fabs
import matplotlib.pyplot as plt


# INPUT PARAMETERS
a = 1
dpdx = Constant(-0.01)
mu = Constant(0.01)
set_log_active(False)    # FEniCS! Shut up!


# # ANALYTICAL SOLUTION (C++)
# ue_code = '''
# class U : public Expression
# {
#     public:
#         double a, mu, dpdx;
#
#     void eval(Array<double>& values, const Array<double>& x) const
#     {
#         values[0] = (-dpdx/(2*sqrt(3)*a*mu)) * (x[1]-0.5*a*sqrt(3)) * (3*pow(x[0],2)-pow(x[1],2));
#     }
# };'''
# u_c = Expression(ue_code, degree=1)
# u_c.a = float(a)
# u_c.mu = float(mu(0)); u_c.dpdx = float(dpdx(0))


# ANALYTICAL TOTAL VOLUME FLOW
Qe = ((a**4*sqrt(3))/(320*mu(0))) * (-dpdx(0))
print("Analytical Q: %2.4E"%Qe)

# NUMERICAL SOLUTION
for degree in range(1,4):

    # ORIGINAL MESH
    mesh = Mesh("triangle.xml")
    x = mesh.coordinates()
    x[:,:] *= a    # Scaling Gmsh-mesh 
    mesh.bounding_box_tree().build(mesh)
    n = 1
    E = []; h = []; Q = []; Eq = []

    print("degree=%d"%(degree))

    for i in range(6):

        if i==2 and degree==1:
            wiz = plot(mesh)
            plt.show()

        # REFERENCE SOLUTION
        V = FunctionSpace(mesh, 'CG', 5)
        u = TrialFunction(V)
        v = TestFunction(V)
        F = inner(grad(u), grad(v))*dx + 1/mu*dpdx*v*dx
        bc = DirichletBC(V, Constant(0), DomainBoundary())
        u_ref = Function(V)
        solve(lhs(F) == rhs(F), u_ref, bcs=bc)
        Q_ref = assemble(project(u_ref,V)*dx(mesh))

        # FUNCTION SPACES
        V = FunctionSpace(mesh, 'CG', degree)
        u = TrialFunction(V)
        v = TestFunction(V)

        # BOUNDARY VALUE PROBLEM
        F = inner(grad(u), grad(v))*dx + 1/mu*dpdx*v*dx
        bc = DirichletBC(V, Constant(0), DomainBoundary())

        # SOLVE
        u_ = Function(V)
        solve(lhs(F) == rhs(F), u_, bcs=bc)

#         # COMPARE WITH ANALYTICAL SOLUTION
#         u_e = interpolate(u_c, V)
#         bc.apply(u_e.vector())
#         u_error = errornorm(u_e, u_, degree_rise=0, mesh=mesh)
#         E.append(u_error)
#         h.append(mesh.hmin())

        # COMPARE WITH REFERENCE SOLUTION
        u_error = errornorm(u_ref, u_, degree_rise=0, mesh=mesh)
        E.append(u_error)
        h.append(mesh.hmin())

        # COMPUTE TOTAL VOLUME FLOW
        Qc = assemble(project(u_,V)*dx(mesh))
        Q.append(Qc)
        # Eq.append(fabs(Qc-Qe))
        Eq.append(fabs(Qc-Q_ref))

        #plot(mesh)
        #interactive()

        # REFINING MESH
        mesh = refine(mesh)


    # PLOT RESULTS
    for i in range(len(E)):
        if i==0:
            r = 0
            rq = 0
        else:
            r = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
            rq = ln(Eq[i]/Eq[i-1])/ln(h[i]/h[i-1])
        print("h=%2.2E E=%2.2E r=%.2f Q=%2.4E Eq=%2.2E rq=%.2f" %(h[i], E[i], r, Q[i], Eq[i], rq))
