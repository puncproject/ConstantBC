import dolfin as df
import numpy as np

class ConstantBC(df.DirichletBC):
    """
    Enforces a constant but unknown boundary. The (single) unknown value at the
    boundary must be determined from the variational formaulation, typically by
    means of a Lagrange multiplier. See examples in the demos.

    Tested for 1st and 2nd order Lagrange finite elements but should in principe
    work for higher orders as well.

    This class is in most ways similar to Dolfin's own DirichletBC class, which
    it inherits. Its constructor is similar to DirichletBC's except that the
    second argument (the value on the boundary) must be omitted, e.g.:

        bc = ConstantBC(V, sub_domain)
        bc = ConstantBC(V, sub_domain, method)
        bc = ConstantBC(V, sub_domains, sub_domain)
        bc = ConstantBC(V, sub_domains, sub_domain, method)

    where sub_domain, sub_domains and method has the same meanings as for
    DirichletBC.
    """

    def __init__(self, *args, **kwargs):

        # Adds the missing argument (the value on the boundary) before calling
        # the parent constructor. The value must be zero to set the
        # corresponding elements in the load vector to zero.
        args = list(args)
        args.insert(1, df.Constant(0.0))
        monitor = False

        df.DirichletBC.__init__(self, *args, **kwargs)

    def monitor(self, monitor):
        self.monitor = monitor

    def apply(self, *args):

        for A in args:

            if isinstance(A, df.GenericVector):
                # Applying to load vectory

                df.DirichletBC.apply(self, A)

            else:
                # Applying to stiffness matrix

                ind = self.get_boundary_values().keys()

                length = len(list(ind))-2

                for it, i in enumerate(list(ind)[1:]):

                    if self.monitor:
                        print("ConstantBC iteration", it, "of", length)

                    neighbors = A.getrow(i)[0]
                    A.zero(np.array([i], dtype=np.intc))

                    surface_neighbors = np.array([n for n in neighbors if n in ind])
                    values = -np.ones(surface_neighbors.shape)

                    self_index = np.where(surface_neighbors==i)[0][0]
                    num_of_neighbors = len(surface_neighbors)-1
                    values[self_index] = num_of_neighbors

                    A.setrow(i, surface_neighbors, values)

                    A.apply('insert')
