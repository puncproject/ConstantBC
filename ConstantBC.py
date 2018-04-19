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

import dolfin as df
import numpy as np
import copy

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
        self.monitor = False
        self.compiled_apply = kwargs.pop('compiled_apply', True)
        if self.compiled_apply:
            code = open('apply.cpp', 'r').read()
            self.compiled_apply = df.compile_extension_module(code=code)

        df.DirichletBC.__init__(self, *args, **kwargs)

    def apply(self, *args):

        for A in args:

            if isinstance(A, df.GenericVector):
                # Applying to load vector.
                # Set all elements to zero but leave the first.

                ind = self.get_boundary_values().keys()
                first_ind = list(ind)[0]
                first_element = A[first_ind][0]

                df.DirichletBC.apply(self, A)

                A[first_ind] = first_element

            else:
                # Applying to stiffness matrix.
                # Leave the first row on the boundary node, but change the
                # remaining to be the average of it's neighbors also on the
                # boundary.

                ind = self.get_boundary_values().keys()
                if self.compiled_apply:
                    self.compiled_apply.apply(A, np.array(list(ind), dtype=np.intc))

                else:
                    length = len(list(ind))-2
                    allneighbors = []
                    inda = np.array(list(ind), dtype=np.intc)
                    for it, i in enumerate(inda[1:]):
                        allneighbors.append(A.getrow(i)[0])
                    zero_rows = np.array(inda[1:], dtype=np.intc)
                    A.zero(zero_rows)

                    for it, i in enumerate(inda[1:]):
                        if self.monitor:
                            print("ConstantBC iteration", it, "of", length)
                        neighbors = allneighbors[it]
                        surface_neighbors = np.array([n for n in neighbors if n in ind])
                        values = -np.ones(surface_neighbors.shape)
                        self_index = np.where(surface_neighbors==i)[0][0]
                        num_of_neighbors = len(surface_neighbors)-1
                        values[self_index] = num_of_neighbors
                        A.setrow(i, surface_neighbors, values)
                    A.apply('insert')

class CircuitConstraints(object):

    def __init__(objects, vsources, isources):
        pass

    def apply(self, *args):
        for A in args:
            if isinstance(A, df.GenericVector):
                self.apply_to_vector(A)
            else:
                self.apply_to_matrix(A)

    def apply_to_matrix(self, A):
        pass

def relabel_bnd(bnd):
    """
    Relabels MeshFunction bnd such that boundaries are marked 1, 2, 3, etc.
    instead of arbitrary numbers. The order is preserved, and by convention the
    first boundary is the exterior boundary. The objects start at 2. The
    background (not marked) is 0.
    """
    new_bnd = bnd
    new_bnd = df.MeshFunction("size_t", bnd.mesh(), bnd.dim())
    new_bnd.set_all(0)

    old_ids = np.array([int(tag) for tag in set(bnd.array())])
    old_ids = np.sort(old_ids)[1:]
    for new_id, old_id in enumerate(old_ids, 1):
        new_bnd.array()[bnd.where_equal(old_id)] = int(new_id)

    num_objects = len(old_ids)-1
    return new_bnd, num_objects 

def load_mesh(fname):
    mesh = df.Mesh(fname+".xml")
    bnd  = df.MeshFunction("size_t", mesh, fname+"_facet_region.xml")
    bnd, num_objects = relabel_bnd(bnd)
    return mesh, bnd, num_objects

def get_charge_sharing_set(vsources, node, group):

    group.append(node)

    i = 0
    while i < len(vsources):
        vsource = vsources[i]
        if vsource[0] == node:
            vsources.pop(i)
            get_charge_sharing_set(vsources, vsource[1], group)
        elif vsource[1] == node:
            vsources.pop(i)
            get_charge_sharing_set(vsources, vsource[0], group)
        else:
            i += 1

def get_charge_sharing_sets(vsources, num_objects):

    vsources = copy.deepcopy(vsources)
    nodes = set(range(num_objects))

    groups = []
    while vsources != []:
        group = []
        get_charge_sharing_set(vsources, vsources[0][0], group)
        groups.append(group)

    for group in groups:
        for node in group:
            if node != -1:
                nodes.remove(node)

    groups = list(filter(lambda group: -1 not in group, groups))

    for node in nodes:
        groups.append([node])

    return groups

