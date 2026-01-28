import numpy as np
from fematiso import fematiso
from feeldof import feeldof
from fekine2d import fekine2d
from feasmbl1 import feasmbl1
from feaplyc2 import feaplyc2

# ---------------------------------------------------------
# Example 9.7.1
# plane stress analysis of a solid using linear triangular elements
# (see Fig. 9.7.1 for the finite element mesh)
#
# Variable descriptions
# k = element matrix
# f = element vector
# kk = system matrix
# ff = system vector
# disp = system nodal displacement vector
# eldisp = element nodal displacement vector
# stress = matrix containing stresses
# strain = matrix containing strains
# gcoord = coordinate values of each node
# nodes = nodal connectivity of each element
# index = a vector containing system dofs associated with each element
# bcdof = a vector containing dofs associated with boundary conditions
# bcval = a vector containing boundary condition values associated with
# the dofs in bcdof
# ---------------------------------------------------------

# input data for control parameters
nel = 8  # number of elements
nnel = 3  # number of nodes per element
ndof = 2  # number of dofs per node
nnode = 10  # total number of nodes in system
sdof = nnode * ndof  # total system dofs
edof = nnel * ndof  # degrees of freedom per element
emodule = 100000.0  # elastic modulus
poisson = 0.3  # Poisson's ratio

# input data for nodal coordinate values
# gcoord[i,j] where i-> node no. and j-> x or y
# Note: In Python, indexing starts at 0, so node 1 is at index 0
gcoord = np.array([
    [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0],
    [2.0, 1.0], [3.0, 0.0], [3.0, 1.0], [4.0, 0.0], [4.0, 1.0]
])

# input data for nodal connectivity for each element
# nodes[i,j] where i-> element no. and j-> connected nodes
# Note: Converting from 1-based (MATLAB) to 0-based (Python) indexing
nodes = np.array([
    [0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3],
    [4, 6, 7], [4, 7, 5], [6, 8, 9], [6, 9, 7]
], dtype=int)

# input data for boundary conditions
# Converting from 1-based to 0-based indexing
bcdof = np.array([0, 1, 2], dtype=int)
bcval = np.array([0.0, 0.0, 0.0])

# first three dofs are constrained
# whose described values are 0

# initialization of matrices and vectors
ff = np.zeros((sdof, 1))  # system force vector
kk = np.zeros((sdof, sdof))  # system matrix
disp = np.zeros((sdof, 1))  # system displacement vector
eldisp = np.zeros((edof, 1))  # element displacement vector
stress = np.zeros((nel, 3))  # matrix containing stress components
strain = np.zeros((nel, 3))  # matrix containing strain components
index = np.zeros(edof, dtype=int)  # index vector
kinmtx = np.zeros((3, edof))  # kinematic matrix
matmtx = np.zeros((3, 3))  # constitutive matrix

# force vector
# Converting from 1-based to 0-based indexing: 17 -> 16, 19 -> 18
ff[16, 0] = 500  # force applied at node 9 in x-axis
ff[18, 0] = 500  # force applied at node 10 in x-axis

# compute element matrices and vectors, and assemble
matmtx = fematiso(1, emodule, poisson)  # constitutive matrix

# DEBUG
print("matmtx =")
print(matmtx)

for iel in range(nel):  # loop for the total number of elements
    nd = np.zeros(3, dtype=int)
    nd[0] = nodes[iel, 0]  # 1st connected node for (iel)-th element
    nd[1] = nodes[iel, 1]  # 2nd connected node for (iel)-th element
    nd[2] = nodes[iel, 2]  # 3rd connected node for (iel)-th element
    
    x1 = gcoord[nd[0], 0]
    y1 = gcoord[nd[0], 1]  # coord values of 1st node
    x2 = gcoord[nd[1], 0]
    y2 = gcoord[nd[1], 1]  # coord values of 2nd node
    x3 = gcoord[nd[2], 0]
    y3 = gcoord[nd[2], 1]  # coord values of 3rd node
    
    index = feeldof(nd, nnel, ndof)  # extract system dofs for the element
    
    # find the derivatives of shape functions
    area = 0.5 * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)  # area of triangle
    area2 = area * 2
    dhdx = (1/area2) * np.array([y2-y3, y3-y1, y1-y2])  # derivatives w.r.t. x
    dhdy = (1/area2) * np.array([x3-x2, x1-x3, x2-x1])  # derivatives w.r.t. y
    
    kinmtx2 = fekine2d(nnel, dhdx, dhdy)  # kinematic matrix
    
    k = kinmtx2.T @ matmtx @ kinmtx2 * area  # element stiffness matrix
    
    kk = feasmbl1(kk, k, index)  # assemble element matrices

# apply boundary conditions
kk, ff = feaplyc2(kk, ff, bcdof, bcval)

# DEBUG
# print('System Stiffness Matrix:')
# # Print columns 0-6
# print('Columns 0 through 6')
np.set_printoptions(linewidth=200, suppress=False, precision=4)
# print(kk[:, 0:7])
# # Print columns 7-13
# print('\nColumns 7 through 13')
# print(kk[:, 7:14])
# # Print columns 14-19
# print('\nColumns 14 through 19')
# print(kk[:, 14:20])
# print('System Force Vector:')
# print(ff)


# solve the matrix equation
disp = np.linalg.solve(kk, ff)

# element stress computation (post computation)
for ielp in range(nel):  # loop for the total number of elements
    nd = np.zeros(3, dtype=int)
    nd[0] = nodes[ielp, 0]  # 1st connected node for (iel)-th element
    nd[1] = nodes[ielp, 1]  # 2nd connected node for (iel)-th element
    nd[2] = nodes[ielp, 2]  # 3rd connected node for (iel)-th element
    
    x1 = gcoord[nd[0], 0]
    y1 = gcoord[nd[0], 1]  # coord values of 1st node
    x2 = gcoord[nd[1], 0]
    y2 = gcoord[nd[1], 1]  # coord values of 2nd node
    x3 = gcoord[nd[2], 0]
    y3 = gcoord[nd[2], 1]  # coord values of 3rd node
    
    index = feeldof(nd, nnel, ndof)  # extract system dofs for the element
    
    # extract element displacement vector
    for i in range(edof):
        eldisp[i, 0] = disp[index[i], 0]
    
    area = 0.5 * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)  # area of triangle
    area2 = area * 2
    dhdx = (1/area2) * np.array([y2-y3, y3-y1, y1-y2])  # derivatives w.r.t. x
    dhdy = (1/area2) * np.array([x3-x2, x1-x3, x2-x1])  # derivatives w.r.t. y
    
    kinmtx2 = fekine2d(nnel, dhdx, dhdy)  # kinematic matrix
    
    # DEBUG: force double precision
    eldisp = eldisp.astype(np.float64)
    kinmtx2 = kinmtx2.astype(np.float64)
    
    estrain = kinmtx2 @ eldisp  # compute strains
    estress = matmtx @ estrain  # compute stresses
    
    for i in range(3):
        strain[ielp, i] = estrain[i, 0]  # store for each element
        stress[ielp, i] = estress[i, 0]  # store for each element
    
    # DEBUG
    print(f"Element {ielp+1}")
    print(f"nd = {nd} index = {index}")
    print(f"(x1,y1)=({x1},{y1}) (x2,y2)=({x2},{y2}) (x3,y3)=({x3},{y3})")
    print(f"eldisp = {eldisp.T}")
    print(f"eldisp dtype = {eldisp.dtype}, shape = {eldisp.shape}")
    print(f"Area = {area} dhdx = {dhdx} dhdy = {dhdy}")
    print(f"kinmtx2 dtype = {kinmtx2.dtype}, shape = {kinmtx2.shape}")
    print(f"kinmtx2 = \n{kinmtx2}")
    print(f"estrain = {estrain.T}")
    print(f"estrain dtype = {estrain.dtype}, valores individuais:")
    print(f"  estrain[0,0] = {estrain[0,0]:.17e}")
    print(f"  estrain[1,0] = {estrain[1,0]:.17e}")
    print(f"  estrain[2,0] = {estrain[2,0]:.17e}")
    print(f"estress = {estress.T}")
    print(f"estress dtype = {estress.dtype}, valores individuais:")
    print(f"  estress[0,0] = {estress[0,0]:.17e}")
    print(f"  estress[1,0] = {estress[1,0]:.17e}")
    print(f"  estress[2,0] = {estress[2,0]:.17e}\n")

# print fem solutions
num = np.arange(1, sdof + 1)
displace = np.column_stack((num, disp))
print("Nodal displacements:")
print(displace)
print("\nElement stresses:")
for i in range(nel):
    print(f"{i+1} {stress[i, :]}")
