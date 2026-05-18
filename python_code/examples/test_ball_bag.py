# Need to restart kernel if changes done in the .py importations!!

import sys,os
notebook_dir = os.getcwd()  
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
print(parent_dir)
sys.path.append(parent_dir)
# from implementation.Cloth import Cloth 
from implementation.Cloth_Ball import Cloth # need to call new code, which is in a new file!
from implementation.utils import createRectangularMesh, duplicate_node_pairs
import time
import numpy as np

def create_rect_mesh(a=1, b=1): # the difference between a and b cannot be double!!
  # Dimensions need to be feasible to make two meshes join
    # a cannot be smaller than b and take into account how curved the mesh is with h param.
    # a is the width, so needs to be long enough to make edges join (at least a = 1)

    # need to form squares more or less depending on a and b!!
    # tried na = 20 first, but simulation very slow so needed to reduce number of nodes (less precise but no worries)!!
    na = int(a/0.1) # 1/10 = 0.1 cm/node -> so for a : a/0.1= 1cm/0.1 = 10 nodes
    nb = int(b/0.1) # then, same for b: b/0.1 to form squares (but integer, no decimals)

    print('Horizontal nodes:', na)
    print('Vertical nodes:', nb)

    np.random.seed(10)
        
    X, T = createRectangularMesh(a=a, b=b, na=na, nb=nb, h=0.80)

    return X, T, na, nb

# Define mesh with progressive flattening along vertical (along b)

X, T, na, nb = create_rect_mesh()

# Exponential Decay Function
X[:,2] = (1 - np.exp(3*(X[:,1]-0.5)))*X[:,2] # only works for b = 1 length

# The substracting element b/2 is the center point of the mesh, 
# so need to define exp. function around this center in order to make both meshes join at the bottom edge
# If center point wrong, the bottom edges don't match and flattening will happen around center at that element
# The value that multiplies in the exp() controls how fast curve changes, so smaller = flatter
# and the 1-... makes it decaying
# b = 1.6
# X[:,2] = (1 - np.exp(0.3*(X[:,1]-b/2)))*X[:,2]

print('Shape for X vecor:', X.shape)

# Make copy of the initial mesh to form the to halfs and join them with the seams

Y = X.copy(); # duplication of mesh

# since we rotated the initial mesh by 90º to have it vertical,
# the points that need to be inversed are now in Z-axis in position 1 
# (not in position 2 as original one w/o rotating)

Y[:,2] = -Y[:,2] # rotate initial mesh by 180º to be opposite to initial one
                # to have () meshes, not (( meshes, so edges join with seams

# join all nodes positions into unique mesh, although two different cloth pieces will be there before joined
X = np.concatenate([X,Y]); T = np.concatenate([T,T+na*nb])


# Rotate initial rectangular mesh -90° around X-axis
theta = -np.pi / 2
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])
X = X @ Rx.T

seam = duplicate_node_pairs(X)
print('Shape of the custom seams matrix:', seam.shape)

X[:,2] += 0.80

clothilde = Cloth(X, T, seam); 
# clothilde.plotMesh()

# initialize ball inside Cloth class by caling method inside cloth
clothilde.ball = clothilde.addBall(position=[-0.056,+0.05,1.7], rad=0.04, mass=0.2, friction=0.2)

# solver parameters
dt = 1/60 #frame rate
tol = 0.0095 # up to 0.75% of relative error in constraint satisfaction to stop iterations

clothilde.preparePolyscope()
clothilde.ball.plotMesh()

#physical parameters
rho = 0.1 #cloth density
delta = 0.08 # aerodynamics parameter: between 0 and rho
kappa = 0.15*1e-4 # stifness or bending resistance
alpha = 0.2 #damping of oscillations
shr = 1.2*1e-4 #allowed shearing resistance
strh = 0.005*1e-4 #allowed stretching resistance
mu_f = 0.45 #friction with the floor
mu_s = 0.4 #friction with the cloth itself
thck = 0.9 #size of the balls
sub_steps = 12 #numer of intermidate steps between each dt

clothilde.setSimulatorParameters(dt=dt,tol=tol,sub_steps = sub_steps,
                                rho=rho,delta=delta,kappa=kappa,shr=shr,
                                str=strh,alpha=alpha,mu_f=mu_f,mu_s=mu_s,
                                thck=thck)

clothilde.restart()

# define the control line for each half of the bag -> take 5 centered nodes for each
first_mesh = list(range(2, 8)) # 1st mesh nodes start at 0, so take from 2 to 7 to have 5 centered nodes
second_mesh = list(range(na*nb + 2, na*nb + 8)) # 2nd mesh nodes start at end of 1rst mesh node indexes
inds_ctr = first_mesh + second_mesh

# only grasp it and stabilize it
u = clothilde.positions[inds_ctr]
for _ in range(200):
    clothilde.simulate(u=u, control=inds_ctr)

clothilde.makeMovie(1,True,2) # True/False to repeat simulation when finishes