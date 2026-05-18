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


na = 30; nb = 18
X, T = createRectangularMesh(a = 0.9,b = 0.5, na = na, nb = nb, h = 0.15)
X[:,2] += 0.4; #adjust height

clothilde = Cloth(X, T)


clothilde.ball = clothilde.addBall(position=[0,0.05,0.5], rad=0.02, mass=0.3, friction=0.3)

print(f'Ball position: {clothilde.ball.position}')   # position
print(f'Ball speed: {clothilde.ball.velocity}')      # velocity
print(f'Ball radius: {clothilde.ball.rad}')    # radius
print(f'Ball friction: {clothilde.ball.mu_b}')     # friction
print(f'Ball position history: {clothilde.ball.history_pos}')  # full trajectory


# solver parameters
dt = 1/60 #frame rate
tol = 0.009 # up to 0.75% of relative error in constraint satisfaction to stop iterations

#physical parameters
rho = 0.1 #cloth density
delta = 0.1 # aerodynamics parameter: between 0 and rho
kappa = 0.35*1e-4 # stifness or bending resistance
kappa_bnd = 0.015*1e-4 # stifness or bending resistance

alpha = 0.3 #damping of oscillations
shr = 1*1e-4 #allowed shearing resistance
strh = 0.001*1e-4 #allowed stretching resistance
mu_f = 0.45 #friction with the floor
mu_s = 0.4 #friction with the cloth itself
thck = 0.95 #size of the balls
sub_steps = 12 #number of intermidate steps between each dt

clothilde.setSimulatorParameters(dt=dt,tol=tol,sub_steps = sub_steps,
                                rho=rho,delta=delta,kappa=kappa,kappa_bnd=kappa_bnd,shr=shr,
                                str=strh,alpha=alpha,mu_f=mu_f,mu_s=mu_s,
                                thck=thck)

clothilde.preparePolyscope()
clothilde.ball.plotMesh()

inds_ctr = [0,na-1, (na*nb)-na, (na*nb)-1]
tf = int(6/dt)
u = X[inds_ctr]
for i in range(tf):
    clothilde.simulate(u = u, control = inds_ctr)

print('Average iterations',clothilde.total_iters/(len(clothilde.history_pos)-1))
clothilde.makeMovie(1,True,2) # True/False to repeat simulation when finishes