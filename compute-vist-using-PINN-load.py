# taken from vist-diffusion-pinn-5200-half-channel-plus-units-load-skip-5-cells.py

# In thie script the turbulent viscosity in the k equation is computed using PINN (solving the ODE for
# the k eq.)


import numpy as np
from numpy import linalg as LA
import math as m
import os
import sys
from matplotlib.image import imread
import matplotlib.pyplot as plt
from matplotlib import rcParams # for changing default values
import scipy.io as sio
from scipy.optimize import minimize
import timeit
import torch
import random
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
from torch.autograd import grad
import torch.optim.lr_scheduler as lr_scheduler

plt.close('all')
plt.interactive(True)
# set all fontsizes to 16
rcParams["font.size"] = 16

viscos = 1/100

# the equation reads  nu*k_yy + 2 = 0
# b.c. k(0)=1; k(1) = 0
# integrate the eq.  k_y = (-2y +C1)/nu
# integrate once more k = -y**2/nu +C1*y/nu + C2/nu
# b.c. atr y=0 gives C2=0
# b.c. atr y=1 gives C1=1
#
# k =  (-y**2 +y)/nu


# solve differential equation for k

# the grid
nj = 10
y  = np.linspace(0,1,10)
k = (-y**2 +y)/viscos # vist = const
vist = np.ones(nj)*viscos

dkdy_grad = np.gradient(k,y,edge_order = 2)
d2kdy2_grad = np.gradient(dkdy_grad,y,edge_order = 2)
dkdy= (-2*y +1)/viscos
d2kdy2= -2/viscos


# source 
b = 2*np.ones(nj)

k = torch.tensor(k, requires_grad=False, dtype=torch.float32).view((-1, 1))
d2kdy2 = torch.tensor(d2kdy2_grad, requires_grad=False, dtype=torch.float32).view((-1, 1))
dkdy = torch.tensor(dkdy_grad, requires_grad=False, dtype=torch.float32).view((-1, 1))
vist = torch.tensor(vist, requires_grad=False, dtype=torch.float32).view((-1, 1))
y = torch.tensor(y, requires_grad=True, dtype=torch.float32).view((-1, 1))
b = torch.tensor(b, requires_grad=False, dtype=torch.float32).view((-1, 1))



# b.c.
vist_0 = viscos
vist_1 = viscos

x = y

# Define get_derivative
dtype = torch.float
device = torch.device("cpu")
def get_derivative(f, y):
    """Compute the nth order derivative of y = f(x) with respect to x."""
    df_dy = grad(f, y, torch.ones(y.size()[0], 1, device=device), create_graph=True)[0]
    return df_dy

class MyNet2(nn.Module):
  def __init__(self):
    super().__init__()
    self.ll1 = nn.Linear(in_features=1,out_features=10)
    self.tanh = nn.Tanh()
    self.ll2 = nn.Linear(in_features=10,out_features=10)
    self.ll3 = nn.Linear(in_features=10,out_features=10)
    self.output = nn.Linear(in_features=10,out_features=1)

  def forward(self,x):
#       print('self',self)
        out = self.ll1(x)
        out = self.tanh(out)
        out = self.ll2(out)
        out = self.tanh(out)
        out = self.ll3(out)
        out = self.output(out)
        return out


    
# Create an instance 
model = MyNet2()

#%% Define loss function
def PDE(y, vist_pred):
        """Compute the cost function."""
        # Differential equation loss
        dvist_dy = get_derivative(vist_pred,y)  
        differential_equation_loss = vist_pred * d2kdy2 + dkdy*dvist_dy + b

        boundary_condition_loss = 0
        imbalance = differential_equation_loss
        differential_equation_loss = torch.sum(differential_equation_loss ** 2)
        # Boundary condition loss initialization
        boundary_condition_loss = 0
        # Sum over dirichlet boundary condition losses
        boundary_condition_loss += (vist_pred[0] - vist_0) ** 2
        boundary_condition_loss += (vist_pred[-1] - vist_1) ** 2
        
        return differential_equation_loss, boundary_condition_loss, imbalance

def loss_and_PDE(y_tensor):
    optimizer.zero_grad() # Clear gradients from the previous iteration
    outputs = model(y_tensor)  #get k 
    loss_de,loss_bc, imbalance = PDE(y_tensor, outputs) # Compute the loss
    loss = loss_de+1000.*loss_bc
# Calculate the L1 regularization term
    l1_regularization = torch.tensor(0.)
    for param in model.parameters():
        l1_regularization += torch.norm(param, p=1)

    # Add the L1 regularization term to the loss
    lambda_l1=0.
    loss += lambda_l1 * l1_regularization # Compute the loss
    loss.backward() # Compute gradients using backpropagation
    return loss,loss_de,loss_bc, imbalance

#%% training
max_no_epoch=100000
max_no_epoch=2000
max_no_epoch=20000
#max_no_epoch=4

learning_rate = 0.1  # 
#learning_rate = 0.5  #  loss = 2
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#saving training result
differential_equation_loss_history = np.zeros(max_no_epoch)
boundary_condition_loss_history = np.zeros(max_no_epoch)
loss_min = 1e30
# Training loop
for epoch in range(max_no_epoch):
    loss,loss_de,loss_bc, imbalance = loss_and_PDE(x)
    differential_equation_loss_history[epoch] += loss_de
    boundary_condition_loss_history[epoch] += loss_bc

# Define checkpoint
    if epoch == 0:
       checkpoint = torch.load('checkpoint-poisson-1D-save.ct',weights_only=False)

# Apply the state_dict to model and optimizer
       model = MyNet2()  # Initialize model; Ensure it's the same architecture
       model.load_state_dict(checkpoint['model_state_dict'])

       optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Initialize optimizer; Ensure it's the same optimizer type
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# change learning rata at the milestoned below
       scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[2400,4000,8000,12000,16000], gamma=0.5)

# Retrieve the training epoch
       epoch = checkpoint['epoch']
       loss = checkpoint['loss']

       model.train()  # For training mode (resuming training)

    optimizer.step()
    scheduler.step()

    loss_np = loss.detach().numpy()

# Print the loss every epoch
    loss_min = np.minimum(loss_np,loss_min)
    torch.set_printoptions(precision=4)
    print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]}, Loss: {loss_np}, Loss_min: {loss_min}")

    if loss_np < 5e-5:
        vist_pred = model(x)
        vist_pred_np =  vist_pred.detach().numpy()[:,0]

        print('break')

        break

# Plot loss_function
fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
ax.semilogy(np.arange(len(boundary_condition_loss_history)), boundary_condition_loss_history,color='red', label='bc error')
ax.semilogy(np.arange(len(boundary_condition_loss_history)), differential_equation_loss_history,color='blue',label="diff eq error")
plt.xlabel(r'epochs')
ax.set_title(r'Errors')
ax.grid(visible=True)
ax.legend(loc='best') 
plt.savefig('loss-vist-poisson-1D-load.png',bbox_inches='tight')

######################## plot vist
fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
plt.subplots_adjust(left=0.20,bottom=0.20)
vist_np =vist.detach().numpy()[:,0]
y_np =y.detach().numpy()[:,0]
ax.plot(y_np, vist_np,'ro',label='CFD')
vist_pred = model(x)  #get k
vist_pred_np =  vist_pred.detach().numpy()[:,0]
ax.plot(y_np, vist_pred_np,color='k',linestyle='-',linewidth=2, label='PINN')
ax.legend(loc='best') 
plt.xlabel('$y$')
plt.ylabel(r'$\nu$')   
ax.grid(visible=True)
plt.savefig('vist-poisson-1D-load.png',bbox_inches='tight')

for i in range(0,len(vist_pred_np)):
   print(f"vist_t,PINN: {vist_pred_np[i]:.5e} at node {i}")



######################## plot diffusion term
fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
plt.subplots_adjust(left=0.20,bottom=0.20)
dkdy_np =  dkdy.detach().numpy()[:,0]
y=  y.detach().numpy()[:,0]
term = dkdy_np*vist_pred_np
dvist_dy = get_derivative( vist_pred,x)  
dy = y[3]-y[2]
diff_non_conserv = vist_pred * d2kdy2 + dkdy*dvist_dy + b
diff_DNS_pred = np.gradient(term,y)
plt.plot(y, diff_DNS_pred,color='k',linestyle='-',linewidth=2, label=r"predicted")
d2kdy2_np=  d2kdy2.detach().numpy()[:,0]

diff_analyt = vist_np*d2kdy2_np
plt.plot(y, diff_analyt,color='b',linestyle='-',linewidth=2, label=r"Analytical")
ax.plot(y, imbalance.detach().numpy(),color='r',linestyle=':',linewidth=5, label='imbalance')
ax.legend(loc='best') 
plt.xlabel('$y^+$')
plt.ylabel('diffusion')   
ax.grid(visible=True)
plt.savefig('diffusion-poisson-1D-load.png',bbox_inches='tight')
