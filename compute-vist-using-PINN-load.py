# the following versio was used:
#
# sys.version 3.12.2 | packaged by conda-forge | (main, Feb 16 2024, 20:50:58) [GCC 12.3.0]
# torch.__version__ 2.5.1


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
from torch.optim.lr_scheduler import ReduceLROnPlateau

plt.close('all')
plt.interactive(True)
# set all fontsizes to 12
rcParams["font.size"] = 16
rcParams["axes.axisbelow"] = True # make sure grid is behind plots

viscos = 1/5200

# solve differential equation for k
# load DNS data
DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];
dudy_DNS=np.gradient(u_DNS,y_DNS)

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
u2_DNS=DNS_stress[:,2];
v2_DNS=DNS_stress[:,3];
w2_DNS=DNS_stress[:,4];
uv_DNS=DNS_stress[:,5];
k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)
dkdy_DNS=np.gradient(k_DNS,y_DNS,edge_order=2)
d2kdy2_DNS=np.gradient(dkdy_DNS,y_DNS,edge_order=2)

         #y/delta                    y^+                   Production          Turbulent_Transport        Viscous_Transport       Pressure_Strain         Pressure_Transport        Viscous_Dissipation           Balance
DNS_k_terms=np.genfromtxt("LM_Channel_5200_RSTE_k_prof.dat",comments="%")

diss_DNS=DNS_k_terms[:,7]
Pk_DNS=DNS_k_terms[:,2]
diff_DNS=DNS_k_terms[:,3]
diff_p_DNS=DNS_k_terms[:,6]
diff_DNS=diff_DNS + diff_p_DNS # this is the total turb. diff including the pressure transport

diff_DNS_visc =   DNS_k_terms[:,4]

diss_iso_DNS = np.maximum(diss_DNS-diff_DNS_visc,0)

diss_DNS=diss_DNS/viscos
Pk_DNS=Pk_DNS/viscos
diff_DNS=diff_DNS/viscos
diss_iso_DNS=diss_iso_DNS/viscos
diff_DNS_visc = diff_DNS_visc/viscos


vist_DNS = np.abs(uv_DNS/dudy_DNS)


# load k-omega grid
kom_data = np.loadtxt('y_u_k_om_uv_5200-RANS-half-channel.txt')
y_kom = kom_data[:,0]
k_kom = kom_data[:,2]
om_kom = kom_data[:,3]
vist_kom = k_kom/om_kom


nj = len(y_kom)

viscos_lam = np.ones(nj)*viscos

u_DNS = np.interp(y_kom, y_DNS, u_DNS)
u_DNS = torch.tensor(u_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

k_DNS = np.interp(y_kom, y_DNS, k_DNS)
k_DNS = torch.tensor(k_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

Pk_DNS = np.interp(y_kom, y_DNS, Pk_DNS)
Pk_DNS = torch.tensor(Pk_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

diss_DNS = np.interp(y_kom, y_DNS, diss_DNS)
diss_DNS = torch.tensor(diss_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

d2kdy2_DNS = np.interp(y_kom, y_DNS, d2kdy2_DNS)
d2kdy2_DNS = torch.tensor(d2kdy2_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

dkdy_DNS = np.interp(y_kom, y_DNS, dkdy_DNS)
dkdy_DNS = torch.tensor(dkdy_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

diff_DNS_visc = np.interp(y_kom, y_DNS, diff_DNS_visc)

diff_DNS = np.interp(y_kom, y_DNS, diff_DNS)
diff_DNS = torch.tensor(diff_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

vist_DNS = np.interp(y_kom, y_DNS, vist_DNS) + viscos
vist_DNS = torch.tensor(vist_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

viscos_lam = torch.tensor(viscos_lam, requires_grad=False, dtype=torch.float32).view((-1, 1))

yplus_DNS = np.interp(y_kom, y_DNS, yplus_DNS)
yplus_DNS = torch.tensor(yplus_DNS, requires_grad=False, dtype=torch.float32).view((-1, 1))

y_DNS = torch.tensor(y_kom, requires_grad=True, dtype=torch.float32).view((-1, 1))

# requires_grad=True).view((-1, 1))

vist_0 = 0
vist_1 = vist_DNS[-3]

print('vist_1/viscos',vist_1/viscos)



x = y_DNS


# Define get_derivative
#%% define a function to get_derivative
dtype = torch.float
device = torch.device("cpu")
def get_derivative(y, x):
    """Compute the nth order derivative of y = f(x) with respect to x."""
    dy_dx = grad(y, x, torch.ones(x.size()[0], 1, device=device), create_graph=True)[0]
    return dy_dx


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.layer_1 = nn.Linear(1, 30)
        self.layer_2 = nn.Linear(30, 30)
        self.layer_3 = nn.Linear(30, 30)
        self.layer_4 = nn.Linear(30, 30)
        self.layer_5 = nn.Linear(30, 1)
       #
    def forward(self, x):
        x = torch.nn.functional.sigmoid(self.layer_1(x)) 
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        x = torch.nn.functional.sigmoid(self.layer_3(x))
        x = torch.nn.functional.sigmoid(self.layer_4(x))
        x = torch.nn.functional.sigmoid(self.layer_5(x))

        return x


    
# Create an instance of the SinNet model
model = MyNet()
torch.manual_seed(5)
#model = MyNet(input_dim, hidden_dim, output_dim)

# Define loss function

#%% Define loss function
def PDE(y, vist_pred):
        """Compute the cost function."""
        global temp
        # Differential equation loss
        dvist_dy = get_derivative( vist_pred,y)  
        temp = (vist_pred+viscos_lam) * d2kdy2_DNS + dkdy_DNS*dvist_dy

        boundary_condition_loss = 0
# set dk/dy = 0 at x = L
#       boundary_condition_loss += (temp[-1]) ** 2
        differential_equation_loss = temp  + (Pk_DNS - diss_DNS)
        imbalance = differential_equation_loss
        differential_equation_loss = torch.sum(differential_equation_loss ** 2)
        # Boundary condition loss initialization
        boundary_condition_loss = 0
        # Sum over dirichlet boundary condition losses
        boundary_condition_loss += (vist_pred[0] - vist_0) ** 2
        boundary_condition_loss += (vist_pred[-1] - vist_1) ** 2
        
        return differential_equation_loss, boundary_condition_loss, imbalance

def loss_and_PDE(x_tensor):
    optimizer.zero_grad() # Clear gradients from the previous iteration
    vist_pred = model(x_tensor)  #get k 
    loss_de,loss_bc, imbalance = PDE(x_tensor, vist_pred) # Compute the loss
#   loss = loss_de+100.*loss_bc
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


# Training

# In[7]:


#%% training
prev_loss = float('inf')  # Initialize with a large value
tolerance = 1.e-7
max_no_epoch=7000
max_no_epoch=60000
max_no_epoch=20000
max_no_epoch=1000000
max_no_epoch=200000
#max_no_epoch=150000
#max_no_epoch=5

#seed=1024
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#np.random.seed(seed)


optim_alg='Adam'
optim_alg='SGD'
learning_rate = 0.01  #  Adam poch 200000, Learning Rate: 0.01, Loss: 2.53e+03, Loss_min: 2.5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#   optimizer=optim.SGD(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[3e4, 6e4, 8e5, 1e5, 1.3e5, 1.5e5], gamma=0.7)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=7500)




#for saving training result
differential_equation_loss_history = np.zeros(max_no_epoch)
loss_min_history = np.zeros(max_no_epoch)
boundary_condition_loss_history = np.zeros(max_no_epoch)
loss_min = 1e30
# Training loop
for epoch in range(max_no_epoch):
# Define checkpoint
    if epoch == 0:
       checkpoint = torch.load('vist-diffusion-pinn-5200-half-channel-save.ct',weights_only=False)

# Apply the state_dict to model and optimizer
       model = MyNet()  # Initialize model; Ensure it's the same architecture
       model.load_state_dict(checkpoint['model_state_dict'])

       optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Initialize optimizer; Ensure it's the same optimizer type
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[3e4, 6e4, 8e5, 1e5, 1.3e5, 1.5e5, 1.8e5, 2.0e5, 2.5e5,2.8e5], gamma=0.7)

       print('sys.version',sys.version)
       print('torch.__version__',torch.__version__)
       print('model=',model)
       for param in model.parameters():
         print('param.data)=',param.data)
       print('lr=',scheduler.get_last_lr()[0])


    loss,loss_de,loss_bc, imbalance = loss_and_PDE(x)
    differential_equation_loss_history[epoch] += loss_de
    boundary_condition_loss_history[epoch] += loss_bc
    optimizer.step()
    loss_change = prev_loss - loss
    prev_loss = loss
#   loss_m = torch.tensor(loss_min)
#   loss_min_history[epoch] += loss_m
#   scheduler.step(loss_m)
#   scheduler.step(loss)
    scheduler.step()
    loss_np = loss.detach().numpy()
# Print the loss every epoch
    loss_min = np.minimum(loss_np,loss_min)
    loss_min = np.float32(loss_min)
    torch.set_printoptions(precision=4)
    vist_pred = model(x)
    vist_pred_np =  np.max(vist_pred.detach().numpy()[:,0]/viscos)
    print('vist_pred_np',vist_pred_np)
    print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]:.2e}, Loss: {loss_np[0]:.2e}, Loss_min: {loss_min[0]:.2e}, vist_max: {vist_pred_np:.2e}")

# Plot loss_function

np.savetxt('loss-vist-diffusion-pinn-5200-half-channel-load.txt',\
   np.c_[differential_equation_loss_history,loss_min_history])

fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
ax.semilogy(np.arange(len(boundary_condition_loss_history)), boundary_condition_loss_history,color='red', label='bc error')
ax.semilogy(np.arange(len(boundary_condition_loss_history)), differential_equation_loss_history,color='blue',label="diff eq error")
plt.xlabel(r'epochs')
ax.set_title(r'Errors')
ax.grid(visible=True)
ax.legend(loc='best') 

vist_pred = model(x)
vist_pred_np =  vist_pred.detach().numpy()[:,0]

######################## plot vist
fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
plt.subplots_adjust(left=0.20,bottom=0.20)
vist_DNS_np =vist_DNS.detach().numpy()[:,0]/viscos
yplus_DNS_np =yplus_DNS.detach().numpy()[:,0]
ax.plot(yplus_DNS_np, vist_DNS_np,color='r',linestyle=':',linewidth=5, label='DNS')
vist_pred = model(x)  #get k
vist_pred_np =  vist_pred.detach().numpy()[:,0]/viscos
ax.plot(yplus_DNS_np, vist_pred_np,color='k',linestyle='-',linewidth=2, label=r"$\nu_t{\mathrm{pred}}$")
ax.legend(loc='best') 
plt.xlabel('$y^+$')
plt.ylabel(r'$\nu_t/\nu$')   
ax.grid(visible=True)
plt.savefig('test-PINN-vist-half-5200-plus-units-load.png',bbox_inches='tight')

######################## plot vist zoom
fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
plt.subplots_adjust(left=0.20,bottom=0.20)
ax.plot(yplus_DNS_np, vist_DNS_np,color='r',linestyle=':',linewidth=5, label='DNS')
ax.plot(yplus_DNS_np, vist_pred_np,color='k',linestyle='-',linewidth=2, label=r"$\nu_t{\mathrm{pred}}$")
ax.plot(y_kom/viscos, vist_kom/viscos,color='b',linestyle='-',linewidth=2, label=r"$\nu_{t,k-\omega}$")
ax.legend(loc='best') 
plt.xlabel('$y^+$')
plt.ylabel(r'$\nu_t/\nu$')   
plt.xlim(0,100)
ax.axis([0,50,0,20])
ax.grid(visible=True)
plt.savefig('vist-PINN-5200-plus-units-load.png',bbox_inches='tight')

np.savetxt('vist_pred-PINN-from-vist-diffusion-pinn-5200-plus-units-load.txt',np.c_[y_kom,vist_pred_np])

######################## plot diffusion term
fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
plt.subplots_adjust(left=0.20,bottom=0.20)
dkdy_DNS_np =  dkdy_DNS.detach().numpy()[:,0]
diff_DNS =  diff_DNS.detach().numpy()[:,0]
y_DNS =  y_DNS.detach().numpy()[:,0]
term = dkdy_DNS_np*vist_pred_np
dvist_dy = get_derivative( vist_pred,x)  
diff_non_conserv = vist_pred * d2kdy2_DNS + dkdy_DNS*dvist_dy
diff_DNS_pred = np.gradient(term,y_DNS)
plt.plot(y_DNS/viscos, (diff_DNS_pred+diff_DNS_visc),color='k',linestyle='-',linewidth=2, label=r"predicted")
plt.plot(y_DNS/viscos, (diff_DNS+diff_DNS_visc),color='b',linestyle='-',linewidth=2, label=r"DNS")
plt.plot(y_DNS/viscos, diff_non_conserv.detach().numpy(),color='r',linestyle='-',linewidth=2, label=r"non-cons")
ax.legend(loc='best') 
plt.xlabel('$y^+$')
plt.ylabel('diffusion')   
ax.grid(visible=True)
plt.savefig('diffusion-PINN-5200-plus-units-load.png',bbox_inches='tight')


######################## plot diffusion term zoom more
fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(y_DNS/viscos, diff_DNS,'r--',linewidth=2, label=r"DNS")
plt.plot(y_DNS/viscos, diff_non_conserv.detach().numpy(),'b-',linewidth=2, label=r"PINN")
plt.plot(y_DNS/viscos, diff_non_conserv.detach().numpy(),'bo',linewidth=2)
ax.legend(loc='best') 
plt.xlabel('$y^+$')
plt.ylabel('diffusion')   
plt.xlim(0,100)
ax.grid(visible=True)
plt.savefig('diffusion-PINN-5200-plus-units-load.png',bbox_inches='tight')


################################# Plot imbalance, Pk and diss zoom
fig, ax = plt.subplots(nrows=1, ncols=1) # Create a figure with one subplot
plt.subplots_adjust(left=0.20,bottom=0.20)
ax.plot(yplus_DNS.detach().numpy(), imbalance.detach().numpy(),color='r',linestyle=':',linewidth=5, label='imbalance')
ax.plot(yplus_DNS.detach().numpy(), Pk_DNS.detach().numpy(),color='k',linestyle='-',linewidth=2, label=r"$P_{k,DNS}$")
ax.plot(yplus_DNS.detach().numpy(), -diss_DNS.detach().numpy(),color='b',linestyle='-',linewidth=2, label=r"$\varepsilon_{DNS}$")
ax.plot(yplus_DNS.detach().numpy(), diff_DNS_visc,color='r',linestyle='--',linewidth=2, label=r"$D^\nu_{DNS}$")
ax.plot(yplus_DNS.detach().numpy(), diff_DNS,color='b',linestyle='--',linewidth=2, label=r"$D^t_{DNS}$")
ax.legend(loc='best') 
plt.xlabel('$y^+$')
ax.grid(visible=True)
plt.xlim(0,100)
plt.savefig('k-balance-PINN-5200-plus-units-load.png',bbox_inches='tight')
    
