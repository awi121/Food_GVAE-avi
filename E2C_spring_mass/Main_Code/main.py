from VAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
#import springmassdamper as smd
import copy
import time
import torch
#import animation_test
from scipy import signal
from datasets import *
import pdb


torch.set_default_dtype(torch.float32)


BS=2048    # Batch size for training

## Run new simulations ##
# d1, sim_length, _, _=smd.run_multimass_sim(run_nums=30,out_data=3,num_repeats=1)  # run simulation of 3 masses and a pendulum
#d1, sim_length, _, _=smd.run_singlemass_sim(run_nums=30,out_data=3,num_repeats=1)   # run simulation of single mass system

## Load previously generated simulation data ##
# d1=torch.load('data_3.pt')

datasets = {'springmass': spring_mass,'experimental':experimental}

propor=0.75
dataset = datasets['experimnetal']('/Users/avi/Desktop/Food_GVAE-master/E2C_spring_mass/data_2/' )
train_set, test_set = dataset[:int(len(dataset) * propor)], dataset[int(len(dataset) * propor):]
train = torch.utils.data.DataLoader(train_set, batch_size=2048, shuffle=True, drop_last=False)
test= torch.utils.data.DataLoader(test_set, batch_size=2048, shuffle=True, drop_last=False)
#train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)   

model=VAE()
device = torch.device("cpu")    # Save the model to the CPU
model.to(device)
# model.load_state_dict(torch.load("./current_model_exp2"))     # Load a previously trained model
'''for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
    pdb.set_trace()'''
## Training loop ##
for i in range(100):
    loss=model.training_step(train,device)
    print(i, loss)

torch.save(model.state_dict(), 'current_model_experimental')    # Save the current model


## Testing loop ##
model=VAE()


xhat, z, x = model.test(test,device)
sim_length=1
## Plot the latent space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(z[i:i+sim_length,0],z[i:i+sim_length,1])
plt.show()

## Plot the state space phase portrait ##
for i in range(0,len(x),sim_length):
    plt.plot(x[i:i+sim_length,0],x[i:i+sim_length,1])
plt.show()
