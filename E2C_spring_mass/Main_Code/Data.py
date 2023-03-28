from matplotlib import animation
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import random
import math
from tqdm import trange
import os
import torch
import pdb 

from os import path
import json
from datetime import datetime
import argparse
def mydiff(x,t,m,k,c,xdes):
    F=k*xdes
    dxdt0=x[1]
    dxdt1=1/m*(F-k*x[0]-c*x[1])
    dxdt = [dxdt0, dxdt1]
    return dxdt
def run_singlemass_sim(sample_size=50,output_dir='/Users/avi/Desktop/Food_GVAE-master/E2C_spring_mass/data'):
    k=abs(random.gauss(20.,5.))
    m=abs(random.gauss(30.,2.5))
    c=2*(k*m)**0.5
    samples=[]
    xdes=0.5

    if not path.exists(output_dir):
        os.makedirs(output_dir)
    for i in trange(sample_size):
        x=np.random.uniform(0,1)
        xdot=np.random.uniform(-10,10)
        state = np.array([x, xdot])
        initial_state = np.copy(state)
        F= k*x+c*xdot
        state = odeint(mydiff, state,[0,1],args=(m,k,c,xdes))
        after_state = np.copy(state[-1])
        samples.append({
            'before': initial_state.tolist(),
            'after': after_state.tolist(),
            'control': F,
        })
    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': sample_size,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)
def robot(output_dir='/Users/avi/Desktop/Food_GVAE-master/E2C_spring_mass/data'):
    a=torch.load('/Users/avi/Desktop/Food_GVAE-master/E2C_spring_mass/data_exp_osc_02142023.pt')
    def normalize(array, min, max):
        return (array - min) / (max - min)
    a1=[]
    a2=[]
    a3=[]
    a4=[]
    a5=[]
    a6=[]
    a7=[]
    a8=[]

    
    for i in range(len(a)):
        a1.append(a[i][0][0].detach().numpy())
        a2.append(a[i][0][1].detach().numpy())
        a3.append(a[i][0][2].detach().numpy())
        a4.append(a[i][0][3].detach().numpy())
        a5.append(a[i][1][0].detach().numpy())
        a6.append(a[i][1][1].detach().numpy())
        a7.append(a[i][1][2].detach().numpy())
        a8.append(a[i][1][3].detach().numpy())


    a1=normalize(np.array(a1),min(a1),max(a1))
    a2=normalize(np.array(a2),min(a2),max(a2))
    a3=normalize(np.array(a3),min(a3),max(a3))
    a4=normalize(np.array(a4),min(a4),max(a4))
    a5=normalize(np.array(a5),min(a5),max(a5))
    a6=normalize(np.array(a6),min(a6),max(a6))
    a7=normalize(np.array(a7),min(a7),max(a7))
    a8=normalize(np.array(a8),min(a8),max(a8))



    a1=torch.from_numpy(a1)
    a2=torch.from_numpy(a2)
    a3=torch.from_numpy(a3)
    a4=torch.from_numpy(a4)
    a5=torch.from_numpy(a5)
    a6=torch.from_numpy(a6)
    a7=torch.from_numpy(a7)
    a8=torch.from_numpy(a8)


    for i in range(len(a)):
        a[i][0][0]=a1[i]
        a[i][0][1]=a2[i]
        a[i][0][2]=a3[i]
        a[i][0][3]=a4[i]
        a[i][1][0]=a5[i]
        a[i][1][1]=a6[i]
        a[i][1][2]=a7[i]
        a[i][1][3]=a8[i]

    samples=[]
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    for i in trange(len(a)):
        if(i==0):
            a[i][0]=torch.hstack((a[i][0],torch.tensor([0.,0.,0.,0.])))
            a[i][1]=torch.hstack((a[i][1],torch.tensor([a[i][1][0],a[i][1][1],a[i][1][2],a[i][1][3]])))
        else:
            a[i][0]=torch.hstack((a[i][0],torch.tensor([a[i][0][0]-a[i-1][0][0],a[i][0][1]-a[i-1][0][1],a[i][0][2]-a[i-1][0][2],a[i][0][3]-a[i-1][0][3]])))
            a[i][1]=torch.hstack((a[i][1],torch.tensor([a[i][1][0]-a[i-1][1][0],a[i][1][1]-a[i-1][1][1],a[i][1][2]-a[i-1][1][2],a[i][1][3]-a[i-1][1][3]])))
        initial_state = a[i][0]
        after_state =a[i][1]

        samples.append({
            'before': initial_state.tolist(),
            'after': after_state.tolist(),
            'control': 0,
        })
    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': 12720,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)
def experimental(output_dir='/Users/avi/Desktop/Food_GVAE-master/E2C_spring_mass/data_2'):
    a=torch.load('/Users/avi/Desktop/Food_GVAE-master/E2C_spring_mass/data_sim.pt')
    def normalize(array, min, max):
        return (array - min) / (max - min)
    a1=[]
    a2=[]
    a3=[]
    a4=[]
    a5=[]
    a6=[]
    a7=[]
    a8=[]
    a9=[]
    
    for i in range(len(a)):
        a1.append(a[i][0][0].detach().numpy())
        a2.append(a[i][0][1].detach().numpy())
        a3.append(a[i][0][2].detach().numpy())
        a4.append(a[i][0][3].detach().numpy())
        a5.append(a[i][1][0].detach().numpy())
        a6.append(a[i][1][1].detach().numpy())
        a7.append(a[i][1][2].detach().numpy())
        a8.append(a[i][1][3].detach().numpy())
        a9.append(a[i][0][4].detach().numpy())

    a1=normalize(np.array(a1),min(a1),max(a1))
    a2=normalize(np.array(a2),min(a2),max(a2))
    a3=normalize(np.array(a3),min(a3),max(a3))
    a4=normalize(np.array(a4),min(a4),max(a4))
    a5=normalize(np.array(a5),min(a5),max(a5))
    a6=normalize(np.array(a6),min(a6),max(a6))
    a7=normalize(np.array(a7),min(a7),max(a7))
    a8=normalize(np.array(a8),min(a8),max(a8))
    a9=normalize(np.array(a9),min(a9),max(a9))


    a1=torch.from_numpy(a1)
    a2=torch.from_numpy(a2)
    a3=torch.from_numpy(a3)
    a4=torch.from_numpy(a4)
    a5=torch.from_numpy(a5)
    a6=torch.from_numpy(a6)
    a7=torch.from_numpy(a7)
    a8=torch.from_numpy(a8)
    a9=torch.from_numpy(a9)

    for i in range(len(a)):
        a[i][0][0]=a1[i]
        a[i][0][1]=a2[i]
        a[i][0][2]=a3[i]
        a[i][0][3]=a4[i]
        a[i][1][0]=a5[i]
        a[i][1][1]=a6[i]
        a[i][1][2]=a7[i]
        a[i][1][3]=a8[i]
        # drop the a[i][0][4] and a[i][1][4] as they are the force
        a[i][0]=a[i][0][0:4]
        a[i][1]=a[i][1][0:4]
    samples=[]
    if not path.exists(output_dir):
        os.makedirs(output_dir)
    force_l= a9
    for i in trange(len(a)):
        if(i==0):
            a[i][0]=torch.hstack((a[i][0],torch.tensor([0.,0.,0.,0.])))
            a[i][1]=torch.hstack((a[i][1],torch.tensor([a[i][1][0],a[i][1][1],a[i][1][2],a[i][1][3]])))
        else:
            a[i][0]=torch.hstack((a[i][0],torch.tensor([a[i][0][0]-a[i-1][0][0],a[i][0][1]-a[i-1][0][1],a[i][0][2]-a[i-1][0][2],a[i][0][3]-a[i-1][0][3]])))
            a[i][1]=torch.hstack((a[i][1],torch.tensor([a[i][1][0]-a[i-1][1][0],a[i][1][1]-a[i-1][1][1],a[i][1][2]-a[i-1][1][2],a[i][1][3]-a[i-1][1][3]])))
        initial_state = a[i][0]
        after_state =a[i][1]

        samples.append({
            'before': initial_state.tolist(),
            'after': after_state.tolist(),
            'control': force_l[i].item(),
        })
    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': 12720,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)
def latent(output_dir='/Users/avi/Desktop/Food_GVAE-master/E2C_spring_mass/latent_2'):
    a=torch.load('/Users/avi/Desktop/Food_GVAE-master/E2C_spring_mass/Sim_data/latent_data.pt')

 
    samples=[]
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    for i in trange(len(a)):
        initial_state = a[i][0]
        after_state =a[i][1]
        force=a[i][2]

        samples.append({
            'before': initial_state.tolist(),
            'after': after_state.tolist(),
            'control': force.item(),
        })
    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': 16000,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)
def main(args):
    sample_size = args.sample_size
    #robot()
    latent()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample data')

    parser.add_argument('--sample_size', required=False,default=10000, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)
    

