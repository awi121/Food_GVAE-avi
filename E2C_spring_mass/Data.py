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
def main(args):
    sample_size = args.sample_size

    run_singlemass_sim(sample_size=sample_size)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample data')

    parser.add_argument('--sample_size', required=False,default=10000, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)
    

