#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:00:40 2019

@author: afbudak
"""

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.special import comb

def ackley(x):
    n = 3;
    a = 20; 
    b = 0.2; 
    c = 2*np.pi;
    s1 = 0;
    s2 = 0;
    for i in range(n):
       s1 = s1+x[i]**2;
       s2 = s2+np.cos(c*x[i]);
    return -a*np.exp(-b*np.sqrt(1/n*s1))-np.exp(1/n*s2)+a+np.exp(1)

def shub(x):
    s1 = np.zeros(1); 
    s2 = np.zeros(1);
    for i in range(5):
        s1 = s1+(i+1)*np.cos((i+2)*x[0]+i+1);
        s2 = s2+(i+1)*np.cos((i+2)*x[1]+i+1);
    
    return s1*s2

def trid(x):
    n = 6;
    s1 = 0;
    s2 = 0;
    for j in range(n):
        s1 = s1+(x[j]-1)**2   
    for j in range(1,n):
        s2 = s2+x[j]*x[j-1]  
    return s1-s2


n = 3;
lb = np.array([-15,-15,-15])
ub = np.array([30,30,30])
fname = ackley
fnamestring = 'ackley'

#n=2
#lb = np.array([-10,-10])
#ub = np.array([10,10])
#fname = shub
#fnamestring = 'shub'

#n=6
#lb = np.array([-n**2,-n**2,-n**2,-n**2,-n**2,-n**2])
#ub = np.array([n**2,n**2,n**2,n**2,n**2,n**2])
#lmbda = torch.tensor([5,5,5,5,5,5]).float()
#fname = trid
#fnamestring = 'trid'

pop_size = 50;
F = 0.75;
max_iter = 500;
CR = 0.5;

num_of_runs_to_eval = 10
run_res_pure_DE = list()
run_res_DE_crit = list()

for run in range(num_of_runs_to_eval):
    pop_parent = lb + np.random.random([pop_size,n]) * (ub-lb)
    val_parent = np.zeros(pop_size)
    for i in range(pop_size):
        val_parent[i] = fname(pop_parent[i,:])
    
    
    best_iter =  np.zeros(max_iter)
    for iter in range(max_iter):
        
        #Generate child pop
        pop_child = np.zeros((pop_size,n))
        best_idx = np.argmin(val_parent)
        best_mem = pop_parent[best_idx,:]
        randomidx = np.random.permutation(50)
        random_pop1 = pop_parent[randomidx,:]
        random_pop2 = pop_parent[np.roll(randomidx,1),:]
        #best_mem_pop = repmat(best_mem,[pop_size,1]);
        child_pop = pop_parent + F*(best_mem - pop_parent + random_pop2 - random_pop1);
        for i in range(pop_size):
            for j in range(n):
                if child_pop[i,j]<lb[j]:
                    child_pop[i,j] = lb[j]
                    
                if child_pop[i,j]>ub[j]:
                    child_pop[i,j] = ub[j]
            
        mpo = np.random.random((pop_size,n)) < CR;
        mui = mpo < 0.5;
        child_pop = child_pop*mpo + pop_parent*mui;
        choosen_child_idx = np.random.choice(pop_size);
        choosen_child = child_pop[choosen_child_idx,:]
        choosen_child_val = fname(choosen_child)
        #compare with parent
        parent_val = val_parent[choosen_child_idx];
        if choosen_child_val<parent_val:
            pop_parent[choosen_child_idx,:] = choosen_child;
            val_parent[choosen_child_idx] = choosen_child_val;
    
        best_iter[iter] = min(val_parent);
    
    run_res_pure_DE.append(best_iter)


for run in range(num_of_runs_to_eval):
    pop_parent = lb + np.random.random([pop_size,n]) * (ub-lb)
    val_parent = np.zeros(pop_size)
    for i in range(pop_size):
        val_parent[i] = fname(pop_parent[i,:])
        
        
    n_in =  2*n
    n_h = 64
    n_out = 1
    model = nn.Sequential(nn.Linear(n_in, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_out))
    
    alpha =  0.001
    model = model.float()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha, betas=(0.9, 0.999))
    
    #train the network for the first time
    training_num = pop_size**2
    xtrain_set = np.zeros((training_num,n_in))
    ytrain_set = np.zeros((training_num,1))
    
    
    
    #Critic update
    for i in range(pop_size):
        for j in range(pop_size):
            trn_idx = i*pop_size + j          
            xtrain_set[trn_idx,:] = np.concatenate((pop_parent[i],pop_parent[j] - pop_parent[i]))
            ytrain_set[trn_idx] = val_parent[j]
    
    xy = np.concatenate((xtrain_set,ytrain_set),axis=1)
    xy = np.random.permutation(xy)
    train_size = int(training_num*0.8)
    test_size = training_num - train_size
    crit_xtrain = xy[:train_size,0:n_in]
    crit_ytrain = xy[:train_size,n_in:]
    crit_xtest = xy[train_size:,0:n_in]
    crit_ytest = xy[train_size:,n_in:]
    
    
    epoch = 50
    batch_size = 64
    #loss_array_train = np.zeros(int(np.floor(train_size/batch_size))*epoch)
    #loss_array_test = np.zeros(int(np.floor(train_size/batch_size))*epoch)
    loss_array_train = np.zeros(epoch+max_iter)
    loss_array_test = np.zeros(epoch+max_iter)
    for epoch in range(epoch):
        temp_conc = np.concatenate((crit_xtrain,crit_ytrain),axis=1)
        temp_conc = np.random.permutation(temp_conc)
        crit_xtrain = temp_conc[:,0:n_in]
        crit_ytrain = temp_conc[:,n_in:]
        for train_iter in range(int(np.floor(train_size/batch_size))):
            xtrain = crit_xtrain[train_iter*batch_size:(train_iter+1)*batch_size,:]
            ytrain = crit_ytrain[train_iter*batch_size:(train_iter+1)*batch_size,:]
            y_pred = model(torch.tensor(xtrain).float())
            loss = criterion(y_pred, torch.tensor(ytrain).float())
            #print('Epoch {}: train loss: {}'.format(epoch, crit_loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        y_hat_train = model(torch.tensor(crit_xtrain).float()).detach().numpy()
        mse_train = np.mean((y_hat_train-crit_ytrain)**2)
        loss_array_train[epoch] = mse_train
        y_hat_test = model(torch.tensor(crit_xtest).float()).detach().numpy()
        mse_test = np.mean((y_hat_test-crit_ytest)**2)
        loss_array_test[epoch] = mse_test
        #print(mse_test)
        
    
    best_iter =  np.zeros(max_iter)
    choosen_child_thing = np.zeros((max_iter,pop_size))
    for iter in range(max_iter):
        
        #Generate child pop
        pop_child = np.zeros((pop_size,n))
        best_idx = np.argmin(val_parent)
        best_mem = pop_parent[best_idx,:]
        randomidx = np.random.permutation(50)
        random_pop1 = pop_parent[randomidx,:]
        random_pop2 = pop_parent[np.roll(randomidx,1),:]
        #best_mem_pop = repmat(best_mem,[pop_size,1]);
        child_pop = pop_parent + F*(best_mem - pop_parent + random_pop2 - random_pop1);
        for i in range(pop_size):
            for j in range(n):
                if child_pop[i,j]<lb[j]:
                    child_pop[i,j] = lb[j]
                    
                if child_pop[i,j]>ub[j]:
                    child_pop[i,j] = ub[j]
            
        mpo = np.random.random((pop_size,n)) < CR;
        mui = mpo < 0.5;
        child_pop = child_pop*mpo + pop_parent*mui;
        #choosen_child_idx = np.random.choice(pop_size);
        child_state_action = np.concatenate((pop_parent, child_pop - pop_parent),axis=1)
        y_child = model(torch.tensor(child_state_action).float()).detach().numpy()
        choosen_child_idx = np.argmin(y_child)
        choosen_child = child_pop[choosen_child_idx,:]
        choosen_child_val = fname(choosen_child)
        choosen_child_thing[iter,:]=val_parent
        #compare with parent
        parent_val = val_parent[choosen_child_idx];
        if choosen_child_val<parent_val:
            pop_parent[choosen_child_idx,:] = choosen_child;
            val_parent[choosen_child_idx] = choosen_child_val;
    
        best_iter[iter] = min(val_parent);
        
        #update critic
        for i in range(pop_size):
            for j in range(pop_size):
                trn_idx = i*pop_size + j          
                xtrain_set[trn_idx,:] = np.concatenate((pop_parent[i],pop_parent[j] - pop_parent[i]))
                ytrain_set[trn_idx] = val_parent[j]
    
        xy = np.concatenate((xtrain_set,ytrain_set),axis=1)
        xy = np.random.permutation(xy)
        train_size = int(training_num*0.8)
        test_size = training_num - train_size
        crit_xtrain = xy[:train_size,0:n_in]
        crit_ytrain = xy[:train_size,n_in:]
        
        
        epoch = 2
        batch_size = 64
        #loss_array_train = np.zeros(int(np.floor(train_size/batch_size))*epoch)
        #loss_array_test = np.zeros(int(np.floor(train_size/batch_size))*epoch)
        loss_array_train = np.zeros(epoch+max_iter)
        loss_array_test = np.zeros(epoch+max_iter)
        for epoch in range(epoch):
            temp_conc = np.concatenate((crit_xtrain,crit_ytrain),axis=1)
            temp_conc = np.random.permutation(temp_conc)
            crit_xtrain = temp_conc[:,0:n_in]
            crit_ytrain = temp_conc[:,n_in:]
            for train_iter in range(int(np.floor(train_size/batch_size))):
                xtrain = crit_xtrain[train_iter*batch_size:(train_iter+1)*batch_size,:]
                ytrain = crit_ytrain[train_iter*batch_size:(train_iter+1)*batch_size,:]
                y_pred = model(torch.tensor(xtrain).float())
                loss = criterion(y_pred, torch.tensor(ytrain).float())
                #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            y_hat_train = model(torch.tensor(crit_xtrain).float()).detach().numpy()
            mse_train = np.mean((y_hat_train-crit_ytrain)**2)
            loss_array_train[epoch] = mse_train
            y_hat_test = model(torch.tensor(crit_xtest).float()).detach().numpy()
            mse_test = np.mean((y_hat_test-crit_ytest)**2)
            loss_array_test[epoch] = mse_test
            #print(mse_test)
    
    run_res_DE_crit.append(best_iter)
    
x = np.arange(0,max_iter,1)
y = [np.mean(run_res_pure_DE,axis=0),np.mean(run_res_DE_crit,axis=0)]
labels = ['SSDE', 'DECN']
for y_arr, label in zip(y, labels):
    plt.plot(x, y_arr, label=label)

plt.legend()
plt.title(str(n)+' dimensional ' + fnamestring + ' function')
plt.ylabel('function val')
plt.xlabel('iterations')
plt.show()
