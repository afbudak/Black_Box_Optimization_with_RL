#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:59:17 2019

@author: afbudak
"""

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from gpflowopt.domain import ContinuousParameter
import gpflow
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.design import LatinHyperCube
from gpflowopt.acquisition import ExpectedImprovement
from gpflowopt.optim import SciPyOptimizer, StagedOptimizer, MCOptimizer

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

def ackley_bayes(x):
    n = 3;
    a = 20; 
    b = 0.2; 
    c = 2*np.pi;
    s1 = 0;
    s2 = 0;
    for i in range(n):
       s1 = s1+x[:,i]**2;
       s2 = s2+np.cos(c*x[:,i]);
       res = -a*np.exp(-b*np.sqrt(1/n*s1))-np.exp(1/n*s2)+a+np.exp(1)
    return res[:,None]

def shub_bayes(x):
#
# Shubert function
# Matlab Code by A. Hedar (Nov. 23, 2005).
# The number of variables n =2.
# 
    s1 = np.zeros(1); 
    s2 = np.zeros(1);
    for i in range(5):
        s1 = s1+(i+1)*np.cos((i+2)*x[:,0]+i+1);
        s2 = s2+(i+1)*np.cos((i+2)*x[:,1]+i+1);
    
    res = s1*s2
    
    return res[:,None]

def trid_bayes(x):
    n = 6;
    s1 = 0;
    s2 = 0;
    for j in range(n):
        s1 = s1+(x[:,j]-1)**2   
    for j in range(1,n):
        s2 = s2+x[:,j]*x[:,j-1]  
    res = s1-s2
    return res[:,None]


n = 3;
lb = np.array([-15,-15,-15])
ub = np.array([30,30,30])
fname = ackley
fnamestring = 'ackley'
fname_bayes = ackley_bayes
lmbda = torch.tensor([1,1,1]).float()
domainackley = ContinuousParameter('x1', -15, 30) + ContinuousParameter('x2', -15, 30) + ContinuousParameter('x3', -15, 30)
domain = domainackley

#n=2
#lb = np.array([-10,-10])
#ub = np.array([10,10])
#lmbda = torch.tensor([5,5]).float()
#fname = shub
#fnamestring = 'shub'
#fname_bayes = shub_bayes
#domainshub = ContinuousParameter('x1', -10, 10) + ContinuousParameter('x2', -10, 10)
#domain = domainshub

#n=6
#lb = np.array([-n**2,-n**2,-n**2,-n**2,-n**2,-n**2])
#ub = np.array([n**2,n**2,n**2,n**2,n**2,n**2])
#lmbda = torch.tensor([5,5,5,5,5,5]).float()
#fname = trid
#fnamestring = 'trid'
#fname_bayes = trid_bayes
#domaintrid = ContinuousParameter('x1', -36, 36) + ContinuousParameter('x2', -36, 36) + ContinuousParameter('x3', -36, 36) + ContinuousParameter('x4', -36, 36) + ContinuousParameter('x5', -36, 36) + ContinuousParameter('x6', -36, 36)
#domain = domaintrid


pop_size = 50;
F = 0.75;
max_iter = 300;
max_iter_bayesian = 200;
CR = 0.5;

num_of_runs_to_eval = 10
run_res_DE_crit = list()
run_res_BO = list()
run_res_actor_critic = list()



############# PERFORMANCE EVALUATION FOR BAYESIAN OPTIMIZATION ################ 
for run in range(4):
    lhd = LatinHyperCube(pop_size, domain)
    X = lhd.generate()
    Y = fname_bayes(X)
    model = gpflow.gpr.GPR(X, Y, gpflow.kernels.Matern52(2, ARD=True))
    model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)
    # Now create the Bayesian Optimizer
    alpha = ExpectedImprovement(model)
    
    acquisition_opt = StagedOptimizer([MCOptimizer(domain, 200),
                                       SciPyOptimizer(domain)])
    
    optimizer = BayesianOptimizer(domain, alpha, optimizer=acquisition_opt, verbose=True)
    best_iter = np.zeros(max_iter)
    # Run the Bayesian optimization
    for i in range(max_iter-150):
        r = optimizer.optimize(fname_bayes, n_iter=1)
        best_iter[i] = -r.fun[0]
        
    run_res_BO.append(best_iter)


############# PERFORMANCE EVALUATION ACTOR-CRITIC ################ 
for run in range(num_of_runs_to_eval):
    pop_parent = lb + np.random.random([pop_size,n]) * (ub-lb)
    val_parent = np.zeros(pop_size)
    for i in range(pop_size):
        val_parent[i] = -1*fname(pop_parent[i,:])
        
    best_idx = np.argmax(val_parent)
    best_mem = pop_parent[best_idx,:]
    
     
    #define critic model
    crit_n_in = 2*n
    n_h = 128
    n_out = 1
    crit_model = nn.Sequential(nn.Linear(crit_n_in, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_out))
    
    alpha =  0.001
    crit_model = crit_model.float()
    crit_criterion = nn.MSELoss()
    crit_optimizer = torch.optim.Adam(crit_model.parameters(), lr=alpha, betas=(0.9, 0.999))
    training_num = pop_size**2
    crit_xtrain_set = np.zeros((training_num,crit_n_in))
    crit_ytrain_set = np.zeros((training_num,1))
    
    #define actor model
    actor_n_in = n
    n_h = 128
    n_out = n
    actor_model = nn.Sequential(nn.Linear(actor_n_in, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_h),
                             nn.ReLU(),
                             nn.Linear(n_h, n_out))
    
    alpha =  0.0001
    actor_model = actor_model.float()
    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=alpha, betas=(0.9, 0.999))
    actor_xtrain_set = np.zeros((training_num,actor_n_in))
    actor_ytrain_set = np.zeros((training_num,actor_n_in))
    
    best_iter =  np.zeros(max_iter)
    choosen_child_thing = np.zeros((max_iter,pop_size))
    
    
    for iter in range(max_iter):
        
        for i in range(pop_size):
            for j in range(pop_size):
                trn_idx = i*pop_size + j          
                actor_ytrain_set[trn_idx,:] = pop_parent[j] - pop_parent[i]
                actor_xtrain_set[trn_idx,:] = pop_parent[i]
                crit_xtrain_set[trn_idx,:] = np.concatenate((pop_parent[i],actor_ytrain_set[trn_idx,:]))
                crit_ytrain_set[trn_idx] = val_parent[j]
    
        xy = np.concatenate((crit_xtrain_set,crit_ytrain_set),axis=1)
        epoch = 20
        for epoch in range(epoch):
            xy = np.random.permutation(xy)
            crit_xtrain = xy[:1000,0:crit_n_in]
            crit_ytrain = xy[:1000,crit_n_in:]
            crit_y_pred = crit_model(torch.tensor(crit_xtrain).float())
            crit_loss = crit_criterion(crit_y_pred, torch.tensor(crit_ytrain).float())
            #print('Epoch {}: train loss: {}'.format(epoch, crit_loss.item()))
            crit_optimizer.zero_grad()
            crit_loss.backward()
            crit_optimizer.step()
            
        xy = np.concatenate((actor_xtrain_set,actor_ytrain_set),axis=1)
        pseudo_lb = np.amin(pop_parent, axis=0)
        pseudo_ub = np.amax(pop_parent, axis=0)
        pop_pseudo_parent_set = pseudo_lb + np.random.random([training_num,n]) * (pseudo_ub-pseudo_lb)
        epoch = 3
        for epoch in range(epoch):
            pop_pseudo_parent_set = np.random.permutation(pop_pseudo_parent_set)
            pop_pseudo_parent = pop_pseudo_parent_set[:1000,:]
            actor_input = torch.tensor(pop_pseudo_parent).float()
            pred_actor = actor_model(actor_input)
            delx = pred_actor
            xfinal = actor_input + delx
            viol_ub = torch.max(torch.tensor([0]).float(),xfinal - torch.tensor(ub).float())
            viol_lb = torch.min(torch.tensor([0]).float(),xfinal - torch.tensor(lb).float())
            viol = viol_ub+viol_lb
            viol = viol**2
            actor_loss = -1*torch.sum(crit_model(torch.cat((actor_input,pred_actor),axis=1))) + sum(sum(lmbda * viol))
            #actor_loss = -1*torch.sum(crit_model(torch.cat((actor_input,pred_actor),axis=1)))
            #print('Epoch {}: train loss: {}'.format(epoch, actor_loss.item()))
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
        
    
        
        #Generate child pop
        child_pop = pop_parent + actor_model(torch.tensor(pop_parent).float()).detach().numpy() + 0.3*np.random.normal(0,np.std(pop_parent,axis=0))
    
        for i in range(pop_size):
            for j in range(n):
                if child_pop[i,j]<lb[j]:
                    child_pop[i,j] = lb[j] + np.random.random(1) * (ub[j]-lb[j])
                   
                if child_pop[i,j]>ub[j]:
                    child_pop[i,j] = lb[j] + np.random.random(1) * (ub[j]-lb[j])
    
        #choosen_child_idx = np.random.choice(pop_size);
        child_state_action = np.concatenate((pop_parent, child_pop - pop_parent),axis=1)
        y_child = crit_model(torch.tensor(child_state_action).float()).detach().numpy()
        choosen_child_idx = np.argmax(y_child)
        choosen_child = child_pop[choosen_child_idx,:]
        choosen_child_val = -1*fname(choosen_child)
        choosen_child_thing[iter,:]=val_parent
        #compare with parent
        worst_parent_idx = np.argmin(val_parent)
        worst_parent = pop_parent[worst_parent_idx,:]
        worst_parent_val = val_parent[worst_parent_idx];
        if choosen_child_val>worst_parent_val:
            pop_parent[worst_parent_idx,:] = choosen_child;
            val_parent[worst_parent_idx] = choosen_child_val;
    
        best_iter[iter] = max(val_parent)
    
    #print(max(val_parent))
        
    run_res_actor_critic.append(best_iter)


############# PERFORMANCE EVALUATION DE+CRITIC ################ 
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
        randomidx = np.random.permutation(pop_size)
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
y = [np.mean(run_res_BO,axis=0),-np.mean(run_res_DE_crit,axis=0),np.mean(run_res_actor_critic,axis=0)]
labels = ['BO', 'DECN', 'BOAC']
for y_arr, label in zip(y, labels):
    plt.plot(x, y_arr, label=label)

plt.legend()
plt.title(str(n)+' dimensional ' + fnamestring + ' function')
plt.ylabel('function val')
plt.xlabel('iterations')
plt.show()


#x = np.arange(0,max_iter,1)
#y = [-np.mean(run_res_DE_crit,axis=0),np.mean(run_res_actor_critic,axis=0)]
#labels = ['DE_w_Critic', 'NOAC']
#for y_arr, label in zip(y, labels):
#    plt.plot(x, y_arr, label=label)
#
#plt.legend()
#plt.title(str(n)+' dimensional ' + fnamestring + ' function')
#plt.ylabel('function val')
#plt.xlabel('iterations')
#plt.show()