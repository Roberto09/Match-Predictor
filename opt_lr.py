import math
import matplotlib.pyplot as plt
from fastai import *

"""Class that helps us see how our model reacts to different learning rates"""
class CLR():
    def __init__(self, train_dl, base_lr = 1e-5, max_lr = 100):
        self.base_lr, self.max_lr = base_lr, max_lr
        self.bn = len(train_dl) - 1 #iterations in our epoch
        ratio = self.max_lr/self.base_lr
        self.mult = ratio ** (1/self.bn)
        self.best_loss = 1e9
        self.iteration = 0
        self.lrs, self.losses = [], []
        
    def calc_lr(self, loss):
        self.iteration += 1
        if math.isnan(loss) or loss > 4*self.best_loss:
            return -1 #stop trying to finf a good learining rate
        if loss < self.best_loss and self.iteration > 1:
            self.best_loss = loss #improve our best loss
        
        mult = self.mult ** self.iteration
        lr = self.base_lr * mult
        self.lrs.append(lr)
        self.losses.append(loss)
        return lr
    
    def plot(self, start = 10, end = -5):
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs, self.losses)
        plt.xscale('log') #learning rates are in log scale
        axes = plt.gca()
        axes.set_ylim([0.8,1.5])


def find_lr(loss_func, opt, clr, model, train_dl):
    running_loss = 0.
    avg_beta = 0.98
    model.train()
    
    for i, curr_batch in enumerate(train_dl):
        x_cat, x_cont, res = curr_batch[0][0].to(device), curr_batch[0][1].to(device), curr_batch[1].to(device) 
        output = model(x_cat, x_cont)
        loss = loss_func(output.view(x_cat.shape[0] * x_cat.shape[1],3), res.view(-1).long())
        
        # calculate smoothed loss
        running_loss = avg_beta * running_loss + (1-avg_beta) * loss
        smoothed_loss = running_loss / (1 - avg_beta**(i+1))
        
        lr = clr.calc_lr(smoothed_loss)
        if lr == -1: break
        for pg in opt.param_groups:
            pg['lr'] = lr
            
        opt.zero_grad()
        loss.backward()
        opt.step()