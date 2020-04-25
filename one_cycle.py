class OneCycle():
    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10 , div=10):
        self.nb = nb # total number of iterations including all epochs
        self.div = div # the division factor used to get lower boundary of learning rate
        self.step_len =  int(self.nb * (1- prcnt/100)/2)
        self.high_lr = max_lr # the optimum learning rate, found from LR range test
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt # percentage of cycle length,we annihilate learning rate below the lower learnig rate (default is 10)
        self.iteration = 0
        self.lrs = []
        self.moms = []
        
    def calc(self): # calculates learning rate and momentum for the batch
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return (lr, mom)
        
    def calc_lr(self):
        if self.iteration==self.nb: # exactly at `d`
            self.iteration = 0
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        
        if self.iteration > 2 * self.step_len: # case c-d
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * ( 1 - ratio * (1-(1/self.div))/self.div)
        elif self.iteration > self.step_len: # case b-c
            ratio = 1- (self.iteration -self.step_len)/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else : # case a-b
            ratio = self.iteration/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr
    
    def calc_mom(self):
        if self.iteration==self.nb: # exactly at `d`
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len: # case c-d
            mom = self.high_mom
        elif self.iteration > self.step_len: # case b-c 
            ratio = (self.iteration -self.step_len)/self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else: # case a-b
            ratio = self.iteration/self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom