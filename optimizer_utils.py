def changeLr(lr, opt):
    """ changeLr function:
    updates learning rate used in opt
    """
    for pg in opt.param_groups: # update learning rate
        pg['lr'] = lr

def changeLrAndMomentums(lr, momentum, opt):
    """ changeLrAndMomentums function:
    updates learning rate and momentum used in opt
    """
    for pg in opt.param_groups: # update learning rate
        pg['lr'] = lr
        pg['betas'] = (momentum, 1.0-momentum)