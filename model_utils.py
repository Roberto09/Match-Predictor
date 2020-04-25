from fastai import *
from fastai.tabular import *

""" SETUP DEVICE """
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def get_matching_idxs(cat):
    """get_matching_idxs:
    for a set of games that happened on the same jornada it will return a list of pairs
    of the matching games were each item has the following:
    (local, visitor)
    """
    # cat is this: rnn_tab_ds.data[0][0][0][:, 0].shape
    dic = {}
    #equipo is at 0 and equipo c is 1
    for i in range(cat.shape[0]):
        if(cat[i][0].item() == 0): continue
        e, e_c, is_loc = int(cat[i][0].item()-1), int(cat[i][1].item()-1), int(cat[1][2].item())
        
        #swapping so that e is the local=1
        if is_loc != 1:
            aux = e
            e = e_c
            e_c = aux
        
        if e in dic or e_c in dic: continue
        dic[e] = e_c
    
    ret_list = list(dic.items())
    ret_list.sort()
    return ret_list

def get_loss(model, data_cat, data_cont, data_res):
    """ get_loss function:
    Calculates Crossentropy loss on the given model and data. 
    """
    # print(model.h)
    model.eval()
    out = model(data_cat.to(device), data_cont.to(device))
    curr_crit = nn.CrossEntropyLoss()
    loss_x = curr_crit(out, data_res)
    return loss_x
    
def pred(model, data_cat, data_cont, data_res):
    """ pred function:
    Given a *trained* model and prediction data it will make a prediction and return us 3 values:
    Predicted class, Predicted probs (using softmax), Raw Model Output
    """
    # print(model.h)
    model.eval()
    m_out = model(data_cat.to(device), data_cont.to(device))
    prob = nn.functional.softmax(m_out, dim=1).data
    # Taking the class with the highest probability score from the output
    out = torch.max(prob, dim=1)[1]
    return out, prob, m_out

def accuracy(model, data_cat, data_cont, data_res):
    """ accuracy function:
    Given a model and data to predict it will make such predictions and simply returning the precentage
    of those that were right regardless of how far/close the softmax value was from the actual one.
    """

    out, _, _= pred(model, data_cat, data_cont, data_res)
    ttl = 0
    for i in range(len(out)):
        if(data_res.view(-1)[i] == out[i]): ttl += 1
    return ttl/len(out)

def get_res(cat, res):
    """ get_res function:
    Generates and returns the result to be compared while calculating crossentropy loss.
    """
    # print(res.shape)
    n_res = None
    for i_bptt in range(cat.shape[1]):
        p_lst = get_matching_idxs(cat[:, i_bptt])
        for e, e_c in p_lst:
            sub_r = res[e, i_bptt]
            n_res = torch.cat([n_res, sub_r.view(-1)], 0) if n_res is not None else sub_r.view(-1)
    return n_res 

def view_predictions_and_results(cat, out, res):
    """ view_predictions_and_results function:
    prints the result of the predictions in a formated and more understandable way
    """
    cnt = 0
    for i_bptt in range(cat.shape[1]):
        p_lst = get_matching_idxs(cat[:, i_bptt])
        for e, e_c in p_lst:
            print(e, " vs ", e_c, " --------------  out: ", out[cnt].item(), " --- res: ",  res[cnt].item())
            cnt += 1
    assert (cnt == len(out) and cnt == len(res))


"""Old functions that used MSE rather than CrossEntropy.
Saved just as reference in case they are needed in the future.
"""
# # This function takes in the model and character as arguments and returns the next character prediction and hidden state
# def get_loss(model, data_cat, data_cont, data_res):
# #     print(model.h)
#     model.eval()
#     out = model(data_cat.to(device), data_cont.to(device))
#     curr_crit = nn.MSELoss()
#     loss_x = curr_crit(out.view(-1), data_res.view(-1))
#     return loss_x
    
# def pred(model, data_cat, data_cont, data_res):
# #     print(model.h)
#     model.eval()
#     m_out = model(data_cat.to(device), data_cont.to(device))
#     return m_out.view(-1), m_out

# def accuracy(model, data_cat, data_cont, data_res):
#     out, _= pred(model, data_cat, data_cont, data_res)
#     ttl = 0
#     for i in range(len(out)):
#         if(data_res.view(-1)[i] == out[i]): ttl += 1;
#     return ttl/len(out)