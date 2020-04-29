from fastai import *
from fastai.tabular import *
from model_utils import get_matching_idxs


class MyTabularModel(Module):
    """ MyTabularModel:
    an exact copy of TabularModel from fastai however it won't do Batchnorm on our input data
    """
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x



class RnnTabularModel(nn.Module):
    """RnnTabularModel:
    helps us apply an RNN to a tabular model
    """
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, out_sz_rnn:int, lyrs:int, bs:int, ps:float=0.,
                 emb_drop:float=0., y_range:OptRange=None):
        super().__init__()
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs]) #Correct
        self.emb_drop = nn.Dropout(emb_drop) #Correct
        self.bn_cont = nn.BatchNorm1d(n_cont) #Correct
        n_emb = sum(e.embedding_dim for e in self.embeds) #Correct
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range #Correct
        
        #rnn stuff
        self.nh = n_cont + n_emb #Correct
        self.rnn = nn.GRU(self.nh, self.nh, lyrs, batch_first=True, dropout=ps) #Check changing nh and adding relu activation(kaggle post)
        self.h_o = nn.Linear(self.nh,out_sz_rnn) #Correct
        self.bn = BatchNorm1dFlat(self.nh) #Correct
        
        self.lyrs = lyrs #Correct
        self.bs = bs #Correct
        self.reset_h()#Correct
        
        #rnn sub-result proccessing stuff
        self.tab_mod = MyTabularModel([], out_sz_rnn*2, out_sz, [30, 15], [ps, ps], )

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        i_proc = torch.zeros(x_cat.shape[0], x_cat.shape[1], self.nh); #Correct
        
        for i_bptt in range(x_cat.shape[1]): #Correct
            curr_cat = x_cat[:,i_bptt,:] #Correct
            curr_cont = x_cont[:,i_bptt,:] #Correct
            
            if self.n_emb != 0: #Correct
                x = [e(curr_cat[:,i]) for i,e in enumerate(self.embeds)] #Correct
                x = torch.cat(x, 1) #Correct
                x = self.emb_drop(x) #Correct
            if self.n_cont != 0: #Correct
                curr_cont = self.bn_cont(curr_cont) #Correct
                x = torch.cat([x, curr_cont], 1) if self.n_emb != 0 else curr_cont #Correct
            i_proc[:, i_bptt] = x #Checked that copying works by printing embeding changes. Maybe just check how copying like this works for pytorch, a way to do this is to check the weight of embedings to see if they are updating
            
        res, h = self.rnn(i_proc, self.h) #Correct
        # if self.training: self.h = h.detach() #Correct       
        self.h = h.detach() #Correct 
        
        rnn_res = self.h_o(self.bn(res)) #Correct
        
        # new part, using results of both teams
        curr_res = []
        # TODO: maybe change this code to python C?
        for bptt_i in range(x_cat.shape[1]):
            p_lst = get_matching_idxs(x_cat[:, bptt_i])
            for eq, eq_c in p_lst:
#                 print("Shape: ", torch.cat([rnn_res[eq][bptt_i], rnn_res[eq_c][bptt_i]], 0).view(-1).shape)
                sub_r = self.tab_mod(tensor([]), torch.cat([rnn_res[eq][bptt_i], rnn_res[eq_c][bptt_i]], 0))
                curr_res.append(sub_r)
                # TODO change this to matrix implementation, will probably work better
#                 curr_res = torch.cat([curr_res, sub_r.view(-1)], 1) if curr_res is not None else sub_r.view(-1)
        curr_res = torch.stack(curr_res)
    
        if self.y_range is not None: #Correct
            curr_res = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(curr_res) + self.y_range[0] #Correct
        
        return curr_res #Correct
    
    def reset_h(self):
        self.h = torch.zeros(self.lyrs, self.bs, self.nh) #Correct