from fastai import *
from fastai.tabular import *
import math


class RNNTabDataset(Dataset):
    """ RNNTabDataset class:
    Creates dataset to be used in Pythor's data loader.
    The form of the dataset is the following:

        ds[ batch ][ var(0), result(1) ]

        results:
        ds[ batch ][ result(1) ] => Tensor   shape => (teams, results)

        variables:
        ds[ batch ][ var(0) ] [ cat(0), cont(1) ]

        cat:
        ds[ batch ][ var(0) ] [ cat(0) ] => Tensor   shape => (teams, jrd, cat_features)

        cont:
        ds[ batch ][ var(0) ] [ cont(0) ] => Tensor   shape => (teams, jrd, cont_features)
    """

    def __init__(self, dfs, bptt, is_valid = False):
        self.dfs = dfs
        self.bs = len(dfs)
        self.bptt = bptt
        self.is_valid = is_valid
        
        self.gen_jrds()
        self.gen_def_data()
        
        self.process()
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def __len__(self):
        return len(self.data)
    
    def gen_jrds(self):
        self.max_jrd = {}
        self.part_sum_jrd = {}
        
        self.ttl_jrds = 0
        
        #calculating max_jrd for every jrd
        for tm in range(len(self.dfs)):
            for gm in range(len(self.dfs[tm][0][0])):
                jrd, ano, tipo = self.dfs[tm][0][0][gm][5].item(), self.dfs[tm][0][0][gm][6].item(), self.dfs[tm][0][0][gm][7].item()
                if(ano not in self.max_jrd): self.max_jrd[ano] = {}
                self.max_jrd[ano][tipo] = max(self.max_jrd[ano][tipo], jrd) if tipo in self.max_jrd[ano] else jrd
                
        anos = list(self.max_jrd.keys())
        anos.sort()
        
        #part sum addition
        acum = 0
        for ano in anos:
            self.part_sum_jrd[ano] = {}
            for tipo in range(2, 0, -1):
                if(tipo in self.max_jrd[ano]):
                    self.part_sum_jrd[ano][tipo] = acum
                    acum += self.max_jrd[ano][tipo]
                    
        self.ttl_jrds = acum
                
    def gen_def_data(self):
        cat_n = len(self.dfs[0][0][0][0])
        cont_n  = len(self.dfs[0][0][1][0])
        complete_batches = self.ttl_jrds//self.bptt if not self.is_valid else 1
        print(complete_batches)
        self.data = [ [ [ torch.zeros(self.bs, self.bptt, cat_n), torch.zeros(self.bs, self.bptt, cont_n) ] , torch.zeros(self.bs, self.bptt) ] for i in range(complete_batches) ]
        
        self.inc_bptt = self.ttl_jrds % self.bptt if not self.is_valid else 0
        if(self.inc_bptt > 0):
            self.data.insert(0, [ [ torch.zeros(self.bs, self.inc_bptt, cat_n), torch.zeros(self.bs, self.inc_bptt, cont_n) ] , torch.zeros(self.bs, self.inc_bptt) ])
    
    def translate_to_position(self, jrd, ano, tipo):
        
        reg_pos = self.part_sum_jrd[ano][tipo] + jrd
        
        if(reg_pos >= self.inc_bptt):
            reg_pos -= self.inc_bptt
            reg_pos += self.bptt if self.inc_bptt != 0 else 0
            
        if(self.is_valid):
            jrd_key = list(self.max_jrd.keys())[0]
            tipo_key = list(self.max_jrd[jrd_key].keys())[0]
            reg_pos -= self.max_jrd[jrd_key][tipo_key] - (self.bptt-1)
            reg_pos += 1
        
        batch_pos = reg_pos // self.bptt
        bptt_pos = reg_pos % self.bptt
        
        return batch_pos, bptt_pos
    
    def insert_data(self, team_idx, jrd, ano, tipo, cat, cont, res):
        batch_pos, bptt_pos = self.translate_to_position(jrd, ano, tipo)
#         print(batch_pos, bptt_pos, team_idx, jrd)
#         print(batch_pos, " ", bptt_pos)
        self.data[batch_pos][0][0][team_idx][bptt_pos] = cat
        self.data[batch_pos][0][1][team_idx][bptt_pos] = cont
        self.data[batch_pos][1][team_idx][bptt_pos] = res
    
    def process(self):
        # for all teams add all games to our data
        for tm in range(self.bs):
            for gm in range(len(self.dfs[tm][1])):
                team_idx, jrd, ano, tipo = self.dfs[tm][0][0][gm][0].item(), self.dfs[tm][0][0][gm][5].item(), self.dfs[tm][0][0][gm][6].item(), self.dfs[tm][0][0][gm][7].item()
                # -1 on team so that the team works out as the batch index, same goes for jrd
                self.insert_data(team_idx-1, jrd-1, ano, tipo, self.dfs[tm][0][0][gm], self.dfs[tm][0][1][gm], self.dfs[tm][1][gm])
        
        
        #dimensions problem solution
        if(not self.is_valid): self.data.pop(0)
        # fix missing spots
        # will leave them as 0s right now
        # here we can add either the immediate next or immediate previous game if we find a -1 at self.dfs[tm][0][0][gm][0]
        # hovever this time we will leave it this way to see if this missing games can jut be ignored on the backpropagation phase.
    
    def prnt(self):
        print(self.data)