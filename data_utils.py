from fastai import *
from fastai.text import *
from fastai.tabular import *
from fastai import *

def get_col_universe_across_teams(teams, cols):
    """ Returns the universe of the elements in all the cols across all the teams.
    """
    col_set = set()
    for team_df in teams:
        for col in cols:
            for element in team_df[col].unique(): col_set.add(element)
    return col_set


def aux_create_no_shuffle_training(train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None, path:PathOrStr='.', bs:int=64,
           val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None,
           device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False, **dl_kwargs)->'DataBunch':
    """ Auxiliary, does the same as the regular data.x._bunch.create method BUT it won't shuffle the training set:
    Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`. Passes `**dl_kwargs` to `DataLoader()`
    """
    datasets = DataBunch._init_ds(train_ds, valid_ds, test_ds)
    val_bs = ifnone(val_bs, bs)
    dls = [DataLoader(d, b, shuffle=s, drop_last=s, num_workers=num_workers, **dl_kwargs) for d,b,s in zip(datasets, (bs,val_bs,val_bs,val_bs), (False,False,False,False)) if d is not None]
    return DataBunch(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

    

def aux_get_dif(a, b):
    """ Auxiliary, given 2 lists (a, b) returns a list holding the a - b opertation.
    """
    dif_arr = []
    curr_set = set()
    for x in b: curr_set.add(x)
    for x in a:
        if x not in curr_set: dif_arr.append(x)
    return dif_arr

def aux_add_missing_categories_ind_team(team_df, cat_mapping):
    """ Auxiliary, adds missing categories to all categroical variables in a given team
    """
    for cat_var, universe in cat_mapping.items():
        # Get missing categories
        cat_difs = aux_get_dif(universe, list(team_df[cat_var].unique()))
        
        # Convert variable to category, add the missing categories and reorder them in the universe order
        # Note that universe must be ordered beforehand
        team_df[cat_var] = team_df[cat_var].astype('category')
        team_df[cat_var].cat.add_categories(cat_difs, inplace = True)
        team_df[cat_var].cat.reorder_categories(universe, inplace = True)

def add_missing_categories_all_teams(teams, cat_mapping):
    """ Adds missing categories to all categroical variables in a all teams
    """
    for team_df in teams:
        aux_add_missing_categories_ind_team(team_df, cat_mapping)


def get_datasets_from_teams(teams, dep_vars, cat_names, cont_names, procs, path):
    """ Returns list of individual datasets generated for each team.
    This is done by extracting such datasets from a Tabular list.
    """
    teams_ds = []
    for team_df in teams:
        data = (TabularList.from_df(team_df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
        .split_none()
        .label_from_df(cols=dep_vars))

        data.x._bunch.create = aux_create_no_shuffle_training
        data = data.databunch(bs = len(team_df))
        teams_ds.append(next(iter(data.train_dl)))

    return teams_ds


def get_max_jrd(teams_ds, JRD_POS, ANO_POS, TIPO_POS):
    """ Returns the max jrd, ano, tipo across all the teams.
    This is not done using the df since we want the actual categorical numbers rather than the categories.
    """
    mx_jrd, mx_ano, mx_tipo = 0, 0, 100000
    # tipo E [1:apertura, 2:clausura] but the order is clausura then apertura.

    # find maximum jrd
    for tm in range(len(teams_ds)):
        lst_gm_i = len(teams_ds[tm][1])-1
        jrd, ano, tipo = teams_ds[tm][0][0][lst_gm_i][JRD_POS].item(), teams_ds[tm][0][0][lst_gm_i][ANO_POS].item(), teams_ds[tm][0][0][lst_gm_i][TIPO_POS].item()
        
        # simple comparission
        if(ano > mx_ano): mx_jrd, mx_ano, mx_tipo = jrd, ano, tipo
        elif(ano == mx_ano):
            if(tipo < mx_tipo): mx_jrd, mx_ano, mx_tipo = jrd, ano, tipo
            elif(tipo == mx_tipo):
                if(jrd > mx_jrd): mx_jrd, mx_ano, mx_tipo = jrd, ano, tipo

    return mx_jrd, mx_ano, mx_tipo

def get_valid_jrds(n_valid_bptt, mx_jrd, mx_ano, mx_tipo):
    """ Returns a set with the jrds that must be included in the validation datasets for all teams (if present).
    """
    valid_jrds = set()
    for i in range(n_valid_bptt): valid_jrds.add((mx_jrd-i, mx_ano, mx_tipo))
    return valid_jrds


def split_valid_jrds(teams_ds, valid_jrds, n_valid_bptt, JRD_POS, ANO_POS, TIPO_POS):
    """ Removes the validation jrds from each team in the team_ds and returns a new list
    of ds that contain such jrds; such list will be the list of validation ds for each team.
    """
    teams_ds_valid = []
    
    for tm in range(len(teams_ds)):
        
        lst_gm_i = len(teams_ds[tm][1])-1
        first_to_use = lst_gm_i+1  # defines where we will our split happen in this team

        for i in range(n_valid_bptt):
            jrd, ano, tipo = teams_ds[tm][0][0][lst_gm_i-i][JRD_POS].item(), teams_ds[tm][0][0][lst_gm_i-i][ANO_POS].item(), teams_ds[tm][0][0][lst_gm_i-i][TIPO_POS].item()
            if (jrd, ano, tipo) in valid_jrds:
                first_to_use -= 1

        # adding validation part to validation dataset
        teams_ds_valid.append( ((teams_ds[tm][0][0][first_to_use :],teams_ds[tm][0][1][first_to_use :]), teams_ds[tm][1][first_to_use :]) )
        # removing validation part from regular (training) dataset
        teams_ds[tm] = ((teams_ds[tm][0][0][:first_to_use ],teams_ds[tm][0][1][:first_to_use ]), teams_ds[tm][1][:first_to_use ])

    return teams_ds_valid