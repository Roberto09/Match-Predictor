import numpy as np
import pandas as pd

def format_columns(df):
    """ Renames columns of current df to more descriptive ones
    """
    rename_loc = {'JG_x':'L_JG',
               'JE_x':'L_JE',
               'JP_x':'L_JP',
               'GF_x':'L_GF',
               'GC_x':'L_GC',
               'DIF_x':'L_DIF',
               'PTS_x':'L_PTS'}

    rename_vis = {'JG_y':'V_JG',
                'JE_y':'V_JE',
                'JP_y':'V_JP',
                'GF_y':'V_GF',
                'GC_y':'V_GC',
                'DIF_y':'V_DIF',
                'PTS_y':'V_PTS'}

    df.rename(columns = rename_loc, inplace = True)
    df.rename(columns = rename_vis, inplace = True)


def sort_df(df):
    """ Returns sorted df in ascending order.
    sort_val = YYYY . T 0 JJJ
    """
    df['sort_val'] = df.ANO + df.JRD/1000 + (df.TIPO == 'apertura')/10
    s_df = df.sort_values('sort_val').drop('sort_val', 1)
    s_df.reset_index(drop=True, inplace=True)
    return s_df

def aggregate_prev_results(teams):
    """ Creates new column with prev results
    NOTE this assumes the teams are already sorted
    """
    for team_df in teams:
        team_df['PREV_RES'] = team_df.RES
        team_df.PREV_RES = team_df.PREV_RES.shift(1, fill_value = 1)

        team_df['PREV_G_EQ'] = team_df.G_EQ
        team_df.PREV_G_EQ = team_df.PREV_G_EQ.shift(1, fill_value = 0.0)

        team_df['PREV_G_EQC'] = team_df.G_EQC
        team_df.PREV_G_EQC = team_df.PREV_G_EQC.shift(1, fill_value = 0.0)


def get_ind_teams(df):
    """ Returns array of individual sorted teams changed to a 'Team, ContraryTeam' context
    by adding the 'ES_LOC' variable.
    """

    teams = []

    local_LOC_columns_rename = {col:'EQ_' + col[2:] for col in df.columns if col[0:2] == 'L_'}
    local_VIS_columns_rename = {col:'EQC_' + col[2:] for col in df.columns if col[0:2] == 'V_'}
    
    vis_LOC_columns_rename = {col:'EQC_' + col[2:] for col in df.columns if col[0:2] == 'L_'}
    vis_VIS_columns_rename = {col:'EQ_' + col[2:] for col in df.columns if col[0:2] == 'V_'}


    for name in df.LOC.unique():
        # Team as local setup
        curr_df_loc = df[df.LOC == name].copy()
        curr_df_loc["ES_LOC"] = True
        curr_df_loc['EQ'] = curr_df_loc.LOC; del curr_df_loc['LOC']
        curr_df_loc['EQC'] = curr_df_loc.VIS; del curr_df_loc['VIS']
        curr_df_loc['G_EQ'] = curr_df_loc.GL; del curr_df_loc['GL']
        curr_df_loc['G_EQC'] = curr_df_loc.GV; del curr_df_loc['GV']
        
        curr_df_loc.rename(columns = local_LOC_columns_rename, inplace = True)
        curr_df_loc.rename(columns = local_VIS_columns_rename, inplace = True)

        # Team as visitor setup
        curr_df_vis = df[df.VIS == name].copy()
        curr_df_vis["ES_LOC"] = False
        curr_df_vis['EQ'] = curr_df_vis.VIS; del curr_df_vis['VIS']
        curr_df_vis['EQC'] = curr_df_vis.LOC; del curr_df_vis['LOC']
        curr_df_vis['G_EQ'] = curr_df_vis.GV; del curr_df_vis['GV']
        curr_df_vis['G_EQC'] = curr_df_vis.GL; del curr_df_vis['GL']
        # Fliping RES for team visitor
        curr_df_vis['AUX_RES'] = curr_df_vis['RES']
        curr_df_vis.loc[curr_df_vis.AUX_RES == 0, 'RES'] = 2
        curr_df_vis.loc[curr_df_vis.AUX_RES == 2, 'RES'] = 0
        del curr_df_vis['AUX_RES']

        curr_df_vis.rename(columns = vis_LOC_columns_rename, inplace = True)
        curr_df_vis.rename(columns = vis_VIS_columns_rename, inplace = True)

        # Concatenating them
        curr_df = pd.concat([curr_df_loc, curr_df_vis]).copy()
        curr_df = sort_df(curr_df)
        teams.append(curr_df)
    
    return teams


def aux_verify_jrd_uniqueness(team_df):
    """ Auxilliary, returns True if a team has all unique jornadas
    """
    x = team_df.groupby(['ANO', 'TIPO', 'JRD']).size().reset_index().rename(columns={0:'count'})
    return len(x['count'].unique()) == 1

def aux_remove_ind_duplicate_jrd(team_df, ano, tipo, jrd):
    """ Auxiliary, removes duplicate jornadas in a team df with the given parameters.
    It only leaves the last unique jornada.
    """
    rows_to_del = team_df.index[(team_df.ANO == ano) & (team_df.TIPO == tipo) & (team_df.JRD == jrd)].to_list()
    rows_to_del.sort()
    rows_to_del.pop()

    for ri in rows_to_del:
        team_df.drop([ri], axis=0, inplace=True)

def aux_remove_duplicate_jrds(team_df):
    """ Auxiliary, removes all duplicate jrds from a team and finally resets the indexes
    """
    x = team_df.groupby(['ANO', 'TIPO', 'JRD']).size().reset_index().rename(columns={0:'count'})
    x = x[x['count'] > 1]
    for index, row in x.iterrows():
        aux_remove_ind_duplicate_jrd(team_df, row['ANO'], row['TIPO'], row['JRD'])

    team_df.reset_index(drop = True, inplace = True)


def remove_duplicate_jrds(teams):
    """ Removes all duplicated jrds from all the teams.
    Returns False if any team had duplicated jrds
    """
    not_has_dupl = True
    for team_df in teams:
        if not aux_verify_jrd_uniqueness(team_df):
            print(f'{team_df.EQ.unique()[0]} has duplicate jrds -> will remove them')
            aux_remove_duplicate_jrds(team_df)
            not_has_dupl = False

    return not_has_dupl


def get_games_team(teams):
    """ Returns a dictionary {team : total games}
    """
    t_info = {}
    for team in teams:
        t_info[team.EQ.unique()[0]] = len(team)
    
    return t_info

def get_teams_split_games(teams, games_bound):
    """ Given a bound returns 2 separate dictionaries
    one with all the teams with more or equal games than the game bound
    and another dictionary with the rest.
    """
    above_bound = {} # >=
    below_bound = {} # <
    for team_df in teams:
        if(len(team_df) >= games_bound): above_bound[team_df.EQ.unique()[0]] = len(team_df)
        else: below_bound[team_df.EQ.unique()[0]] = len(team_df)
    return above_bound, below_bound