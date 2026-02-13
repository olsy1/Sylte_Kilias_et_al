
import numpy as np
import get_data
import matplotlib.pyplot as plt
import population_activity as pop
import helper_functions as hf
import copy
import place_cell as pc
import analysis
import decoding2
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def compute_null_potent_fractions(res_align, data1_TN, pos1_T):
    P_pot, P_null, rank_r, S = make_potent_null_projectors(data1_TN, pos1_T)
    
    frac_null, frac_potent = [], []
    E_null_list, E_potent_list = [], []
    
    X1 = res_align[0][0]  
    for d in range(len(res_align)):
        Xd_native = res_align[d][9]  # (N,B) 
        Delta_total = Xd_native - X1  

        Delta_pot = P_pot @ Delta_total
        Delta_null = P_null @ Delta_total
        
        E_pot = float(np.sum(Delta_pot**2))
        E_null = float(np.sum(Delta_null**2))
        denom = E_pot + E_null + 1e-12
        
        frac_potent.append(E_pot / denom)
        frac_null.append(E_null / denom)
        E_potent_list.append(E_pot)
        E_null_list.append(E_null)
    
    return {
        'frac_null': frac_null,
        'frac_potent': frac_potent,
        'E_null': E_null_list,
        'E_potent': E_potent_list,
        'rank': rank_r,
        'singvals': S,
    }


def null_rigid_vs_residual_within_null(res_align, data1_TN, pos1_T):
    """
    decompose drift into null/potent x rigid/residual
    """
    P_pot, P_null, _, _ = make_potent_null_projectors(data1_TN, pos1_T)
    
    out = {
        'null_rigid_within_null': [],
        'null_resid_within_null': [],
        'potent_rigid_within_potent': [], 
        'potent_resid_within_potent': []   
    }
    
    X1 = res_align[0][0]
    
    for d in range(len(res_align)):
        Xd_native = res_align[d][9]
        ts = res_align[d][8]
        
        mu1 = ts['mu1'].reshape(-1,1)
        mu2 = ts['mu2'].reshape(-1,1)
        s1 = float(ts['s1'])
        s2 = float(ts['s2'])
        R_high = ts['R_high']
        
        predicted_Xd = mu2 + s2 * (R_high @ ((X1 - mu1) / s1))
        
        Delta_rigid = predicted_Xd - X1
        E_residual = Xd_native - predicted_Xd
        
        Enull_rigid = float(np.sum((P_null @ Delta_rigid)**2))
        Enull_resid = float(np.sum((P_null @ E_residual)**2))
        denom_null = Enull_rigid + Enull_resid + 1e-12
        
        out['null_rigid_within_null'].append(Enull_rigid / denom_null)
        out['null_resid_within_null'].append(Enull_resid / denom_null)
        
        Epot_rigid = float(np.sum((P_pot @ Delta_rigid)**2))
        Epot_resid = float(np.sum((P_pot @ E_residual)**2))
        denom_pot = Epot_rigid + Epot_resid + 1e-12
        
        out['potent_rigid_within_potent'].append(Epot_rigid / denom_pot)
        out['potent_resid_within_potent'].append(Epot_resid / denom_pot)
    
    return out


def make_potent_null_projectors(
    X_TN,                 # (T, N) 
    pos_T,                
    ridge_lambda=10,     
    svd_rel_tol=1e-3,    
):

    _, y_1based = decoding2.scale_and_digitize(pos_T)            
    n_bins = int(np.max(y_1based))
    dec = decoding2.PositionDecoder(n_positions=n_bins, ridge_lambda=ridge_lambda)
    dec.fit(X_TN, y_1based)

    W = dec.weights.T   # (B, N)

    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r = int(np.sum(S >= svd_rel_tol * S[0])) if S.size else 0
    V = Vt.T  # (N, N)

    P_pot = V[:, :r] @ V[:, :r].T if r > 0 else np.zeros((V.shape[0], V.shape[0]))
    P_null = np.eye(V.shape[0]) - P_pot

    return P_pot, P_null, r, S


def frob2(M):
    return float(np.sum(M*M))

def rigid_prediction_from_day1(X1_NB, ts):
    mu1 = ts['mu1'].reshape(-1,1)
    mu2 = ts['mu2'].reshape(-1,1)
    s1  = float(ts['s1'])
    s2  = float(ts['s2'])
    R   = ts['R_high'] 
    return mu2 + s2 * (R @ ((X1_NB - mu1) / s1))


def split_drift_quadrants(res_align, data1_TN, pos1_T):
    N, B = res_align[0][0].shape

    P_pot, P_null, rank_r, S = make_potent_null_projectors(data1_TN, pos1_T)

    fracs = {'PR': [], 'NR': [], 'PRes': [], 'NRes': []}
    for d in range(len(res_align)):
        X1 = res_align[0][0]   # (N,B)
        Xd = res_align[d][9]   # (N,B)
        ts = res_align[d][8]

        Xd_hat = rigid_prediction_from_day1(X1, ts)

        Delta_rigid = Xd_hat - X1
        E_res       = Xd     - Xd_hat

        E_PR   = frob2(P_pot  @ Delta_rigid)
        E_NR   = frob2(P_null @ Delta_rigid)
        E_PRes = frob2(P_pot  @ E_res)
        E_NRes = frob2(P_null @ E_res)

        S_four = E_PR + E_NR + E_PRes + E_NRes + 1e-12 
        fracs['PR'].append(E_PR   / S_four)
        fracs['NR'].append(E_NR   / S_four)
        fracs['PRes'].append(E_PRes/ S_four)
        fracs['NRes'].append(E_NRes/ S_four)

    fracs['rank'] = rank_r
    fracs['singvals'] = S
    return fracs











