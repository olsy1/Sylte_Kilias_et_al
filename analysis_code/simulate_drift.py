
import numpy as np
from scipy import stats
import get_data
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
from typing import List, Tuple
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import random
from sklearn.decomposition import PCA


def apply_drift(hist: np.ndarray, drift_percentage: np.ndarray, direction: np.ndarray, amplitude_factor: np.ndarray) -> np.ndarray:
    n_neurons, n_bins = hist.shape
    drifted_hist = np.zeros_like(hist)
    for neuron in range(n_neurons):
        shift = int(drift_percentage[neuron] * n_bins / 100) 
        if direction[neuron] == 'right':
            shifted = np.roll(hist[neuron], shift)
        else:
            shifted = np.roll(hist[neuron], -shift)
        
        
        drifted_hist[neuron] = shifted
    return drifted_hist


def apply_permutation(hist):
    n_neurons, n_bins = hist.shape
    permuted_hist = np.zeros_like(hist)
    
    for neuron in range(n_neurons):
        permuted_hist[neuron] = np.random.uniform(1,30, n_bins)

    return permuted_hist


def update_amplitude(amplitude_factor: np.ndarray, amplitude_direction: np.ndarray,
                     amplitude_change_speed: np.ndarray) -> np.ndarray:

    amplitude_factor += amplitude_direction * amplitude_change_speed


    mask_min = amplitude_factor <= 0
    amplitude_factor[mask_min] = 0
    amplitude_direction[mask_min] = 1  


    mask_max = amplitude_factor >= 4
    amplitude_factor[mask_max] = 4
    amplitude_direction[mask_max] = -1 

    return amplitude_factor



def simulate_variable_circular_drift(transients, lin_pos, n_days, max_drift):
    drifted_data = [transients]
    for day in range(1, n_days):
        run_starts, run_ends = get_data.identify_runs(lin_pos)
        daily_drift_direction = np.random.choice(['left', 'right'], len(transients))
        daily_drift_percentage = np.random.uniform(1, max_drift, len(transients))
        
        drifted_day_data = np.zeros_like(transients)
        for start, end in zip(run_starts, run_ends):
            run_data = transients[:, start:end]
            drifted_run_data = apply_drift(run_data, daily_drift_percentage, daily_drift_direction)
            drifted_day_data[:, start:end] = drifted_run_data
        drifted_data.append(drifted_day_data)
    return drifted_data


def generate_rotation_matrix(dim, angle, state=0):
    random_state = np.random.RandomState(state)
    orthogonal_matrix = stats.special_ortho_group.rvs(dim, random_state=random_state)
    return orthogonal_matrix #expm(angle * skew_symmetric_matrix)


def gradual_rotation_origin(data, n_days, max_angle):
    dim = data.shape[0] 
    total_rotation_matrix = generate_rotation_matrix(dim, max_angle)
    incremental_rotation_matrix = expm(logm(total_rotation_matrix) / n_days)
    
    rotated_data = [data]
    for day in range(1, n_days):
        rotated_day_data = np.dot(rotated_data[-1].T,  incremental_rotation_matrix).T 
        rotated_data.append(rotated_day_data)
    return rotated_data

def gradual_rotation(data, n_days, angle, state=0):
    if n_days < 1:
        return []
    if n_days == 1:
        return [data]

    dim = data.shape[0]
    total_R = generate_rotation_matrix(dim, angle, state=state)
    incremental_R = expm(logm(total_R) / (n_days - 1))

    data_mean = np.mean(data, axis=1, keepdims=True)     
    centered = data - data_mean                           

    rotated_data = []
    current_R = np.eye(dim)
    for day in range(n_days):
        if day > 0:
            current_R = incremental_R @ current_R
        rotated_data.append(current_R @ centered + data_mean)

    return rotated_data


def generate_scaling_matrix(dim, scale_factor):
    return np.eye(dim) * scale_factor

def gradual_scaling(data, n_days, max_scale_factor):
    dim = data.shape[0]
    total_scaling_factor = max_scale_factor
    daily_scaling_factor = total_scaling_factor ** (1 / n_days)
    
    scaled_data = [data]
    for day in range(1, n_days):
        scaling_matrix = generate_scaling_matrix(dim, daily_scaling_factor)
        scaled_day_data = np.dot(scaling_matrix, scaled_data[-1])
        scaled_data.append(scaled_day_data)
    return scaled_data




def improved_simulate_drift(
    transients: np.ndarray,
    lin_pos: np.ndarray,
    n_days: int,
    max_remap_prob: float = 0.3,
    amplitude_drift_prob: float = 0.3,
    amplitude_change_scale: float = 0.3,
    smoothness: float = 0.00,
    use_uniform_density: bool = False,
    n_position_bins: int = 20,
    shift_noise_std: float = 0.0,
    amplitude_noise_std: float = 0.0,
    global_seed: int = 0 
) -> List[np.ndarray]:
    np.random.seed(global_seed)
    random.seed(global_seed)

    n_neurons, n_timepoints = transients.shape
    
    drifted_data = [transients.copy()]
    
    density = np.ones((n_days, n_position_bins))
    
    initial_tuning, _ = get_data.average_spatial_tuning_function(transients, lin_pos)
    
    cumulative_shifts = np.zeros(n_neurons, dtype=float)
    cumulative_amplitude_changes = np.ones(n_neurons)
    
    def ensure_valid_density(dens):
        dens[dens < 0] = 0
        dens[np.isnan(dens)] = 0
        dens = np.maximum(dens, 0)
        return dens / np.sum(dens)
    

    run_starts, run_ends = get_data.identify_runs(lin_pos)
    
    def apply_drift_to_run(run_data, run_pos, shifts, amplitudes, day_seed):
        day_rng = np.random.RandomState(day_seed)
        drifted_run_data = np.zeros_like(run_data)
        run_length = len(run_pos)
        for neuron in range(n_neurons):
            noisy_shift = shifts[neuron] + day_rng.laplace(0, shift_noise_std)
            shift = int(noisy_shift * run_length / n_position_bins)
            noisy_amplitude = amplitudes[neuron] * np.exp(day_rng.normal(0, amplitude_noise_std))
            drifted_run_data[neuron] = np.roll(run_data[neuron], shift) * noisy_amplitude
        return drifted_run_data
    
    day_seed = random.randint(0, 2**32 - 1)
    day_rng = np.random.RandomState(day_seed)

    if not use_uniform_density:
        density[0] = day_rng.rand(n_position_bins)
        density[0] = ensure_valid_density(density[0])
    else:
        density[0] = np.ones(n_position_bins) / n_position_bins
    
    for day in range(1, n_days):
        day_seed = random.randint(0, 2**32 - 1)
        day_rng = np.random.RandomState(day_seed)
        
        if not use_uniform_density:
            step = day_rng.normal(0, smoothness, n_position_bins)
            density[day] = density[day-1] + step
            density[day][density[day] < 0] = -density[day][density[day] < 0]
            density[day][density[day] > 1] = 2 - density[day][density[day] > 1]
            density[day] = ensure_valid_density(density[day])
        else:
            density[day] = np.ones(n_position_bins) / n_position_bins

        # determine which neurons will drift
        remap_prob = day_rng.uniform(max_remap_prob, max_remap_prob)
        neurons_to_remap = day_rng.choice([False, True], size=n_neurons, p=[1-remap_prob, remap_prob])
        
        # determine which neurons will have amplitude drift
        amplitude_drift_mask = day_rng.random(n_neurons) < amplitude_drift_prob
        
        # drift
        for neuron in np.where(neurons_to_remap)[0]:
            new_peak = day_rng.choice(n_position_bins, p=density[day])
            current_peak = (np.argmax(initial_tuning[neuron]) + int(cumulative_shifts[neuron])) % n_position_bins
            shift = (new_peak - current_peak) % n_position_bins  #ensure positive shift
            cumulative_shifts[neuron] = (cumulative_shifts[neuron] + shift) % n_position_bins
        
        # amplitude drift
        amplitude_changes = np.ones(n_neurons)
        drifting_neurons = np.where(amplitude_drift_mask)[0]
        changes = day_rng.normal(0, amplitude_change_scale, len(drifting_neurons))
        amplitude_changes[drifting_neurons] = np.exp(changes - (amplitude_change_scale**2)/2)
        cumulative_amplitude_changes *= amplitude_changes
        
        # ensure amplitudes are positive
        cumulative_amplitude_changes = np.maximum(cumulative_amplitude_changes, 1e-6)
        
        # apply cumulative shifts and amplitude changes to neural activity
        drifted_day_data = np.zeros_like(transients)
        for start, end in zip(run_starts, run_ends):
            run_data = transients[:, start:end]
            run_pos = lin_pos[start:end]
            drifted_run_data = apply_drift_to_run(run_data, run_pos, cumulative_shifts, cumulative_amplitude_changes, day_seed)
            drifted_day_data[:, start:end] = drifted_run_data
        
        drifted_data.append(drifted_day_data)
    
    return drifted_data


def generate_density_function(n_positions: int, n_days: int, smoothness: float = 0.1) -> np.ndarray:
    density = np.ones((n_days, n_positions))
    for day in range(1, n_days):
        walk = np.cumsum(np.random.normal(0, smoothness, n_positions))
        density[day] = norm.pdf(np.linspace(0, 1, n_positions), loc=0.5, scale=0.5) + walk
        density[day] = np.maximum(density[day], 0) 
        density[day] /= density[day].sum()
    return density


def simulate_ou_drift_on_pcs(
    transients: np.ndarray,
    lin_pos: np.ndarray,
    n_days: int,
    n_pcs: int = 5,
    theta: float = 0.1,
    sigma: float = 0.3,
    drift_scale: float = 1.0,
    use_activity_pca: bool = False,
    global_seed: int = 0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    np.random.seed(global_seed)
    random.seed(global_seed)
    
    n_neurons, n_timepoints = transients.shape
    
    initial_tuning, _ = get_data.average_spatial_tuning_function(transients, lin_pos)
    n_position_bins = initial_tuning.shape[1]
    
    if use_activity_pca:
        #PCA on full activity traces
        max_components = min(n_neurons, n_timepoints)
        pca = PCA(n_components=max_components)
        pca.fit(transients.T)
        all_coeffs = pca.transform(transients.T)      
    else:
        # PCA on spatial tuning functions 
        max_components = min(n_neurons, n_position_bins)
        pca = PCA(n_components=max_components)
        pca.fit(initial_tuning)
        all_coeffs = pca.transform(initial_tuning)  
    
    drifting_coeffs = all_coeffs[:, :n_pcs].copy()  # First n_pcs  drift
    stable_coeffs = all_coeffs[:, n_pcs:].copy()    
    
    initial_drifting_coeffs = drifting_coeffs.copy()
    drift_data = [transients.copy()]  
    maps = [initial_tuning.copy()]    
    
    for day in range(1, n_days):
        day_rng = np.random.RandomState(global_seed + day + n_neurons)

        if use_activity_pca:
            noise = day_rng.randn(n_timepoints, n_pcs)
        else:
            noise = day_rng.randn(n_neurons, n_pcs)
            
        drifting_coeffs = (drifting_coeffs + 
                           theta * (initial_drifting_coeffs - drifting_coeffs) +
                           sigma * drift_scale * noise)


        combined_coeffs = np.concatenate([drifting_coeffs, stable_coeffs], axis=1)

        if use_activity_pca:
            drifted_activity = pca.inverse_transform(combined_coeffs).T  
            drifted_tuning, _ = get_data.average_spatial_tuning_function(drifted_activity, lin_pos)
            
        else:
            drifted_tuning = pca.inverse_transform(combined_coeffs)
            
    
        maps.append(drifted_tuning)
        
        if use_activity_pca:
            drift_data.append(drifted_activity)
        else:
            drift_data.append(transients.copy())
    
    return drift_data, maps


