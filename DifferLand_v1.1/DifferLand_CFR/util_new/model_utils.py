# The following functions are adapted from the DifferLand package
import jax
import jax.numpy as jnp
from jax.lax import fori_loop
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, r'../../DifferLand_v1.1')
from DifferLand_CFR.model.DALEC_990_parinfo import dalec990_param_parmin, dalec990_param_parmax
from DifferLand_CFR.model.DALEC_990_parinfo import dalec990_pool_parmax, dalec990_pool_parmin
from DifferLand_CFR.util.normalization import unnormalize_parameters, normalize_parameters
from DifferLand_CFR.util.init_mlp_params import init_mlp_params
from DifferLand_CFR.util.normalization import par2nor
from DifferLand_CFR.optimization.loss_functions import compute_nnse


def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
def reconstruct_drivers(landvalue, start_year, end_year, total_years):
    driver_file_path = f"../../data/CRUJRA/drivers/drivers_{landvalue}.nc"
    driver_ds = xr.open_dataset(driver_file_path)


    cycle_years = list(range(start_year, end_year + 1))
    total_cycle_years = len(cycle_years)
    n_cycles = total_years // total_cycle_years
    remain_years = total_years % total_cycle_years
    year_list = cycle_years * n_cycles + cycle_years[:remain_years]
    yearly_datasets = []
    all_dates = []
    for year in year_list:
        start_idx = (year - 1901) * 365
        if is_leap(year):
            part1 = driver_ds.isel(time=slice(start_idx, start_idx + 59))
            feb28 = driver_ds.isel(time=start_idx + 58)
            insert_day = feb28.copy()
            part2 = driver_ds.isel(time=slice(start_idx + 59, start_idx + 365))
            year_ds = xr.concat([part1, insert_day, part2], dim='time')
            start_date = datetime(year, 1, 1)
            dates = [start_date + timedelta(days=i) for i in range(366)]
        else:
            year_ds = driver_ds.isel(time=slice(start_idx, start_idx + 365))
            start_date = datetime(year, 1, 1)
            dates = [start_date + timedelta(days=i) for i in range(365)]
        yearly_datasets.append(year_ds)
        all_dates.extend(dates)
    new_ds = xr.concat(yearly_datasets, dim='time')
    dates_array = pd.DatetimeIndex(all_dates)
    doy = dates_array.dayofyear.values
    new_ds['DOY'] = ('time', doy)
    new_ds = new_ds.assign_coords(time=np.arange(1, len(dates_array) + 1))
    return new_ds

def generate_met_matrix(landvalue, start_year=1985, end_year=2020, total_year=100):
    """Generate the met matrix from the driver dataset"""
    driver_ds = reconstruct_drivers(landvalue, start_year, end_year, total_year)
    time = driver_ds.time.values
    t_min = driver_ds.T2M_MIN.values
    t_max = driver_ds.T2M_MAX.values
    rad = driver_ds.SSRD.values
    ca = driver_ds.CO2.values
    doy = driver_ds.DOY.values
    vpd =  driver_ds.VPD.values
    precipitation = driver_ds.TOTAL_PREC.values
    burned_area = np.full(time.shape[0], 0)
    lat = np.full(time.shape, driver_ds.LAT.values)
    delta_t = np.full(time.shape[0], time[1]-time[0])
    t_mean = np.full(time.shape[0], np.mean((t_min+t_max)/2))
    mean_precipitation = np.full(time.shape[0], np.mean(precipitation))
    t_norm = ((t_max + t_min)/2 - t_mean) / np.std((t_max + t_min)/2)
    rad_norm = (rad - np.mean(rad)) / np.std(rad)
    vpd_norm = (vpd - np.mean(vpd)) / np.std(vpd)
    ca_norm = (ca - np.mean(ca)) / np.std(ca)
    end_of_year_sel = jnp.where(driver_ds.DOY.values==driver_ds.DOY.values[0].item())[0][1:] - 1
    end_of_year_index = np.full(len(driver_ds.DOY.values), False)
    end_of_year_index[end_of_year_sel] = True
    end_of_year_arr = jnp.array(end_of_year_index).astype(np.float32)

    met_matrix = jnp.array([time, t_min, t_max, rad, ca, doy, burned_area,
                   vpd, precipitation, lat, delta_t, t_mean, mean_precipitation,
                            t_norm, rad_norm, vpd_norm, ca_norm, end_of_year_arr]).T
    return met_matrix


def unnormalize(normalized_parameters):
    """Convert model parameters from the real space to the physcial range"""
    return unnormalize_parameters(normalized_parameters, param_parmin=dalec990_param_parmin, param_parmax=dalec990_param_parmax)
def normalize(unnormalized_parameters):
    """Convert model parameters from the physical range to the real space"""
    return normalize_parameters(unnormalized_parameters, param_parmin=dalec990_param_parmin, param_parmax=dalec990_param_parmax)
def unnormalize_pools(normalized_pools):
    """Convert pool values from the real space to the physical value range"""
    return unnormalize_parameters(normalized_pools, param_parmin=dalec990_pool_parmin, param_parmax=dalec990_pool_parmax)
def normalize_pools(unnormalized_pools):
    """Convert pool values from the the physical value range to real space"""
    return normalize_parameters(unnormalized_pools, param_parmin=dalec990_pool_parmin, param_parmax=dalec990_pool_parmax)


def initialize_nn_params(stress_type, random_seed=np.random.randint(9999999)):
    """Initialize parameters for neural network subcomponents"""
    if stress_type == "baseline":
        gpp_params = init_mlp_params([1,1,1], n=random_seed)
    elif stress_type == "default":
        gpp_params = init_mlp_params([1,1,1], n=random_seed)
    elif stress_type == "nn_paw":
        gpp_params = init_mlp_params([1,10,1], n=random_seed)
    elif stress_type == "nn_whole_no_lai":
        gpp_params = init_mlp_params([5,10,10,2], n=random_seed)
    elif stress_type == "nn_whole":
        gpp_params = init_mlp_params([6,10,10,2], n=random_seed)
    elif stress_type == "gpp_acm_et_nn":
        beta_params = init_mlp_params([1, 10, 1], n=random_seed)
        et_params = init_mlp_params([7,10,10,1], n=random_seed)
        gpp_params = (beta_params, et_params)
    return gpp_params

def initialize_physical_parameters(ce_opt, lcma_opt, cue_opt, model, random_seed=np.random.randint(9999999)):
    """Initialize parameters for the physical model parameters in DALEC"""
    key = jax.random.PRNGKey(random_seed)
    key, subkey = jax.random.split(key)
    pool_initial = jax.random.normal(subkey, dalec990_pool_parmin.shape)
    del subkey

    key, subkey = jax.random.split(key)
    param_initial = jax.random.normal(subkey, dalec990_param_parmin.shape)

    if ce_opt >= 5:
        param_initial = param_initial.at[10].set(par2nor(ce_opt, 5, 50))
    if lcma_opt >= 5:
        param_initial = param_initial.at[16].set(par2nor(lcma_opt, 5, 200))
    
    if cue_opt >= 0.2:
        param_initial = param_initial.at[1].set(par2nor(1-cue_opt, 0.2, 0.8))
    else:
        param_initial = param_initial.at[1].set(par2nor(0.5, 0.2, 0.8))

    pool_initial = pool_initial.at[6].set(par2nor(500, model.parmin.initial_PAW, model.parmax.initial_PAW))
    
    del subkey
    
    return pool_initial, param_initial

def get_stress_type(model_configuration):
    """Get stress type str for the model_configuration index"""
    if model_configuration == 1:
        return "baseline"
    elif model_configuration == 2:
        return "default"
    elif model_configuration == 3:
        return "nn_paw"
    elif model_configuration == 4:
        return "nn_whole_no_lai"
    elif model_configuration == 5:
        return "nn_whole"
    elif model_configuration == 6:
        return "gpp_acm_et_nn"
    else:
        print("ERROR: Model type must be between 1-6:")
        print("1) baseline")
        print("2) JS-beta")
        print("3) nn-beta")
        print("4) GPP&ET(NN)_MET")
        print("5) GPP&ET(NN)_MET+LAI")
        print("6) GPP(ACM)_ET(NN)")
        sys.exit(1)