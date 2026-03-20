import jax.numpy as jnp
import jax
from jax.lax import fori_loop
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, r'../../DALEC/DifferLand_v1.1')
from DifferLand_CFR.util_new.model_utils import normalize_pools, generate_met_matrix
from DifferLand_CFR.util_new.transloc import land2latlon
from DifferLand_CFR.optimization.loss_functions import compute_nnse

def validate_and_plot_results(model, param_state, met_matrix_full, data_pack, config_pack, NoBM=False, SnapshotBM=False):
    """
    Performs full validation, calculates metrics, and generates a detailed comparison plot.
    The plotting section is written explicitly for each panel for clarity.

    Args:
        model: The initialized DALEC990 model object.
        param_state: The final, optimized parameter state tuple.
        met_matrix_full: The 100-year meteorological matrix for calibration.
        data_pack: A dictionary containing all observational data arrays.
        config_pack: A dictionary containing configuration like land_value, years, etc.

    Returns:
        A tuple containing (metrics_dict, figure_object).
    """
    land_value = config_pack['land_value']
    lat, lon = land2latlon(land_value)
    
    final_param_norm, pool_initial_norm, final_gpp_params = param_state
    bm_obs_cal, bm_obs_val = data_pack['bm_cal'], data_pack['bm_val']
    gpp_obs_cal, gpp_obs_val = data_pack['gpp_cal'], data_pack['gpp_val']
    et_obs_cal, et_obs_val = data_pack['et_cal'], data_pack['et_val']
    reco_obs_cal, reco_obs_val = data_pack['reco_cal'], data_pack['reco_val']
    lai_obs_cal, lai_obs_val = data_pack['lai_cal'], data_pack['lai_val']
    reco_days_cal, reco_days_val = data_pack['reco_days_cal'], data_pack['reco_days_val']
    lai_days_cal, lai_days_val = data_pack['lai_days_cal'], data_pack['lai_days_val']

    # --- Step 1&2: Running calibration & validation simulations ---
    final_state_unnorm, output_matrix_100yr = model.forward(final_param_norm, pool_initial_norm, final_gpp_params, met_matrix_full)
    calib_output = output_matrix_100yr[-model.calib_flux_days:, :]
    
    pool_initial_for_validation = normalize_pools(final_state_unnorm)
    validation_duration = config_pack['valid_end_year'] - config_pack['valid_start_year'] + 1
    met_matrix_val = generate_met_matrix(config_pack['land_value'], start_year=config_pack['valid_start_year'], end_year=config_pack['valid_end_year'], total_year=validation_duration)
    _, output_matrix_val = model.forward(final_param_norm, pool_initial_for_validation, final_gpp_params, met_matrix_val)
    
    # --- Step 3: Processing model outputs to match observation scales ---
    if not NoBM:
        if not SnapshotBM:
            modeled_biomass_daily_train = ((output_matrix_100yr[:, model.pfn.next_labile_pool] + output_matrix_100yr[:, model.pfn.next_foliar_pool] + 
                        output_matrix_100yr[:, model.pfn.next_root_pool] + output_matrix_100yr[:, model.pfn.next_wood_pool]) * 2) / 100
            modeled_biomass_annual_train = modeled_biomass_daily_train[model.annual_end_indices]

            modeled_biomass_daily_test = ((output_matrix_val[:, model.pfn.next_labile_pool] + output_matrix_val[:, model.pfn.next_foliar_pool] + 
                        output_matrix_val[:, model.pfn.next_root_pool] + output_matrix_val[:, model.pfn.next_wood_pool]) * 2) / 100
            cycle_years = list(range(config_pack['valid_start_year'], config_pack['valid_end_year'] + 1))
            cycle_year_lengths = [366 if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0) else 365 for y in cycle_years]
            num_cycles = validation_duration // len(cycle_years)
            remaining_years = validation_duration % len(cycle_years)
            year_lengths = cycle_year_lengths * num_cycles + cycle_year_lengths[:remaining_years]
            annual_start_indices = [0] + [sum(year_lengths[:i]) for i in range(1, len(year_lengths)+1)]
            annual_end_indices = (jnp.array(annual_start_indices[1:], dtype=jnp.int32) - 1)
            modeled_biomass_annual_test = modeled_biomass_daily_test[annual_end_indices]

            # bm_obs = jnp.concatenate((jnp.array([0.]), bm_obs_cal, bm_obs_val))
            # modeled_biomass_annual = jnp.concatenate((jnp.array([0.]), modeled_biomass_annual_train, modeled_biomass_annual_test))
            bm_obs = jnp.concatenate((bm_obs_cal, bm_obs_val))
            modeled_biomass_annual = jnp.concatenate((modeled_biomass_annual_train, modeled_biomass_annual_test))

    # Process RECO (periodic average)
    modeled_reco_daily_train = calib_output[:, model.pfn.nee] + calib_output[:, model.pfn.gpp]
    num_reco_obs_train = reco_obs_cal.shape[0]
    reco_valid_condition_train = (reco_days_cal >= 0)
    valid_reco_indices_train = jnp.where(reco_valid_condition_train, reco_days_cal, -1)
    start_indices_train = jnp.concatenate([jnp.array([0]), valid_reco_indices_train[:-1] + 1])
    end_indices_train = valid_reco_indices_train
    segment_ids_train = jnp.full(model.calib_flux_days, num_reco_obs_train, dtype=jnp.int32)
    all_days_indices_train = jnp.arange(model.calib_flux_days)
    def paint_body_fn_train(i, ids): return jnp.where((all_days_indices_train >= start_indices_train[i]) & (all_days_indices_train <= end_indices_train[i]), i, ids)
    segment_ids_train = fori_loop(0, num_reco_obs_train, paint_body_fn_train, segment_ids_train)
    reco_sums_train = jax.ops.segment_sum(modeled_reco_daily_train, segment_ids_train, num_segments=num_reco_obs_train + 1)[:num_reco_obs_train]
    reco_counts_train = jax.ops.segment_sum(jnp.ones_like(modeled_reco_daily_train), segment_ids_train, num_segments=num_reco_obs_train + 1)[:num_reco_obs_train]
    modeled_reco_avg_train = jnp.where(reco_counts_train > 0, reco_sums_train / reco_counts_train, 0.0)
    
    modeled_reco_daily_test = output_matrix_val[:, model.pfn.nee] + output_matrix_val[:, model.pfn.gpp]
    num_reco_obs_test = reco_obs_val.shape[0]
    reco_valid_condition_test = (reco_days_val >= 0) & (reco_days_val < len(modeled_reco_daily_test))
    valid_reco_indices_test = jnp.where(reco_valid_condition_test, reco_days_val, -1)
    start_indices_test = jnp.concatenate([jnp.array([0]), valid_reco_indices_test[:-1] + 1])
    end_indices_test = valid_reco_indices_test
    segment_ids_test = jnp.full(len(modeled_reco_daily_test), num_reco_obs_test, dtype=jnp.int32)
    all_days_indices_test = jnp.arange(len(modeled_reco_daily_test))
    def paint_body_fn_test(i, ids): return jnp.where((all_days_indices_test >= start_indices_test[i]) & (all_days_indices_test <= end_indices_test[i]), i, ids)
    segment_ids_test = fori_loop(0, num_reco_obs_test, paint_body_fn_test, segment_ids_test)
    reco_sums_test = jax.ops.segment_sum(modeled_reco_daily_test, segment_ids_test, num_segments=num_reco_obs_test + 1)[:num_reco_obs_test]
    reco_counts_test = jax.ops.segment_sum(jnp.ones_like(modeled_reco_daily_test), segment_ids_test, num_segments=num_reco_obs_test + 1)[:num_reco_obs_test]
    modeled_reco_avg_test = jnp.where(reco_counts_test > 0, reco_sums_test / reco_counts_test, 0.0)
    
    # Process LAI (instantaneous sampling)
    lai_valid_condition_train = (lai_days_cal >= 0)
    valid_lai_indices_train = jnp.where(lai_valid_condition_train, lai_days_cal, 0)
    modeled_lai_at_obs_train = calib_output[valid_lai_indices_train, model.pfn.lai]
    lai_valid_condition_test = (lai_days_val >= 0) & (lai_days_val < len(output_matrix_val))
    valid_lai_indices_test = jnp.where(lai_valid_condition_test, lai_days_val, 0)
    modeled_lai_at_obs_test = output_matrix_val[valid_lai_indices_test, model.pfn.lai]

    # --- Step 4: Calculating all performance metrics (NNSE) ---
    if not NoBM:
        if not SnapshotBM:
            bm_nnse_train = compute_nnse(bm_obs_cal, modeled_biomass_annual_train, jnp.ones_like(bm_obs_cal))
            bm_nnse_val = compute_nnse(bm_obs_val, modeled_biomass_annual_test, jnp.ones_like(bm_obs_val))
    gpp_nnse_train = compute_nnse(gpp_obs_cal, calib_output[:, model.pfn.gpp], jnp.ones_like(gpp_obs_cal))
    et_nnse_train = compute_nnse(et_obs_cal, calib_output[:, model.pfn.ET], jnp.ones_like(et_obs_cal))
    reco_nnse_train = compute_nnse(reco_obs_cal, modeled_reco_avg_train, reco_valid_condition_train)
    lai_nnse_train = compute_nnse(lai_obs_cal, modeled_lai_at_obs_train, lai_valid_condition_train)
    gpp_nnse_val = compute_nnse(gpp_obs_val, output_matrix_val[:, model.pfn.gpp], jnp.ones_like(gpp_obs_val))
    et_nnse_val = compute_nnse(et_obs_val, output_matrix_val[:, model.pfn.ET], jnp.ones_like(et_obs_val))
    reco_nnse_val = compute_nnse(reco_obs_val, modeled_reco_avg_test, reco_valid_condition_test)
    lai_nnse_val = compute_nnse(lai_obs_val, modeled_lai_at_obs_test, lai_valid_condition_test)
    
    if not NoBM:
        if not SnapshotBM:
            metrics_dict = {
                "train": {"bm_nnse": bm_nnse_train, "gpp_nnse": gpp_nnse_train, "et_nnse": et_nnse_train, "reco_nnse": reco_nnse_train, "lai_nnse": lai_nnse_train },
                "test": {"bm_nnse": bm_nnse_val, "gpp_nnse": gpp_nnse_val, "et_nnse": et_nnse_val, "reco_nnse": reco_nnse_val, "lai_nnse": lai_nnse_val }
            }
        else:
            metrics_dict = {
                "train": {"gpp_nnse": gpp_nnse_train, "et_nnse": et_nnse_train, "reco_nnse": reco_nnse_train, "lai_nnse": lai_nnse_train },
                "test": {"gpp_nnse": gpp_nnse_val, "et_nnse": et_nnse_val, "reco_nnse": reco_nnse_val, "lai_nnse": lai_nnse_val }
            }
    else:
        metrics_dict = {
            "train": {"gpp_nnse": gpp_nnse_train, "et_nnse": et_nnse_train, "reco_nnse": reco_nnse_train, "lai_nnse": lai_nnse_train },
            "test": {"gpp_nnse": gpp_nnse_val, "et_nnse": et_nnse_val, "reco_nnse": reco_nnse_val, "lai_nnse": lai_nnse_val }
        }
    
    # --- Generating final plots ---
    # Prepare time axes
    time_et_gpp_cal = pd.date_range(start=f"{config_pack['calib_start_year']}-01-01", periods=len(calib_output))
    time_et_gpp_val = pd.date_range(start=f"{config_pack['valid_start_year']}-01-01", periods=len(output_matrix_val))
    reco_time_cal = pd.to_datetime(data_pack['reco_cal_ds_time'])
    reco_time_val = pd.to_datetime(data_pack['reco_val_ds_time'])
    lai_time_cal = pd.to_datetime(data_pack['lai_cal_ds_time'])
    lai_time_val = pd.to_datetime(data_pack['lai_val_ds_time'])

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # 1. Plot ET (daily comparison)
    ax = axes[0]
    ax.plot(time_et_gpp_cal, et_obs_cal, color='skyblue', label='Observed (Train)')
    ax.plot(time_et_gpp_val, et_obs_val, color='sandybrown', label='Observed (Test)')
    ax.plot(time_et_gpp_cal, calib_output[:, model.pfn.ET], color='green', label='Model Output', linewidth=1.5)
    ax.plot(time_et_gpp_val, output_matrix_val[:, model.pfn.ET], color='green', linewidth=1.5)
    ax.set_ylabel(r'ET (mm/day)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.text(0.02, 0.8, f"Train NNSE: {metrics_dict['train']['et_nnse']:.3f}", transform=ax.transAxes, fontsize=12)
    ax.text(0.98, 0.8, f"Test NNSE: {metrics_dict['test']['et_nnse']:.3f}", transform=ax.transAxes, fontsize=12, ha='right')

    # 2. Plot GPP (daily comparison)
    ax = axes[1]
    ax.plot(time_et_gpp_cal, gpp_obs_cal, color='skyblue')
    ax.plot(time_et_gpp_val, gpp_obs_val, color='sandybrown')
    ax.plot(time_et_gpp_cal, calib_output[:, model.pfn.gpp], color='green', linewidth=1.5)
    ax.plot(time_et_gpp_val, output_matrix_val[:, model.pfn.gpp], color='green', linewidth=1.5)
    ax.set_ylabel(r'GPP (gC/m$^{2}$/day)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.text(0.02, 0.8, f"Train NNSE: {metrics_dict['train']['gpp_nnse']:.3f}", transform=ax.transAxes, fontsize=12)
    ax.text(0.98, 0.8, f"Test NNSE: {metrics_dict['test']['gpp_nnse']:.3f}", transform=ax.transAxes, fontsize=12, ha='right')

    # 3. Plot RECO (periodic average)
    ax = axes[2]
    ax.plot(reco_time_cal, reco_obs_cal, color='skyblue')
    ax.plot(reco_time_val, reco_obs_val, color='sandybrown')
    ax.plot(reco_time_cal, modeled_reco_avg_train, color='green', linestyle='-', linewidth=1.5)
    ax.plot(reco_time_val, modeled_reco_avg_test, color='green', linestyle='-', linewidth=1.5)
    ax.set_ylabel(r'RECO (gC/m$^{2}$/day)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.text(0.02, 0.8, f"Train NNSE: {metrics_dict['train']['reco_nnse']:.3f}", transform=ax.transAxes, fontsize=12)
    ax.text(0.98, 0.8, f"Test NNSE: {metrics_dict['test']['reco_nnse']:.3f}", transform=ax.transAxes, fontsize=12, ha='right')

    # 4. Plot LAI (instantaneous sampling)
    ax = axes[3]
    ax.plot(lai_time_cal, lai_obs_cal, color='skyblue')
    ax.plot(lai_time_val, lai_obs_val, color='sandybrown')
    ax.plot(lai_time_cal, modeled_lai_at_obs_train, color='green', linestyle='-', linewidth=1.5)
    ax.plot(lai_time_val, modeled_lai_at_obs_test, color='green', linestyle='-', linewidth=1.5)
    ax.set_ylabel(r'LAI (m$^{2}$/m$^{2}$)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.text(0.02, 0.8, f"Train NNSE: {metrics_dict['train']['lai_nnse']:.3f}", transform=ax.transAxes, fontsize=12)
    ax.text(0.98, 0.8, f"Test NNSE: {metrics_dict['test']['lai_nnse']:.3f}", transform=ax.transAxes, fontsize=12, ha='right')

    # Final adjustments
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=12)
    axes[3].set_xlabel('Year', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    fig_bm = None
    if not NoBM:
        if not SnapshotBM:
            fig_bm = plt.figure(figsize=(6, 6))
            ax_biomass = fig_bm.add_subplot(1, 1, 1)

            ax_biomass.plot(bm_obs, label='Forest recovery curve', linestyle='dashed')
            ax_biomass.plot(modeled_biomass_annual, label='DALEC modeled biomass')
            ax_biomass.axvline(x=100, color='red', linestyle='--')

            ax_biomass.set_xlabel('Age (years)')
            ax_biomass.set_ylabel('Biomass (Mg ha$^{-1}$)')
            ax_biomass.set_xlim(left=0, right=config_pack['total_year'] + validation_duration)
            ax_biomass.set_ylim(bottom=0)
            
            ax_biomass.text(0.4, 0.2, f'Lat: {lat:.2f}, Lon: {lon:.2f}', fontsize=12, color='blue', transform=ax_biomass.transAxes)
            ax_biomass.text(0.4, 0.1, f'Biomass NNSE: {bm_nnse_train:.4f}', fontsize=12, color='green', transform=ax_biomass.transAxes)
            
            ax_biomass.legend(frameon=False, loc='upper left')
            # ax_biomass.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

    return fig, fig_bm, metrics_dict
