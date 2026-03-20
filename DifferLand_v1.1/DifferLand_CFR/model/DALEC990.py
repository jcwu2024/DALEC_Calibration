import jax.numpy as jnp
import jax
from functools import partial
from DifferLand_CFR.model.auxi.phenology import leaf_fall_factor, lab_release_factor
from DifferLand_CFR.model.auxi.ACM import ACM
from DifferLand_CFR.model.DALEC_990_parinfo import dalec990_parmax, dalec990_parmin
from DifferLand_CFR.model.DALEC_990_parinfo import dalec990_param_parmin, dalec990_param_parmax, dalec990_pfn
from DifferLand_CFR.model.DALEC_990_parinfo import dalec990_pool_parmax, dalec990_pool_parmin
from DifferLand_CFR.optimization.forward import parameter_prediction_forward
from DifferLand_CFR.model.DALEC import DALEC
from DifferLand_CFR.util.normalization import unnormalize_parameters, nor2par
from DifferLand_CFR.optimization.loss_functions import negative_log_sigmoid, compute_nnse
# Wujc: 2025-06-20
from jax.lax import fori_loop

class DALEC990(DALEC):
    # def __init__(self, train_end_idx=-1, water_stress_type="default", ce_opt=-1, reco=False):
    # Wujc: 2026-02-21
    def __init__(self, cycle_start_year=1985, cycle_end_year=2020, total_sim_years=100, calib_start_year=2001, calib_end_year=2012, 
                 water_stress_type="default", ce_opt=-1, lcma_opt=-1, cue_opt=-1):
        super().__init__()
        self.parmin = dalec990_parmin
        self.parmax = dalec990_parmax
        self.param_parmin = dalec990_param_parmin
        self.param_parmax = dalec990_param_parmax
        self.pool_parmin = dalec990_pool_parmin
        self.pool_parmax = dalec990_pool_parmax
        self.water_stress_type = water_stress_type
        self.pfn = dalec990_pfn
        self.id = 990
        self.ce_opt = ce_opt

        # Wujc: 2025-09-05
        self.lcma_opt = lcma_opt
        # self.train_end_idx=train_end_idx
        self.pool_range = jnp.arange(self.pfn.next_labile_pool, self.pfn.next_puw_pool)
        # Wujc: 2025-06-20
        # self.pool_range = jnp.arange(self.pfn.next_som_pool, self.pfn.next_paw_pool)
        # self.reco = reco

        # Wujc: 2026-02-21
        self.cue_opt = cue_opt

        # Wujc: 2025-06-20
        # 计算驱动数据循环周期的年份列表和每年天数
        cycle_years = list(range(cycle_start_year, cycle_end_year + 1))
        cycle_year_lengths = []
        for y in cycle_years:
            if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0):
                cycle_year_lengths.append(366)
            else:
                cycle_year_lengths.append(365)
        # 根据总模拟时长，构建完整的年份天数列表
        num_cycles = total_sim_years // len(cycle_years)
        remaining_years = total_sim_years % len(cycle_years)
        self.year_lengths = cycle_year_lengths * num_cycles + cycle_year_lengths[:remaining_years]
        # 计算年度起始索引，用于生物量校准
        annual_start_indices = [0]
        for length in self.year_lengths:
            next_start = annual_start_indices[-1] + length
            annual_start_indices.append(next_start)
        # annual_start_indices 长度比年份数多 1，这里去掉最后一个，保留每年的起始索引
        self.annual_start_indices = jnp.array(annual_start_indices[:-1], dtype=jnp.int32)
        # annual_end_indices 去掉第一个，并把每年下一年起始索引减 1，得到当年最后一天索引
        self.annual_end_indices = (jnp.array(annual_start_indices[1:], dtype=jnp.int32) - 1)
        # 计算用于校准的通量数据对应的总天数 (例如 2001-2012)
        self.calib_flux_days = 0
        for y in range(calib_start_year, calib_end_year + 1):
            if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0):
                self.calib_flux_days += 366
            else:
                self.calib_flux_days += 365
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, pools, met, dalec_parameters, gpp_params):
        """A forward step of the model"""
        time = met[0]
        t_min = met[1]
        t_max = met[2]
        rad = met[3]
        ca = met[4]
        doy = met[5]
        burned_area = met[6]
        vpd = met[7]
        precipitation = met[8]
        lat = met[9]
        delta_t = met[10]
        t_mean = met[11]
        mean_precipitation = met[12]
        norm_temp = met[13]
        norm_solar = met[14]
        norm_vpd = met[15]
        norm_ca = met[16]
        
        labile_pool = pools[0]
        foliar_pool = pools[1]
        root_pool = pools[2]
        wood_pool = pools[3]
        litter_pool = pools[4]
        som_pool = pools[5]
        paw_pool = pools[6]
        puw_pool = pools[7]
                
        
        decomposition_rate = dalec_parameters[0]  # decomposition rate
        f_auto = dalec_parameters[1]  # fraction of GPP respired
        f_fol = dalec_parameters[2]  # fraction of NPP to foliar
        f_root = dalec_parameters[3]  # fraction of (1-f_fol)*NPP to root
        leaf_lifespan = dalec_parameters[4] # leaf lifespan
        tor_wood = dalec_parameters[5]  # turn over rate wood - 1% loss per year value
        tor_root = dalec_parameters[6]  # turn over rate root
        tor_litter = dalec_parameters[7]  # TOR litter
        tor_som = dalec_parameters[8]  # TOR SOM

        Q10 = dalec_parameters[9]  # Temp factor
        ce = dalec_parameters[10]  # Canopy Efficiency (time invariant)
        Bday = dalec_parameters[11]  # Bday
        f_lab = dalec_parameters[12]  # Fraction to clab
        clab_release_period = dalec_parameters[13]  # Clab release period
        Fday = dalec_parameters[14]  # Fday
        leaf_fall_period = dalec_parameters[15]  # leaf fall period
        LCMA = dalec_parameters[16]  # Leaf carbon mass per area
        uWUE = dalec_parameters[17]  # IWUE: GPP*VPD/ET: gC/kgH2o *hPa
        PAW_Qmax = dalec_parameters[18]
        field_capacity = dalec_parameters[19]
        wilting_point_frac = dalec_parameters[20]
        foliar_cf = dalec_parameters[21]  # foliar biomass cf
        ligneous_cf = dalec_parameters[22]  # ligneous biomass cf
        dom_cf = dalec_parameters[23]  # dom cf
        resilience = dalec_parameters[24]  # resilience factor
        lab_lifespan = dalec_parameters[25]  # labile pool lifespan
        moisture_factor = dalec_parameters[26]  # moisture factor
        h2o_xfer = dalec_parameters[27]
        PUW_Qmax = dalec_parameters[28]
        boese_r = dalec_parameters[29]
    
        lai= foliar_pool/LCMA

        if self.water_stress_type == "baseline":
            beta = 1
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "default":        
            wilting_point =field_capacity * wilting_point_frac
            beta = (paw_pool - wilting_point) / (field_capacity - wilting_point)
            beta = jnp.where(beta <=1, beta, 1)
            beta = jnp.where(beta >=0, beta, 0)
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "nn_paw":
            beta = jax.nn.sigmoid(parameter_prediction_forward(gpp_params, jnp.array([paw_pool/1500,]))[0])
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "nn_whole":
            beta = -9999
            raw_gpp, raw_ET = parameter_prediction_forward(gpp_params, jnp.array([norm_temp, lai/8, norm_solar,  paw_pool/1500, norm_vpd, norm_ca]))
            gpp = jnp.maximum(0.01 * raw_gpp , raw_gpp)
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)
        elif self.water_stress_type == "nn_whole_no_lai":
            beta = -9999
            raw_gpp, raw_ET = parameter_prediction_forward(gpp_params, jnp.array([norm_temp, norm_solar,  paw_pool/1500, norm_vpd, norm_ca]))
            gpp = jnp.maximum(0.01 * raw_gpp , raw_gpp)
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)
        elif self.water_stress_type == "gpp_acm_et_nn":
            beta = -9999
            beta_params, et_params = gpp_params
            beta = jax.nn.sigmoid(parameter_prediction_forward(beta_params, jnp.array([paw_pool/1500,]))[0])
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            raw_ET = parameter_prediction_forward(et_params, jnp.array([norm_temp, lai/8, norm_solar,  paw_pool/1500, norm_vpd, norm_ca, gpp/8]))[0]
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)

        temperate = (jnp.exp(Q10 * (0.5 * (t_max + t_min) - t_mean)) 
                    * ((precipitation / mean_precipitation - 1) * moisture_factor + 1))

        respiration_auto = f_auto * gpp
        leaf_production = (gpp - respiration_auto) * f_fol
        labile_production = (gpp - respiration_auto - leaf_production) * f_lab
        root_production = (gpp - respiration_auto - leaf_production - labile_production) * f_root
        wood_production = gpp - respiration_auto - leaf_production - labile_production - root_production

        lff = leaf_fall_factor(time, leaf_lifespan, leaf_fall_period, Fday)
        lrf = lab_release_factor(time, lab_lifespan, clab_release_period, Bday)

        labile_release = labile_pool * (1 - (1 - lrf) ** delta_t) / delta_t
        leaf_litter = foliar_pool * (1 - (1 - lff) ** delta_t) / delta_t
        wood_litter = wood_pool * (1 - (1 - tor_wood) ** delta_t) / delta_t
        root_litter = root_pool * (1 - (1 - tor_root) ** delta_t) / delta_t
        
        respiration_hetero_litter = litter_pool * (1 - (1 - temperate * tor_litter) ** delta_t) / delta_t
        respiration_hetero_som = som_pool * (1 - (1 - temperate * tor_som) ** delta_t) / delta_t
        litter_to_som = litter_pool * (1 - (1 - temperate * decomposition_rate) ** delta_t) / delta_t
        
        q_paw = paw_pool ** 2 / PAW_Qmax / delta_t * (1-h2o_xfer)
        paw2puw = q_paw * h2o_xfer / (1-h2o_xfer)
        paw_focal_sel = paw_pool <= PAW_Qmax / 2
        q_paw = jnp.where(paw_focal_sel, q_paw, (paw_pool - PAW_Qmax / 4) / delta_t * (1 - h2o_xfer))
        paw2puw = jnp.where(paw_focal_sel, paw2puw, (paw_pool - PAW_Qmax / 4) / delta_t * h2o_xfer)
        
        q_puw = puw_pool ** 2 / PUW_Qmax / delta_t
        puw_focal_sel = puw_pool <= PUW_Qmax / 2
        q_puw = jnp.where(puw_focal_sel, q_puw, (puw_pool - PUW_Qmax / 4) / delta_t)

        # apply adjustments such that each pool never exceeds its minimum or maximum threshold.
        next_labile_pool = labile_pool + (labile_production - labile_release) * delta_t
        Clab_min_sel = next_labile_pool >= self.parmin.Clab
        next_labile_pool = jnp.where(Clab_min_sel, next_labile_pool, self.parmin.Clab)
        labile_release = jnp.where(Clab_min_sel, labile_release, labile_production - (next_labile_pool - labile_pool) / delta_t)
        Clab_max_sel = next_labile_pool <= self.parmax.Clab
        next_labile_pool = jnp.where(Clab_max_sel, next_labile_pool, self.parmax.Clab)
        labile_release = jnp.where(Clab_max_sel, labile_release, labile_production - (next_labile_pool - labile_pool) / delta_t)
        
        next_foliar_pool = foliar_pool + (leaf_production - leaf_litter + labile_release) * delta_t
        Cfol_min_sel = next_foliar_pool >= self.parmin.Cfol
        next_foliar_pool = jnp.where(Cfol_min_sel, next_foliar_pool, self.parmin.Cfol)
        leaf_litter = jnp.where(Cfol_min_sel, leaf_litter, (1 - Cfol_min_sel) * self.parmin.Cfol)
        Cfol_max_sel = next_foliar_pool <= self.parmax.Cfol
        next_foliar_pool = jnp.where(Cfol_max_sel, next_foliar_pool, self.parmax.Cfol)
        leaf_litter = jnp.where(Cfol_max_sel, leaf_litter, leaf_production + labile_release - (next_foliar_pool - foliar_pool) / delta_t)
        
        next_root_pool = root_pool + (root_production - root_litter) * delta_t
        Croot_min_sel = next_root_pool >= self.parmin.Croot
        next_root_pool = jnp.where(Croot_min_sel, next_root_pool, self.parmin.Croot)
        root_litter = jnp.where(Croot_min_sel, root_litter, root_production - (next_root_pool - root_pool) / delta_t)
        Croot_max_sel = next_root_pool <= self.parmax.Croot
        next_root_pool = jnp.where(Croot_max_sel, next_root_pool, self.parmax.Croot)
        root_litter = jnp.where(Croot_max_sel, root_litter, root_production - (next_root_pool - root_pool) / delta_t)
        
        next_wood_pool = wood_pool + (wood_production - wood_litter) * delta_t
        Cwood_min_sel = next_wood_pool >= self.parmin.Cwood
        next_wood_pool = jnp.where(Cwood_min_sel, next_wood_pool, self.parmin.Cwood)
        wood_litter = jnp.where(Cwood_min_sel, wood_litter, wood_production - (next_wood_pool - wood_pool) / delta_t)
        Cwood_max_sel = next_wood_pool <= self.parmax.Cwood
        next_wood_pool = jnp.where(Cwood_max_sel, next_wood_pool, self.parmax.Cwood)
        wood_litter = jnp.where(Cwood_max_sel, wood_litter, wood_production - (next_wood_pool - wood_pool) / delta_t)

        
        next_litter_pool = litter_pool + (leaf_litter + root_litter - respiration_hetero_litter - litter_to_som) * delta_t
        Clitter_min_sel = next_litter_pool >= self.parmin.Clitter
        next_litter_pool = jnp.where(Clitter_min_sel, next_litter_pool, self.parmin.Clitter)
        litter_to_som = jnp.where(Clitter_min_sel, litter_to_som, leaf_litter + root_litter - respiration_hetero_litter - (next_litter_pool - litter_pool) / delta_t)
        litter_to_som_sel = litter_to_som >= 0
        litter_to_som = jnp.where(litter_to_som_sel, litter_to_som, 0)
        respiration_hetero_litter = jnp.where(litter_to_som_sel, respiration_hetero_litter, leaf_litter + root_litter - (next_litter_pool - litter_pool) / delta_t)
        Clitter_max_sel = next_litter_pool <= self.parmax.Clitter
        next_litter_pool = jnp.where(Clitter_max_sel, next_litter_pool, self.parmax.Clitter)
        litter_to_som = jnp.where(Clitter_max_sel, litter_to_som, leaf_litter + root_litter - respiration_hetero_litter - (next_litter_pool - litter_pool) / delta_t)

        next_som_pool = som_pool + (litter_to_som - respiration_hetero_som + wood_litter) * delta_t
        Csom_min_sel = next_som_pool >= self.parmin.Csom
        next_som_pool = jnp.where(Csom_min_sel, next_som_pool, self.parmin.Csom)
        respiration_hetero_som = jnp.where(Csom_min_sel, respiration_hetero_som, litter_to_som + wood_litter - (next_som_pool - som_pool) / delta_t)
        Csom_max_sel = next_som_pool <= self.parmax.Csom
        next_som_pool = jnp.where(Csom_max_sel, next_som_pool, self.parmax.Csom)
        respiration_hetero_som = jnp.where(Csom_max_sel, respiration_hetero_som, litter_to_som + wood_litter - (next_som_pool - som_pool) / delta_t)
        
        next_paw_pool = paw_pool + (-q_paw - paw2puw + precipitation - ET) * delta_t
        water_min_paw_sel = next_paw_pool >= self.parmin.initial_PAW
        next_paw_pool = jnp.where(water_min_paw_sel, next_paw_pool, self.parmin.initial_PAW)
        q_paw = jnp.where(water_min_paw_sel, q_paw, precipitation  - ET  - (next_paw_pool - paw_pool) / delta_t * (1-h2o_xfer))
        paw2puw = jnp.where(water_min_paw_sel, paw2puw, precipitation  - ET  - (next_paw_pool - paw_pool) / delta_t * h2o_xfer)
        q_paw_sel = q_paw >= 0.0
        violation = jnp.maximum(-q_paw * 0.01, 0)
        ET = jnp.where(q_paw_sel, ET, precipitation - (next_paw_pool - paw_pool) / delta_t)
        q_paw = jnp.where(q_paw_sel, q_paw, 0)
        paw2puw = jnp.where(q_paw_sel, paw2puw, 0)
        water_max_paw_sel = next_paw_pool <= self.parmax.initial_PAW
        next_paw_pool = jnp.where(water_max_paw_sel, next_paw_pool, self.parmax.initial_PAW)
        q_paw = jnp.where(water_max_paw_sel, q_paw, precipitation  - ET  - (next_paw_pool - paw_pool) / delta_t * (1-h2o_xfer))
        paw2puw = jnp.where(water_max_paw_sel, paw2puw, precipitation  - ET  - (next_paw_pool - paw_pool) / delta_t * h2o_xfer)

        next_puw_pool = puw_pool + (paw2puw - q_puw) * delta_t
        next_puw_min_sel = next_puw_pool >= self.parmin.initial_PUW
        next_puw_pool = jnp.where(next_puw_min_sel, next_puw_pool, self.parmin.initial_PUW)
        q_puw = jnp.where(next_puw_min_sel, q_puw,  paw2puw - (next_puw_pool - puw_pool) / delta_t)
        next_puw_max_sel = next_puw_pool <= self.parmax.initial_PUW
        next_puw_pool = jnp.where(next_puw_max_sel, next_puw_pool, self.parmax.initial_PUW)
        q_puw = jnp.where(next_puw_min_sel, q_puw,  paw2puw - (next_puw_pool - puw_pool) / delta_t)

        nee = -gpp + respiration_auto + respiration_hetero_litter + respiration_hetero_som

        new_pools = jnp.array([next_labile_pool, next_foliar_pool,
                            next_root_pool, next_wood_pool, next_litter_pool,
                            next_som_pool, next_paw_pool, next_puw_pool])
        
        new_fluxes = jnp.array([[lai, gpp, ET, temperate, respiration_auto, leaf_production, labile_production, root_production,
    wood_production, lff, lrf, labile_release, leaf_litter, wood_litter, root_litter, respiration_hetero_litter,
    respiration_hetero_som, litter_to_som, q_paw, q_puw, paw2puw, nee, next_labile_pool, next_foliar_pool,
    next_root_pool, next_wood_pool, next_litter_pool, next_som_pool, next_paw_pool, next_puw_pool, beta, violation]])
    
        return new_pools, new_fluxes
        
    @partial(jax.jit, static_argnums=(0,))
    def unnormalize(self, normalized_parameters):
        """Convert model parameters from the real space to the physcial range"""
        return unnormalize_parameters(normalized_parameters, param_parmin=self.param_parmin, param_parmax=self.param_parmax)
    
    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_pools(self, normalized_pools):
        """Convert pool values from the real space to the physical value range"""
        return unnormalize_parameters(normalized_pools, param_parmin=self.pool_parmin, param_parmax=self.pool_parmax)
    
    @partial(jax.jit, static_argnums=(0,))
    def pre_edc(self, dalec_params, mean_temp, k):
        """EDC constraints applied before model integration"""
        a_auto = dalec_params[1]
        a_fol = (1 - a_auto) * dalec_params[2]
        a_lab = (1 - a_auto - a_fol) * dalec_params[12]
        a_root = (1 - a_auto - a_fol - a_lab) * dalec_params[3]

        edc1 = negative_log_sigmoid(dalec_params[7], dalec_params[8], 100000 * k)
        edc2 = negative_log_sigmoid(dalec_params[0], dalec_params[8], 10000 * k)
        edc3 = negative_log_sigmoid(1 / (dalec_params[4] * 365.25), dalec_params[5], 200000 * k) 
        edc4 = negative_log_sigmoid(dalec_params[6], dalec_params[8] * jnp.exp(dalec_params[9] * mean_temp), 10 * k)
        edc5 = negative_log_sigmoid(5 * a_root, (a_fol + a_lab), 100 * k) + negative_log_sigmoid(5 * (a_fol + a_lab), a_root, 100 * k)

        return edc1 + edc2 + edc3 + edc4 + edc5
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, param_initial, pool_initial, gpp_params, met):
        """Forward hybrid model integration"""
        dp = self.unnormalize(param_initial)
        ip = self.unnormalize_pools(pool_initial)
        final_state, all_fluxes = jax.lax.scan(jax.jit(partial(self.step,
                                                            dalec_parameters=dp,
                                                            gpp_params=gpp_params)), ip, met)
        # return all_fluxes.squeeze()
        # Wujc: 2025-06-21
        return final_state, all_fluxes.squeeze()
    
    # @partial(jax.jit, static_argnums=(0,))
    # def compute_loss(self, param_initial, pool_initial, gpp_params, met_matrix, target_matrix, k):
    #     "Compute value of the loss function"
    #     output_matrix = self.forward(param_initial, pool_initial, gpp_params, met_matrix)
    #     gpp_loss = compute_nnse(target_matrix[:self.train_end_idx, 0], output_matrix[:self.train_end_idx, self.pfn.gpp], target_matrix[:self.train_end_idx, 1])
    #     # use ecosystem respiration (RECO) for 
    #     if self.reco:
    #         nee_reco_loss = compute_nnse(target_matrix[:self.train_end_idx, 8], output_matrix[:self.train_end_idx, self.pfn.nee] + output_matrix[:self.train_end_idx, self.pfn.gpp], target_matrix[:self.train_end_idx, 9])
    #     else:
    #         nee_reco_loss = compute_nnse(target_matrix[:self.train_end_idx, 2], output_matrix[:self.train_end_idx, self.pfn.nee], target_matrix[:self.train_end_idx, 3])
    #     et_loss = compute_nnse(target_matrix[:self.train_end_idx, 4], output_matrix[:self.train_end_idx, self.pfn.ET], target_matrix[:self.train_end_idx, 5])
    #     lai_loss = compute_nnse(target_matrix[:self.train_end_idx, 6], output_matrix[:self.train_end_idx, self.pfn.lai], target_matrix[:self.train_end_idx, 7])
    #     ce_loss = 0
    #     ce = nor2par(param_initial[10], 5, 50)
    #     ce_loss = 0.1 * (ce - self.ce_opt) ** 2
    #     ce_loss = jnp.where(self.ce_opt >=5, ce_loss, 0.0)
    #     edc_loss = self.pre_edc(self.unnormalize(param_initial), jnp.mean(met_matrix[:self.train_end_idx, 13]), k)
    #     end_of_year_idx = met_matrix[:, 17]
    #     total_precip = jnp.sum(met_matrix[:, 8])
    #     edc_loss += self.post_edc(output_matrix, self.unnormalize_pools(pool_initial), total_precip, k, end_of_year_idx)
    #     return -gpp_loss -nee_reco_loss -et_loss -lai_loss + ce_loss + edc_loss
    
    
    # Wujc: 2025-07-09
    # @partial(jax.jit, static_argnums=(0,))
    def compute_loss(self, param_initial, pool_initial, gpp_params, met_matrix,
                    biomass_obs, gpp_obs, reco_obs, lai_obs, et_obs, reco_calib_indices, lai_calib_indices,
                    k, NoBM=False, SnapshotBM=False, NoCE=True, NoLCMA=True, NoCUE=True, NoLES=False):
        _, output_matrix = self.forward(param_initial, pool_initial, gpp_params, met_matrix)
        """Labile, Foliar, Wood, and Root pools are summed to get the modeled biomass"""
        if NoBM:
            biomass_loss = 0.0
        elif SnapshotBM:
            modeled_biomass_daily = ((output_matrix[:, self.pfn.next_labile_pool] + output_matrix[:, self.pfn.next_foliar_pool] + 
                        output_matrix[:, self.pfn.next_root_pool] + output_matrix[:, self.pfn.next_wood_pool]) * 2) / 100
            modeled_snapshot = modeled_biomass_daily[self.annual_end_indices[-1]]
            obs_snapshot = jnp.atleast_1d(biomass_obs)[-1]
            biomass_loss = 1.0 - ((modeled_snapshot - obs_snapshot) / obs_snapshot) ** 2
        else:
            modeled_biomass_daily = ((output_matrix[:, self.pfn.next_labile_pool] + output_matrix[:, self.pfn.next_foliar_pool] + 
                        output_matrix[:, self.pfn.next_root_pool] + output_matrix[:, self.pfn.next_wood_pool]) * 2) / 100
            modeled_biomass_annual = modeled_biomass_daily[self.annual_end_indices]  # Last value of each year
            biomass_loss = compute_nnse(biomass_obs, modeled_biomass_annual, jnp.ones_like(biomass_obs))

        # Only the last part of the output matrix is used for calibration
        calib_flux_output = output_matrix[-self.calib_flux_days:, :]
        gpp_loss = compute_nnse(gpp_obs, calib_flux_output[:, self.pfn.gpp], jnp.ones_like(gpp_obs))
        
        et_loss = compute_nnse(et_obs, calib_flux_output[:, self.pfn.ET], jnp.ones_like(et_obs))

        modeled_reco_daily = calib_flux_output[:, self.pfn.nee] + calib_flux_output[:, self.pfn.gpp]
        reco_valid_condition = (reco_calib_indices >= 0) & (reco_calib_indices < self.calib_flux_days)
        valid_reco_indices = jnp.where(reco_valid_condition, reco_calib_indices, -1)
        start_indices = jnp.concatenate([jnp.array([0]), valid_reco_indices[:-1] + 1])
        end_indices = valid_reco_indices
        num_reco_obs = reco_obs.shape[0]
        all_days_indices = jnp.arange(self.calib_flux_days)
        segment_ids = jnp.full(self.calib_flux_days, num_reco_obs, dtype=jnp.int32)
        def paint_body_fn(i, current_segment_ids):
            mask = (all_days_indices >= start_indices[i]) & (all_days_indices <= end_indices[i])
            new_segment_ids = jnp.where(mask, i, current_segment_ids)
            return new_segment_ids
        segment_ids = fori_loop(0, num_reco_obs, paint_body_fn, segment_ids)
        reco_sums = jax.ops.segment_sum(modeled_reco_daily, segment_ids, num_segments=num_reco_obs + 1)
        ones = jnp.ones_like(modeled_reco_daily)
        reco_counts = jax.ops.segment_sum(ones, segment_ids, num_segments=num_reco_obs + 1)
        reco_sums_valid = reco_sums[:num_reco_obs]
        reco_counts_valid = reco_counts[:num_reco_obs]
        modeled_reco_avg = jnp.where(reco_counts_valid > 0, reco_sums_valid / reco_counts_valid, 0.0)
        reco_loss = compute_nnse(reco_obs, modeled_reco_avg, jnp.where(reco_valid_condition, 1.0, 0.0))

        lai_valid_condition = (lai_calib_indices >= 0) & (lai_calib_indices < self.calib_flux_days)
        valid_lai_indices = jnp.where(lai_valid_condition, lai_calib_indices, -1)
        modeled_lai = calib_flux_output[valid_lai_indices, self.pfn.lai]
        lai_loss = compute_nnse(lai_obs, modeled_lai, jnp.where(lai_valid_condition, 1.0, 0.0))

        dalec_params = self.unnormalize(param_initial)

        if NoCE:
            ce_loss = 0.0
        else:
            ce = dalec_params[10]
            safe_ce_opt = jnp.maximum(self.ce_opt, 5.0)
            ce_loss = 0.1 * (jnp.log(ce) - jnp.log(safe_ce_opt)) ** 2
            ce_loss = jnp.where(self.ce_opt >= 5, ce_loss, 0.0)

        if NoLCMA:
            lcma_loss = 0.0
        else:
            lcma = dalec_params[16]

            safe_lcma_opt = jnp.maximum(self.lcma_opt, 5.0)
            lcma_loss = 1.0 * (jnp.log(lcma) - jnp.log(safe_lcma_opt)) ** 2
            lcma_loss = jnp.where(self.lcma_opt >= 5, lcma_loss, 0.0)

        # Wujc: 2026-02-21
        if NoCUE:
            cue_loss = 0.0
        else:
            cue = 1 - dalec_params[1]
            safe_cue_opt = jnp.maximum(self.cue_opt, 0.2)
            cue_loss = 1.0 * (jnp.log(cue) - jnp.log(safe_cue_opt)) ** 2
            cue_loss = jnp.where(self.cue_opt >= 0.2, cue_loss, 0.0)
        
        if NoLES:
            les_loss = 0.0
        else:
            LCMA = dalec_params[16]
            ln_lcma = jnp.log(LCMA)
            target_effective_ll = jnp.exp(ln_lcma * 1.0514 - 1.7608) / 12.0
            target_annual_turnover = 1.0 / target_effective_ll

            daily_lff = output_matrix[:, self.pfn.lff]
            total_lff_sum = jnp.sum(daily_lff)
            num_years = len(self.year_lengths)
            actual_annual_turnover = total_lff_sum / num_years

            epsilon = 1e-9
            log_actual_turnover = jnp.log(actual_annual_turnover + epsilon)
            log_target_turnover = jnp.log(target_annual_turnover + epsilon)
            
            les_loss = 1.0 * (log_actual_turnover - log_target_turnover)**2

        edc_loss = self.pre_edc(self.unnormalize(param_initial), jnp.mean(met_matrix[:, 13]), k)
        end_of_year_idx = met_matrix[:, 17]
        total_precip = jnp.sum(met_matrix[:, 8])

        # 如果 不同化生物量 或者是 快照生物量模式，则强制要求生物量稳态
        steady_biomass = NoBM or SnapshotBM
        edc_loss += self.post_edc(output_matrix, self.unnormalize_pools(pool_initial), total_precip, k, end_of_year_idx, steady_biomass)
        
        return -biomass_loss - gpp_loss - et_loss - reco_loss - lai_loss + ce_loss + lcma_loss + cue_loss + les_loss + edc_loss

    # @partial(jax.jit, static_argnums=(0,))
    # def post_edc(self, output_matrix, Pstart, total_precip, k, end_of_year_idx):
    #     """EDC constraints applied after model run"""
    #     etol = 0.1
    #     MPOOLS_Jan = (Pstart[:-1] + jnp.sum(output_matrix[:, self.pool_range].T * end_of_year_idx, axis=1)) / (jnp.sum(end_of_year_idx) + 1)
    #     POOLS = output_matrix[:, self.pool_range]

    #     FLUXES = output_matrix[:, :self.pfn.next_labile_pool]
    #     FTOTAL = jnp.sum(FLUXES, axis=0)
    #     Fin_1 = FTOTAL[self.pfn.labile_production]
    #     Fout_1 = FTOTAL[self.pfn.labile_release] 
    #     Fin_2 = FTOTAL[self.pfn.leaf_production] + FTOTAL[self.pfn.labile_release]
    #     Fout_2 = FTOTAL[self.pfn.leaf_litter] 
    #     Fin_3 = FTOTAL[self.pfn.root_production]
    #     Fout_3 = FTOTAL[self.pfn.root_litter] 
    #     Fin_4 = FTOTAL[self.pfn.wood_production]
    #     Fout_4 = FTOTAL[self.pfn.wood_litter] 
    #     Fin_5 = FTOTAL[self.pfn.leaf_litter] + FTOTAL[self.pfn.root_litter] 
    #     Fout_5 = FTOTAL[self.pfn.respiration_hetero_litter] + FTOTAL[self.pfn.litter_to_som] 
    #     Fin_6 = FTOTAL[self.pfn.wood_litter] + FTOTAL[self.pfn.litter_to_som] 
    #     Fout_6 = FTOTAL[self.pfn.respiration_hetero_som] 
    #     Fin_7 = total_precip
    #     Fout_7 = FTOTAL[self.pfn.ET] + FTOTAL[self.pfn.q_paw] + FTOTAL[self.pfn.paw2puw]
    #     # Note, we excluded PUW here because we are not directly assimilating EWT data, this can be changed
    #     # in the future when EWT is actually included as a constraint.
      
    #     Fin = jnp.array([Fin_1, Fin_2, Fin_3, Fin_4, Fin_5, Fin_6, Fin_7])
    #     Fout = jnp.array([Fout_1, Fout_2, Fout_3, Fout_4, Fout_5, Fout_6, Fout_7])

    #     Pend = POOLS[-1, :]
    #     Rm = Fin / Fout
    #     Rs = Rm * MPOOLS_Jan / Pstart[:-1]
    #     EQF = 2
    #     edc_loss = 0
    #     for i in range(6):
    #         edc_loss += negative_log_sigmoid(jnp.log(EQF), jnp.abs(jnp.log(Rs[i])), 3 * k)
    #         edc_loss += negative_log_sigmoid(etol, jnp.abs(Rs[i] - Rm[i]), 3 * k)
            
    #     # add total violation due to pool reaching the minimum threshold
    #     edc_loss += jnp.sum(output_matrix[:, self.pfn.violation])
    #     return edc_loss

    # Wujc: 2025-07-10
    # 为防止土壤碳快速消耗或爆炸增长，限制土壤碳输出输入比在0.667~1.5之间。
    # 水只能从植物可用水库转移到植物不可用水库，不能反过来。权重设置为0.001，不能过大。
    # @partial(jax.jit, static_argnums=(0,))
    # def post_edc(self, output_matrix, Pstart, total_precip, k, end_of_year_idx):
    #     """EDC constraints applied after model run"""       
    #     calib_days = self.calib_flux_days
    #     FLUXES = output_matrix[:, :self.pfn.next_labile_pool]
    #     FLUXES_CAL = FLUXES[-calib_days:, :]
    #     FTOTAL_CAL = jnp.sum(FLUXES_CAL, axis=0)

    #     # Fin_1_CAL = FTOTAL_CAL[self.pfn.labile_production]
    #     # Fout_1_CAL = FTOTAL_CAL[self.pfn.labile_release]
    #     # Fin_2_CAL = FTOTAL_CAL[self.pfn.leaf_production] + FTOTAL_CAL[self.pfn.labile_release]
    #     # Fout_2_CAL = FTOTAL_CAL[self.pfn.leaf_litter]
    #     # Fin_3_CAL = FTOTAL_CAL[self.pfn.root_production]
    #     # Fout_3_CAL = FTOTAL_CAL[self.pfn.root_litter]
    #     # Fin_4_CAL = FTOTAL_CAL[self.pfn.wood_production]
    #     # Fout_4_CAL = FTOTAL_CAL[self.pfn.wood_litter]
    #     # Fin_5_CAL = FTOTAL_CAL[self.pfn.leaf_litter] + FTOTAL_CAL[self.pfn.root_litter] 
    #     # Fout_5_CAL = FTOTAL_CAL[self.pfn.respiration_hetero_litter] + FTOTAL_CAL[self.pfn.litter_to_som]
    #     Fin_6_CAL = FTOTAL_CAL[self.pfn.wood_litter] + FTOTAL_CAL[self.pfn.litter_to_som] 
    #     Fout_6_CAL = FTOTAL_CAL[self.pfn.respiration_hetero_som]

    #     # Fin_cal = jnp.array([Fin_1_CAL, Fin_2_CAL, Fin_3_CAL, Fin_4_CAL, Fin_5_CAL, Fin_6_CAL])
    #     # Fout_cal = jnp.array([Fout_1_CAL, Fout_2_CAL, Fout_3_CAL, Fout_4_CAL, Fout_5_CAL, Fout_6_CAL])
        
    #     # Rm_cal = Fin_cal / Fout_cal
    #     Rm_cal = Fin_6_CAL / Fout_6_CAL

    #     EQF = 2
    #     edc_loss = 0
        
    #     # for i in range(6):
    #     #     edc_loss += negative_log_sigmoid(jnp.log(EQF), jnp.abs(jnp.log(Rm_cal[i])), 3 * k)
    #     edc_loss += negative_log_sigmoid(jnp.log(EQF), jnp.abs(jnp.log(Rm_cal)), 3 * k)

    #     # add total violation due to pool reaching the minimum threshold
    #     edc_loss += jnp.sum(output_matrix[:, self.pfn.violation]) * 0.001

    #     return edc_loss
    
    # Wujc: 2026-02-23
    @partial(jax.jit, static_argnums=(0,))
    def post_edc(self, output_matrix, Pstart, total_precip, k, end_of_year_idx, steady_biomass):
        """EDC constraints applied after model run"""       
        calib_days = self.calib_flux_days
        FLUXES = output_matrix[:, :self.pfn.next_labile_pool]
        FLUXES_CAL = FLUXES[-calib_days:, :]
        FTOTAL_CAL = jnp.sum(FLUXES_CAL, axis=0)

        # 还原所有 6 个碳库的输入输出计算
        Fin_1_CAL = FTOTAL_CAL[self.pfn.labile_production]
        Fout_1_CAL = FTOTAL_CAL[self.pfn.labile_release]
        Fin_2_CAL = FTOTAL_CAL[self.pfn.leaf_production] + FTOTAL_CAL[self.pfn.labile_release]
        Fout_2_CAL = FTOTAL_CAL[self.pfn.leaf_litter]
        Fin_3_CAL = FTOTAL_CAL[self.pfn.root_production]
        Fout_3_CAL = FTOTAL_CAL[self.pfn.root_litter]
        Fin_4_CAL = FTOTAL_CAL[self.pfn.wood_production]
        Fout_4_CAL = FTOTAL_CAL[self.pfn.wood_litter]
        Fin_5_CAL = FTOTAL_CAL[self.pfn.leaf_litter] + FTOTAL_CAL[self.pfn.root_litter] 
        Fout_5_CAL = FTOTAL_CAL[self.pfn.respiration_hetero_litter] + FTOTAL_CAL[self.pfn.litter_to_som]
        Fin_6_CAL = FTOTAL_CAL[self.pfn.wood_litter] + FTOTAL_CAL[self.pfn.litter_to_som] 
        Fout_6_CAL = FTOTAL_CAL[self.pfn.respiration_hetero_som]

        Fin_cal = jnp.array([Fin_1_CAL, Fin_2_CAL, Fin_3_CAL, Fin_4_CAL, Fin_5_CAL, Fin_6_CAL])
        Fout_cal = jnp.array([Fout_1_CAL, Fout_2_CAL, Fout_3_CAL, Fout_4_CAL, Fout_5_CAL, Fout_6_CAL])
        
        # 计算所有 6 个碳库的 输入/输出 比例
        Rm_cal = Fin_cal / Fout_cal

        # 输出输入比在0.5~2.0之间
        EQF = 2.0 
        edc_loss = 0.0
        
        # ==========================================================
        # 1. 无论何种模式，都强制限制土壤有机碳库（索引 5）稳定
        # ==========================================================
        edc_loss += negative_log_sigmoid(jnp.log(EQF), jnp.abs(jnp.log(Rm_cal[5])), 3 * k)

        # ==========================================================
        # 2. 动态计算植物与凋落物碳库（索引 0 到 4）的稳态 Loss
        # ==========================================================
        veg_loss = 0.0
        for i in range(5):
            veg_loss += negative_log_sigmoid(jnp.log(EQF), jnp.abs(jnp.log(Rm_cal[i])), 3 * k)
            
        # 使用 jnp.where 巧妙控制：如果 steady_biomass 为 True，则应用该惩罚，否则加 0
        edc_loss += jnp.where(steady_biomass, veg_loss, 0.0)

        # 添加防止碳库越过最小物理边界的惩罚
        edc_loss += jnp.sum(output_matrix[:, self.pfn.violation]) * 0.001

        return edc_loss