import pandas as pd
import numpy as np

import sys
sys.path.insert(1, r'../../DifferLand_v1.1')
from DifferLand_CFR.util_new.transloc import latlon2land

def get_curve(CURVE_FILENAME, land_value, total_year=100):
    ages = np.arange(1, total_year+1)
    curve_df = pd.read_csv(CURVE_FILENAME)
    curve_df['land'] = [latlon2land(lat, lon) for lat, lon in zip(curve_df['lat'], curve_df['lon'])]
    sel = curve_df[curve_df['land'] == land_value]
    agb_pot = sel['AGBpot'].iloc[0]
    b = sel['b'].iloc[0]
    c = sel['c'].iloc[0]
    curve = agb_pot * (1 - np.exp(-b * ages)) ** c
    return curve