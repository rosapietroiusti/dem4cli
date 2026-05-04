


# define data directories and paths

# define v1 vs. v2 of dem4cli with the new vs. old data - change this! 


"""

To integrate
- GMST trajectories from Luke - DONE
- new population data from Dominik
- new cohort size data from Dominik 

- how to include old and new versions? all as flags? kind of messy.... or just overwrite and have only v2 ? or make a separate .py file with functions 
- For now work on separate .py file with flags in each fxn - later 
- Later, figure out how to do this in a more object oriented way !! as a self object that I assign all these things to! And the fxns automatically do what they have to - Ask Ali! 


"""

import os
from ._utils import *


class Settings(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Settings' has no attribute '{key}'")

    __setattr__ = dict.__setitem__

    def __repr__(self):
        lines = ["Settings("]
        for k, v in self.items():
            lines.append(f"  {k}: {v}")
        lines.append(")")
        return "\n".join(lines)



def init_settings(**overrides):

    # --- DEFAULTS ---
    cfg = Settings({
        "version": 2,
        "pop_resolution": 0.1,
        "GMT_mapping": "year_to_year",
        "cohort_sizes_source": "UNWPP2024",
        "countrymask": "shapefile",

        "GMT_min": 1.5,
        "GMT_max": 3.5,
        "GMT_inc": 0.1,

        "scen_thresholds": {
            "3.0": [2.9, 3.0],
            "NDC": [2.35, 2.4],
            "2.0": [1.95, 2.0],
            "1.5": [1.45, 1.5],
        },

        "scenarios": [
            "historical", "ssp126", "ssp245", "ssp370", "ssp585"
        ],
    })

    # --- APPLY USER OVERRIDES ---
    cfg.update(overrides)

    script_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(script_dir, "data")

    cfg.script_dir = script_dir
    cfg.data_dir = data_dir

    res = float_to_str(cfg.pop_resolution)

    if cfg.pop_resolution not in [0.1, 0.5]:
        raise ValueError("pop_resolution must be 0.1 or 0.5")

    # --- VERSION-DEPENDENT LOGIC ---
    if cfg.version == 1:

        cfg.pop_resolution = 0.5
        cfg.dir_population = os.path.join(data_dir, "gridded-pop")

        cfg.cohort_sizes_source = "WCDE"
        cfg.dir_cohortsizes = os.path.join(data_dir, "cohort-sizes", "WCDE")

        cfg.countrymask = "fractional_mask"
        cfg.filepath_countrymask = os.path.join(
            data_dir,
            "country-masks/isipedia-countries/preprocessed",
            f"countrymasks_fractional_{res}deg_filledcoasts.nc"
        )

        cfg.filepath_lookuptable = os.path.join(
            data_dir, "country-masks/lookup_table_dem4cli_v1.xlsx"
        )

    elif cfg.version == 2:

        cfg.dir_population = (
            f"/data/brussel/vo/000/bvo00012/data/dataset/COMPASS/v2/"
            f"population_count/{res}deg"
        )

        if cfg.cohort_sizes_source == "UNWPP2024":
            cfg.dir_cohortsizes = os.path.join(
                data_dir, "cohort-sizes/UN_WPP2024"
            )
        elif cfg.cohort_sizes_source == "WCDE":
            cfg.dir_cohortsizes = os.path.join(
                data_dir, "cohort-sizes/WCDE_v3.2.beta"
            )
        else:
            raise ValueError("Invalid cohort_sizes_source")

        if cfg.countrymask == "shapefile":
            cfg.filepath_countrymask = os.path.join(
                data_dir,
                "country-masks/natural_earth/Cultural_10m/Countries",
                "ne_10m_admin_0_countries.shp"
            )
        elif cfg.countrymask == "fractional_mask":
            cfg.filepath_countrymask = os.path.join(
                data_dir,
                "country-masks/isipedia-countries/preprocessed",
                f"countrymasks_fractional_{res}deg_filledcoasts.nc"
            )
        else:
            raise ValueError("Invalid countrymask")

        cfg.filepath_lookuptable = os.path.join(
            data_dir, "country-masks/lookup_table_dem4cli_v2.csv"
        )

    else:
        raise ValueError("version must be 1 or 2")

    # --- COMMON PATHS ---
    cfg.filepath_lifeexpectancy = os.path.join(
        data_dir,
        "life-expectancy/UN_WPP2024",
        "WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx"
    )

    cfg.filepath_world_bank_meta = os.path.join(
        data_dir, 
        'income-groups/world_bank/CLASS.xlsx')


    cfg.filepath_model_gmst = os.path.join(
        data_dir, 
        'gmst-models/gmst_models_1850_2100_allmodels.csv')

    cfg.dir_temperature_trajectories = os.path.join(data_dir, 
        'temperature-trajectories') 




    return cfg



    # TODO 
    # add cfg.filepath_shp_subnational
    # add ssp choice here


    # modify lifetime exposure, gmt mapping and gridscale to work with the config object
    # test it ! 

