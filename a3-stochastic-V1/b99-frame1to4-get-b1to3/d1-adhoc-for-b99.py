import datetime
import os
import sys
from paths import a3_stochastic_relative_path, b98_sampling_path, a1_blendshape_path

sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo" / "example_train"))
sys.path.insert(0, str(a1_blendshape_path / "database"))
sys.path.insert(0, str(b98_sampling_path ))
sys.path.insert(0, str(b98_sampling_path / "300joints_suite_good_old"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/ "joint_position_ooc_l"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/ "joint_position_ooc_r"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/ "joint_position_frontalis"))
sys.path.insert(0, r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub")
sys.path.insert(0, str(a3_stochastic_relative_path / "b95-debug-use"))
from generate_debug_output import generate_output
import maya.standalone
import maya.cmds as cmds
import inspect

neutral_to_other_pose_flag = 1
previous_pose_folder = r"//bbysvm-nfs1.its.sfu.ca/GrUViLab/Data/richardPeng/SFU_PhD_cloud/2-Projects/1-physically based facial animation/3-Experiments/a3-stochastic-V1-backup-and-results/results/job_251205-230610/251205-230619"

# didn't know why this path can't be added in.

# sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d2_helper_functions")




import j77_V3_estimate_r_limit_newest  # Import the whole module
# target_rotation_deg = j77_V3_estimate_r_limit_newest.calculate_rotation_parameters()

# Import the helper script (assuming it's in a package/module)
# from j77_V3_estimate_r_limit_newest import calculate_rotation_parameters    



run_c96_finalized_datav21generating = 0
run_a2_fromFrameGetDeltaB = 1
run_both = 1
run_rotation = 0
run_translation = 0

# keyframe_num = 20 # 50 # 100 # 10 # 4  # 10

use_incremental_NN_flag = 0  # 1  # 0

import argparse
import pickle
import numpy as np

import cProfile
import pstats


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
from matplotlib.colors import Normalize  # Import Normalize from matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation 
# from IPython import display
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection

import importlib

# import joint_position
# import joint_position_first_wave
import joint_position
import joint_position_first_wave
import joint_position_third_wave
importlib.reload(joint_position)
importlib.reload(joint_position_first_wave)
importlib.reload(joint_position_third_wave)

import joint_position_ooc_l_base_0
import joint_position_ooc_l_base_pose11_0
import joint_position_ooc_l_base_pose11_AU1_0
import joint_position_ooc_l_base_pose11_AU1_AU2_0

import joint_position_ooc_r_base_0
import joint_position_ooc_r_base_pose11_0
import joint_position_ooc_r_base_pose11_AU1_0
import joint_position_ooc_r_base_pose11_AU1_AU2_0

import joint_position_frontalis_base_0
import joint_position_frontalis_base_AU1_0
import joint_position_frontalis_base_AU1_AU2_0


parser = argparse.ArgumentParser()
# Maximum iteration number
parser.add_argument("--max_iteration", type=int, default=12000)
# Key Parameters for Richard's
parser.add_argument("--step_counter_internal_max", type=int, default=100, help="id of environment")
# this many steps
# parser.add_argument("--distance_min", type=float, default=0.05, help="DSAC_V2 or DSAC_V1")
#set algorithm default to DSAC_V2, but it can be replaced by DSAC_V1 if you want to use the old version of DSAC.
parser.add_argument("--num_eval_episode", type=int, default=5)
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--keyframe_num", type=int, default=10)
parser.add_argument("--keyframe_translation_num", type=int, default=10)
parser.add_argument("--distance_threshold_percentage", type=float, default=0.999, help="distance_threshold_percentage")
parser.add_argument("--distance_jaw_threshold_percentage", type=float, default=0.999, help="distance_jaw_threshold_percentage")
parser.add_argument("--distance_ooc_l_threshold_percentage", type=float, default=0.999, help="distance_ooc_l_threshold_percentage")
parser.add_argument("--distance_ooc_r_threshold_percentage", type=float, default=0.999, help="distance_ooc_r_threshold_percentage")
parser.add_argument("--distance_frontalis_threshold_percentage", type=float, default=0.999, help="distance_frontalis_threshold_percentage")
parser.add_argument("--unit_denominator", type=float, default=100000.0, help="unit_denominator")
parser.add_argument("--sample_batch_size", type=int, default=20, help="unit_denominator")

args = vars(parser.parse_args())
import a2_fromFrameGetDeltaB as a2_fromFrameGetDeltaB


keyframe_num = args["keyframe_num"]
keyframe_translation_num = args["keyframe_translation_num"]


if 1 == run_c96_finalized_datav21generating:
    import c96_finalized_datav21generating as c96_finalized_datav21generating
    c96_finalized_datav21generating.c96_finalized_datav21generating(keyframe_num)


class JawRotationConfig:
    def __init__(self, target_rotation_deg=None, estimated_steps=None, t_limit=None, r_limit=None, ooc_l_limit=None, ooc_r_limit=None, frontalis_limit=None, task=None):
        self.configs = {
            '0.9999': {
                'phases': {
                    1: {
                        'degree': 0.000354841096639418,
                        'expected_steps': 75,
                        'rotation_limit': 0.05351417,
                        'translation_limit': 0.00007502,  # Add missing key with default value
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                        'task': 'Neutral to V'
                    },
                    # 2: {
                    #     'degree': 0.000354841096639418,
                    #     'expected_steps': 45,  # 20108/33076 × 75 ≈ 45
                    #     'rotation_limit': 0.05351417,
                    #     'task': 'N to AA'
                    # },
                    # 3: {
                    #     'degree': 0.000354841096639418,
                    #     'expected_steps': 61,  # 27029/33076 × 75 ≈ 61
                    #     'rotation_limit': 0.05351417,
                    #     'task': 'AA to T'
                    # },
                    # 4: {
                    #     'degree': 0.000354841096639418,
                    #     'expected_steps': 42,  # 18457/33076 × 75 ≈ 42
                    #     'rotation_limit': 0.05351417,
                    #     'task': 'T to IY'
                    # },
                    # 5: {
                    #     'degree': 0.000354841096639418,
                    #     'expected_steps': 105,  # 46178/33076 × 75 ≈ 105
                    #     'rotation_limit': 0.05351417,
                    #     'task': 'IY to V'
                    # },
                    # 6: {
                    #     'degree': 0.000354841096639418,
                    #     'expected_steps': 48,  # 21298/33076 × 75 ≈ 48
                    #     'rotation_limit': 0.05351417,
                    #     'task': 'V to N'
                    # },
                    # 7: {
                    #     'degree': 0.000354841096639418,
                    #     'expected_steps': 136,  # 60141/33076 × 75 ≈ 136
                    #     'rotation_limit': 0.05351417,
                    #     'task': 'N to Y'
                    # },
                    # 8: {
                    #     'degree': 0.000354841096639418,
                    #     'expected_steps': 45,  # Same as phase 2 (similar expected_steps)
                    #     'rotation_limit': 0.05351417,
                    #     'task': 'Y to UW'
                    # }
                },
                'result_folder': None
            },

            '0.999': {
                'phases': {
                    1: {
                        'degree': target_rotation_deg or 0.000354841096639418,  # Use passed value or default
                        'expected_steps': estimated_steps or 75,
                        'rotation_limit': r_limit or 0.05351417,
                        'translation_limit': t_limit or 0.00007502, #  or 0.00007502,
                        'ooc_l_translation_limit': ooc_l_limit or 0.0001,  # Add this
                        'ooc_r_translation_limit': ooc_r_limit or 0.0001,  # Add this
                        'frontalis_translation_limit': frontalis_limit or 0.0001,  # Add this                        
                        'task': task or 'Neutral to V'                        
                    },
                    2: {
                        'degree': 0.000354841096639418,
                        'expected_steps': 45,
                        'rotation_limit': 0.05351417,
                        'task': 'N to AA'
                    },
                    3: {
                        'degree': 0.000354841096639418,
                        'expected_steps': 61,
                        'rotation_limit': 0.05351417,
                        'task': 'AA to T'
                    },
                    4: {
                        'degree': 0.000354841096639418,
                        'expected_steps': 42,
                        'rotation_limit': 0.05351417,
                        'task': 'T to IY'
                    },
                    5: {
                        'degree': 0.000354841096639418,
                        'expected_steps': 105,
                        'rotation_limit': 0.05351417,
                        'task': 'IY to V'
                    },
                    6: {
                        'degree': 0.000354841096639418,
                        'expected_steps': 48,
                        'rotation_limit': 0.05351417,
                        'task': 'V to N'
                    },
                    7: {
                        'degree': 0.000354841096639418,
                        'expected_steps': 136,
                        'rotation_limit': 0.05351417,
                        'task': 'N to Y'
                    },
                    8: {
                        'degree': 0.000354841096639418,
                        'expected_steps': 45,
                        'rotation_limit': 0.05351417,
                        'task': 'Y to UW'
                    }
                },
                'result_folder':str(results_path / "job_250611-105222/250611-105237")
            },

            '0.99': {
                'phases': {
                    1: {
                        'degree': 0.04044802306349184,
                        'expected_steps': 750,
                        'rotation_limit': 0.000238,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    },
                    2: {
                        'degree': 0.06112688,  # rotation_anatomy_limit_phase2
                        'expected_steps': 260,
                        'rotation_limit': 0.02019004,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    },
                    3: {
                        'degree': 0.06112688,  # Not provided in your data
                        'expected_steps': 260,
                        'rotation_limit': 0.02019004,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    }
                },
                'result_folder':str(results_path / "job_250611-132640/250611-132654")
            },            

            '0.9': {
                'phases': {
                    1: {
                        'degree': 0.3651137509011078,
                        'expected_steps': 3503,
                        'rotation_limit': 0.00009884,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    },
                    2: {
                        'degree': 0.9080949540937429,
                        'expected_steps': 3266,
                        'rotation_limit': 0.00035912,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    },
                    3: {
                        'degree': -0.1207412343196439,
                        'expected_steps': 968,
                        'rotation_limit': 0.00037021,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    }
                },
                'result_folder': None
            },           

            '0.5': {
                'phases': {
                    1: {
                        'degree': 1.5025544699609665,
                        'expected_steps': 17798,
                        'rotation_limit': 0.00010379,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    },
                    2: {
                        'degree': None,
                        'expected_steps': 17798,
                        'rotation_limit': 0.00031714,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    },
                    3: {
                        'degree': None,
                        'expected_steps': 17798,
                        'rotation_limit': 0.00031714,
                        'translation_limit': 0.00007502,  # Add missing key
                        'ooc_l_translation_limit': 0.0001,  # Add missing key
                        'ooc_r_translation_limit': 0.0001,  # Add missing key
                        'frontalis_translation_limit': 0.0001,  # Add missing key
                    }
                },
                'result_folder': None
            },    

            '0.1': {
                'phases': {
                    1: {
                        # 'degree': 1.80,                        
                        # 'expected_steps': 26507,
                        # 'rotation_limit':  0.00007502,
                        'degree': target_rotation_deg or 1.80,  # Use passed value or default
                        'expected_steps': estimated_steps or 26507,
                        'rotation_limit': r_limit or 0.00007502,
                        'translation_limit': t_limit, #  or 0.00007502,
                        'ooc_l_translation_limit': ooc_l_limit or 0.0001,  # Add this
                        'ooc_r_translation_limit': ooc_r_limit or 0.0001,  # Add this
                        'frontalis_translation_limit': frontalis_limit or 0.0001,  # Add this                        
                        'task': task or 'Neutral to V'
                    },
                    # 2: {
                    #     'degree': 2.760640384121652, # rotate down
                    #     'expected_steps': 20108,
                    #     'rotation_limit': 0.00015033,
                    #     'task': 'N to AA',
                    # },
                    # 3: {
                    #     'degree': -4.973067617011925, # rotate up
                    #     'expected_steps': 27029,
                    #     'rotation_limit': 0.00020326,
                    #     'task': 'AA to T',
                    # },
                    # 4: {
                    #     'degree': 0,
                    #     'expected_steps': 18457,
                    #     'rotation_limit': 0.001,
                    #     'task': 'T to IY',
                    # },
                    # 5: {
                    #     'degree': 1.8, 
                    #     'expected_steps': 46178,
                    #     'rotation_limit': 4.306e-5,
                    #     'task': 'IY to V',
                    # },
                    # 6: {
                    #     'degree': 4.81,
                    #     'expected_steps': 53856,
                    #     'rotation_limit': 0.00009767,
                    #     'task': 'V to N',
                    # },
                    # 7: {
                    #     'degree': 3.00,
                    #     'expected_steps': 20822,
                    #     'rotation_limit': 0.00015924,
                    #     'task': 'N to Y',
                    # },
                    # 8: {
                    #     'degree': 0, # rotate down
                    #     'expected_steps': 76816,
                    #     'rotation_limit': 0.001,
                    #     'task': 'Y to UW',
                    # }
                },
                'result_folder': None
            },    

        }

        # Duplicate 0.9 configuration for 0.8, 0.7, and 0.6 to ensure they use fast testing settings
        for t in ['0.8', '0.7', '0.6']:
            if t not in self.configs:
                self.configs[t] = self.configs['0.9']
    
    def _resolve_threshold(self, threshold):
        threshold_key = str(threshold)
        if threshold_key in self.configs:
            return threshold_key
        try:
            val = float(threshold)
            closest_key = min(self.configs.keys(), key=lambda k: abs(float(k) - val))
            print(f"[INFO] Threshold {threshold} not found. Using closest: {closest_key}")
            return closest_key
        except Exception:
            return '0.999'

    def get_phase_config(self, threshold, phase):
        """Get configuration for a specific threshold and phase"""
        threshold_key = self._resolve_threshold(threshold)
        if phase not in self.configs[threshold_key]['phases']:
            raise ValueError(f"Phase {phase} not found for threshold {threshold_key}")
            
        return self.configs[threshold_key]['phases'][phase]

    def set_current_threshold(self, threshold, phase=1):
        """Set the current threshold and phase, then update parameters
        
        Args:
            threshold (float/str): The threshold key to use (e.g., '0.99', '0.9')
            phase (int): The phase number (default: 1)
        
        Returns:
            self: Returns the instance for method chaining
        """
        threshold_key = self._resolve_threshold(threshold)
        phase_config = self.get_phase_config(threshold_key, phase)
        
        # Update all parameters from the phase configuration
        self.expected_total_steps_for_jaw = phase_config['expected_steps']
        self.rotation_anatomy_limit = phase_config['rotation_limit']
        self.translation_anatomy_limit = phase_config['translation_limit']
        # self.ooc_l_translation_limit = 
        # self.ooc_r_translation_limit = 
        # self.frontalis_translation_limit = 
        self.ooc_l_translation_limit = phase_config['ooc_l_translation_limit'] # phase_config.get('ooc_l_translation_limit', 0.0001)  # Add this
        self.ooc_r_translation_limit = phase_config['ooc_r_translation_limit'] # phase_config.get('ooc_r_translation_limit', 0.0001)  # Add this
        self.frontalis_translation_limit = phase_config['frontalis_translation_limit'] # phase_config.get('frontalis_translation_limit', 0.0001)  # Add this

        self.degree = phase_config['degree']
        self.magnitude = phase_config['magnitude']
        
        # Store additional information
        self.current_threshold = threshold_key
        self.current_phase = phase
        self.result_folder = self.configs[threshold_key]['result_folder']
        
        return self


    def get_all_phases(self, threshold):
        """Return all phases for a threshold in a structured format"""
        threshold_key = self._resolve_threshold(threshold)
        
        phases = self.configs[threshold_key]['phases']
        
        # Convert to dictionary of parameters by phase
        params = {
            'expected_total_steps_for_jaw': {},
            'rotation_anatomy_limit': {},
            'ooc_l_translation_limit': {},  # Add this
            'ooc_r_translation_limit': {},  # Add this
            'frontalis_translation_limit': {},  # Add this            
            'unit_prime_jaw_translate': {},
            'translation_anatomy_limit': {},
            'degrees': {},
            # Add other parameters as needed
        }
        
        for phase_num, config in phases.items():
            # Debug: Print what keys are available in config
            print(f"[DEBUG] Phase {phase_num} config keys: {config.keys()}")

            params['expected_total_steps_for_jaw'][phase_num] = config['expected_steps']
            params['rotation_anatomy_limit'][phase_num] = config['rotation_limit']
            params['translation_anatomy_limit'][phase_num] = config.get('translation_limit', 0.00007502)  # Use .get() with default
            params['ooc_l_translation_limit'][phase_num] = config.get('ooc_l_translation_limit', 0.0001)  # Use .get() with default
            params['ooc_r_translation_limit'][phase_num] = config.get('ooc_r_translation_limit', 0.0001)  # Use .get() with default
            params['frontalis_translation_limit'][phase_num] = config.get('frontalis_translation_limit', 0.0001)  # Use .get() with default

            # params['unit_prime_jaw_translate'][phase_num] = config['unit_prime_jaw_translate']
            params['degrees'][phase_num] = config['degree']
        
        return params


# Call the helper function to get the values
# target_rotation_deg, estimated_steps, r_limit, this_task, x_bar, maya_file_to_open= j77_V3_estimate_r_limit_newest.calculate_rotation_parameters()
target_rotation_deg, estimated_steps, t_limit, r_limit, ooc_l_limit, ooc_r_limit, frontalis_limit, this_task, x_bar, maya_file_to_open, estimated_completion_steps_ooc_l, estimated_completion_steps_ooc_r = j77_V3_estimate_r_limit_newest.calculate_rotation_parameters()
# x_bar = getattr(dataV22_for_rest_adhoc, "x_middleres_OrbicularisOris" + str(i))
# don't use this unless very necessary (i.e. you dont have calculated results of x_bar like phoneme N)


# Initialize config with the calculated values
jaw_config = JawRotationConfig(
    target_rotation_deg=target_rotation_deg,
    estimated_steps=estimated_steps,
    t_limit=t_limit,
    r_limit=r_limit,
    ooc_l_limit=ooc_l_limit,
    ooc_r_limit=ooc_r_limit,
    frontalis_limit=frontalis_limit,
    task=this_task
)

current_threshold = str(args['distance_jaw_threshold_percentage'])
jaw_config.get_all_phases(current_threshold)

# Load all phases at once
phase_params = jaw_config.get_all_phases(current_threshold)

config_key = args["distance_threshold_percentage"]

# phase_params_tempt = jaw_config.configs[config_key]['phases'][current_threshold]
# this_task = phase_params_tempt['task']  # This will give you the full string "Neutral to IY"

contract_to_neutral_task = ['AA to T', 'IY to V', 'UW to T']        



# If you specifically want the second part ("IY" in this case)
task_parts = this_task.split(' to ')
destination_phoneme = task_parts[1]  # This gives "IY"
print("Destination phoneme:", destination_phoneme)

# this_task = phase_params['task'][1] # "Neutral to N"

print("this_task", this_task)

import dataV22
importlib.reload(dataV22)
import dataV22_for_rest_adhoc
importlib.reload(dataV22_for_rest_adhoc)

sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_ooc_l"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_ooc_r"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_frontalis"))

import joint_position_ooc_l_base_0
import joint_position_ooc_l_base_pose11_0
import joint_position_ooc_l_base_pose11_AU1_0
import joint_position_ooc_l_base_pose11_AU1_AU2_0

import joint_position_ooc_r_base_0
import joint_position_ooc_r_base_pose11_0
import joint_position_ooc_r_base_pose11_AU1_0
import joint_position_ooc_r_base_pose11_AU1_AU2_0

import joint_position_frontalis_base_0
import joint_position_frontalis_base_AU1_0
import joint_position_frontalis_base_AU1_AU2_0

    
vertex_index_combined = joint_position_ooc_l_base_0.sampled_points_vertex_idx + joint_position_ooc_l_base_pose11_0.sampled_points_vertex_idx  + joint_position_ooc_l_base_pose11_AU1_0.sampled_points_vertex_idx  + joint_position_ooc_l_base_pose11_AU1_AU2_0.sampled_points_vertex_idx

# vertex_index = vertex_index_combined
vertex_ooc_l_index = vertex_index_combined    # [-1:] # [-3:]
print(" vertex_index ooc_l length is", len(vertex_ooc_l_index))
print(" vertex_index ooc_l ", vertex_ooc_l_index[0:10])

vertex_index_combined = joint_position_ooc_r_base_0.sampled_points_vertex_idx + joint_position_ooc_r_base_pose11_0.sampled_points_vertex_idx  + joint_position_ooc_r_base_pose11_AU1_0.sampled_points_vertex_idx  + joint_position_ooc_r_base_pose11_AU1_AU2_0.sampled_points_vertex_idx

# vertex_index = vertex_index_combined
vertex_ooc_r_index = vertex_index_combined    # [-1:] # [-3:]
print(" vertex_index ooc_r length is", len(vertex_ooc_r_index))
print(" vertex_index ooc_r ", vertex_ooc_r_index[0:10])

vertex_index_combined = joint_position_frontalis_base_0.sampled_points_vertex_idx + joint_position_frontalis_base_AU1_0.sampled_points_vertex_idx  + joint_position_frontalis_base_AU1_AU2_0.sampled_points_vertex_idx

# vertex_index = vertex_index_combined
vertex_frontalis_index = vertex_index_combined    # [-1:] # [-3:]
print(" vertex_index frontalis length is", len(vertex_frontalis_index))
print(" vertex_index frontalis ", vertex_frontalis_index[0:10])




def retrieve_landmark_positions(maya_file_to_open, joint_groups_to_query):
    """
    Retrieves the world space positions of a group of landmarks from a Maya file.
    
    Args:
        maya_file_to_open (str): Path to the Maya file.
        joint_groups_to_query (dict): A dictionary where keys are joint group names 
                                      and values are the number of landmarks.
                                      e.g., {'ooc_r': 300, 'frontalis': 300}

    Returns:
        dict: A dictionary where keys are joint group names and values are lists of locations.
    """
    real_stdout = sys.stdout
    try:
        # Redirect stdout and stderr to null
        
        sys.stdout = open(os.devnull, 'w')
        
        # print("passed the test, now quit.")
        # # import sys
        # sys.exit(0)  # Exit with status code 0          

        sys.stderr = open(os.devnull, 'w')        
        # This print will bypass devnull and actually show up in your console:
        # import inspect
        

        maya.standalone.initialize()
        print("Maya standalone initialized.")
    except Exception as e:
        print(f"Maya standalone already initialized or failed to initialize: {e}")

    if not os.path.exists(maya_file_to_open):
        raise RuntimeError(f"File not found: {maya_file_to_open}")

    try:
        opened_file = cmds.file(maya_file_to_open, open=True, force=True)
        
        print("Opened file:", maya_file_to_open)
    except Exception as e:
        print(f"Failed to open Maya file: {e}")
        if 'maya.standalone' in sys.modules:
             maya.standalone.uninitialize()
        return {}

    # Function to delete existing cache nodes
    def delete_cache_nodes():
        cache_nodes = cmds.ls(type='cacheFile')
        for node in cache_nodes:
            cmds.delete(node)

    # Delete existing cache nodes
    delete_cache_nodes()

    cmds.cacheFile(createCacheNode=True, singleCache=True, startTime=1, endTime=50, fileName='myCache', dir='cacheDir')

    for frame_number in range(15):
        cmds.currentTime(str(frame_number), edit=True)  
    # cmds.currentTime(15)

    retrieved_positions = {}
    for joint_group_name, num_landmarks in joint_groups_to_query.items():
        landmark_locations = []
        print(f"Retrieving landmarks for group: {joint_group_name}")
        for i in range(num_landmarks):
            if i + 1 <= 9:
                joint_name = f'{joint_group_name}_0{i + 1}'
            else:
                joint_name = f'{joint_group_name}_{i + 1}'

            if cmds.objExists(joint_name):
                location = cmds.xform(joint_name, query=True, translation=True, worldSpace=True)
                landmark_locations.append(location)
            else:
                print(f"Warning: Joint '{joint_name}' not found.")
        
        retrieved_positions[joint_group_name] = landmark_locations
        print(f"Found {len(landmark_locations)} landmarks for {joint_group_name}.")

    cmds.select(cl=True)
    cmds.currentCtx()
    # Close the reference    



    # try:
    #     maya.standalone.uninitialize()
    #     print("Maya standalone uninitialized.")
    # except Exception as e:
    #     print(f"Failed to uninitialize Maya standalone: {e}")

    sys.stdout = real_stdout
    sys.stderr = sys.__stderr__

    return retrieved_positions


def retrieve_tissue_coordinates(mesh_names):
    """
    Retrieves the world space coordinates of vertices for a list of meshes.

    Args:
        mesh_names (list): A list of mesh object names to query.

    Returns:
        dict: A dictionary where keys are mesh names and values are lists of vertex coordinates.
    """
    real_stdout = sys.stdout

    import maya.cmds as cmds
    all_coordinates = {}
    
    try:
        # Redirect stdout and stderr to null to suppress Maya's output
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')        
        maya.standalone.initialize()
    except Exception:
        # Maya already initialized
        pass



    try:
        opened_file = cmds.file(maya_file_to_open, open=True, force=True)
        
        print("Opened file:", maya_file_to_open)
    except Exception as e:
        print(f"Failed to open Maya file: {e}")
    # Function to delete existing cache nodes
    def delete_cache_nodes():
        cache_nodes = cmds.ls(type='cacheFile')
        for node in cache_nodes:
            cmds.delete(node)

    # Delete existing cache nodes
    delete_cache_nodes()

    cmds.cacheFile(createCacheNode=True, singleCache=True, startTime=1, endTime=50, fileName='myCache', dir='cacheDir')

    # cmds.currentTime(15)
    for frame_number in range(15):
        cmds.currentTime(str(frame_number), edit=True)      

    for mesh_name in mesh_names:
        if cmds.objExists(mesh_name):
            # Using a single call to xform is much faster for many vertices
            all_vertex_positions = cmds.xform(f"{mesh_name}.vtx[*]", query=True, translation=True, worldSpace=True)
            # The result is a flat list [x1, y1, z1, x2, y2, z2, ...], so we group it
            coordinates = [all_vertex_positions[i:i+3] for i in range(0, len(all_vertex_positions), 3)]
            all_coordinates[mesh_name] = coordinates
        else:
            all_coordinates[mesh_name] = None
            # This print will be suppressed.
            print(f"Mesh '{mesh_name}' not found in the scene.")


    cmds.select(cl=True)
    cmds.currentCtx()
    # Close the reference                
    sys.stdout = real_stdout
    sys.stderr = sys.__stderr__    
            
    return all_coordinates


             

def main(x_bar):
    

    # Your code here
    loadPathRotation = r"D:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va002\job_240612-203822 rotation 0.05\240612-203830"

    if 1==run_a2_fromFrameGetDeltaB:
        if 1 == run_both:
            deltaBList = []
            
            save_folder = r"C:\Users\zpa18\keyLabLocalplaceHolder"

            # action_mode = "translation"    
            # action_mode = "rotation"    
            action_mode = "both"    
            for i in range(0, 1):  # keyframe_num - 1
            
                log_path = None

                
                # x_bar = getattr(dataV22_for_rest_adhoc, "x_middleres_OrbicularisOris" + str(i))

                # neutral to N
                
                # path_2 not used here.
                previous_pose_folder_2 =str(results_path / "job_250429-210906/250429-210922")

                if this_task in contract_to_neutral_task:
                    AA_to_rest_flag = 1
                else:
                    AA_to_rest_flag = 0

                if 1 == 1: # neutral_to_other_pose_flag = 1
                    # if enter, neutral_to_other_pose_flag = 1
                    try:
                        # Try to open and read data from the first file
                        # with open(previous_pose_folder + "/deltaB_temp.pkl", 'rb') as f: # deltaB_temp.pkl
                        #     data_t = pickle.load(f)
                        #     deltaB = data_t['delta_b0_conca']
                            
                            joint_groups_to_query = {
                                'oor': 300,
                                'ooc_l': 300,
                                'ooc_r': 300,
                                'frontalis': 300
                            }
                            
                            retrieved_positions = retrieve_landmark_positions(maya_file_to_open, joint_groups_to_query)
                            
                            # print("passed the test, now quit.")
                            # sys.exit(0) # Exit with status code 0
                            x_joint = retrieved_positions.get('oor', [])
                            x_ooc_l_joint = retrieved_positions.get('ooc_l', [])
                            x_ooc_r_joint = retrieved_positions.get('ooc_r', [])
                            x_frontalis_joint = retrieved_positions.get('frontalis', [])
                            
                            # x_frontalis_joint = data_t['x_frontalis_joint_']

                            # x_joint_rotation = data_t['x_joint_rotation_']

                            tissue_names_to_query = [
                                "middleres_OrbicularisOris",
                                "middleres_Jaw",
                                "middleres_OrbicularisOculi_l",
                                "middleres_OrbicularisOculi_r",
                                "middleres_Frontalis"
                            ]
                            
                            tissue_coordinates = retrieve_tissue_coordinates(tissue_names_to_query)
                            


                            s_next = tissue_coordinates.get("middleres_OrbicularisOris")
                            s_jaw_next = tissue_coordinates.get("middleres_Jaw")
                            s_ooc_l_next = tissue_coordinates.get("middleres_OrbicularisOculi_l")
                            s_ooc_r_next = tissue_coordinates.get("middleres_OrbicularisOculi_r")
                            s_frontalis_next = tissue_coordinates.get("middleres_Frontalis")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        print("Trying to open and read data from the second file...")


                    x_bar = s_next
                    x_jaw_bar = s_jaw_next

                    # # quick fix, the wrong opposite sequence:

                    # tempt001 = s_ooc_l_next
                    # s_ooc_l_next = s_ooc_r_next
                    # s_ooc_r_next = tempt001

                    x_ooc_l_bar = s_ooc_l_next
                    x_ooc_r_bar = s_ooc_r_next
                    x_frontalis_bar = s_frontalis_next


                    
                    
                    x_ooc_l_bar_original = getattr(dataV22_for_rest_adhoc, "x_middleres_OrbicularisOculi_l" + str(i))
                    x_ooc_r_bar_original = getattr(dataV22_for_rest_adhoc, "x_middleres_OrbicularisOculi_r" + str(i))
                    x_frontalis_bar_original = getattr(dataV22_for_rest_adhoc, "x_middleres_Frontalis" + str(i))           

                    # Simple quick comparison
                    print(f"\n=== QUICK COMPARISON FOR i={i} ===")
                    print(f"x_ooc_l_bar == x_ooc_l_bar_original: {np.array_equal(x_ooc_l_bar, x_ooc_l_bar_original)}")
                    print(f"x_ooc_r_bar == x_ooc_r_bar_original: {np.array_equal(x_ooc_r_bar, x_ooc_r_bar_original)}")
                    print(f"x_frontalis_bar == x_frontalis_bar_original: {np.array_equal(x_frontalis_bar, x_frontalis_bar_original)}")                                   

                else:
                    x_jaw_bar = getattr(dataV22_for_rest_adhoc, "x_middleres_Jaw" + str(i))  # not used in this experiment
                    x_ooc_l_bar = getattr(dataV22_for_rest_adhoc, "x_middleres_OrbicularisOculi_l" + str(i))
                    x_ooc_r_bar = getattr(dataV22_for_rest_adhoc, "x_middleres_OrbicularisOculi_r" + str(i))
                    x_frontalis_bar = getattr(dataV22_for_rest_adhoc, "x_middleres_Frontalis" + str(i))                    

                def visualize(vertices, object_name):
                    """
                    Visualize the vertices, the sampled joint positions, and the surface mesh.

                    Args:
                    - vertices (list of lists): List of 3D coordinates of vertices.
                    - faces (list of lists): List of indices defining the faces of the mesh.
                    - sampled_points (list of lists): List of 3D coordinates of sampled joint positions.
                    """
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    # Plot the vertices
                    vertices_np = np.array(vertices)
                    ax.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2], c='b', marker='o', s=1, label='Vertices')

                    # Set the view angle
                    ax.view_init(elev=(180+30), azim=0, roll=90)  # Change these values to set the desired view

                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    ax.legend()
                    # plt.show()
                    

                    # Save the plot to a file
                    output_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\d83 initiate rest to AU12 smile"
                    output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()     

                # coords_rest_non_concatenate = x_bar  # x_bar
                # # coords_rest_non_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]
                # visualize(coords_rest_non_concatenate, "initial_pose_phoneme_N_V2")

                # print("passed the test, now quit.")
                # # import sys
                # sys.exit(0)  # Exit with status code 0  

                # x_jaw_bar = s_jaw_next
                # x_ooc_l_bar = s_ooc_l_next
                # x_ooc_r_bar = s_ooc_r_next
                # x_frontalis_bar = s_frontalis_next


                # x_joint = getattr(dataV22, "x_joint_lip_lower_concatenate" + str(i))
                # x_joint = getattr(dataV22, "x_joint_cancatenate" + str(i))
                
                # x_joint_first_wave = getattr(joint_position_first_wave, "x_joint_position_1")
                # x_joint_second_wave = getattr(joint_position, "x_joint_position_1")
                # x_joint = x_joint_first_wave + x_joint_second_wave
                x_joint_first_wave = getattr(joint_position_first_wave, "x_joint_position_1")
                x_joint_second_wave = getattr(joint_position, "x_joint_position_1")
                x_joint_third_wave = getattr(joint_position_third_wave, "x_joint_position_1")
                # x_joint = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave

                
                # x_jaw_joint = getattr(dataV22, "x_jaw_joint_concatenate" + str(i))           
                x_jaw_joint = getattr(dataV22_for_rest_adhoc, "x_joint_cancatenate" + str(i))  # temporaty use a dummy one
                # # useless

                x_jaw_joint_rotation = getattr(dataV22_for_rest_adhoc, "x_joint_rotation_cancatenate" + str(i)) # x_joint_rotation_cancatenate0 = [[0, 0, 0]]


                # # # Example usage: print the combined length of the positions
                
                # x_joint_first_wave = getattr(joint_position_ooc_l_base_0, "x_joint_position_1")
                # x_joint_second_wave = getattr(joint_position_ooc_l_base_pose11_0, "x_joint_position_1")
                # x_joint_third_wave = getattr(joint_position_ooc_l_base_pose11_AU1_0, "x_joint_position_1")
                # x_joint_fourth_wave = getattr(joint_position_ooc_l_base_pose11_AU1_AU2_0, "x_joint_position_1")
                # x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave

                # # vertex_index_combined = joint_position_ooc_l_base_0.sampled_points_vertex_idx + joint_position_ooc_l_base_pose11_0.sampled_points_vertex_idx  + joint_position_ooc_l_base_pose11_AU1_0.sampled_points_vertex_idx  + joint_position_ooc_l_base_pose11_AU1_AU2_0.sampled_points_vertex_idx

                # x_ooc_l_joint = x_joint_combined
                # print(f"Total length of combined x_ooc_l_joint: {len(x_joint_combined)}")
                
                # x_joint_first_wave = getattr(joint_position_ooc_r_base_0, "x_joint_position_1")
                # x_joint_second_wave = getattr(joint_position_ooc_r_base_pose11_0, "x_joint_position_1")
                # x_joint_third_wave = getattr(joint_position_ooc_r_base_pose11_AU1_0, "x_joint_position_1")
                # x_joint_fourth_wave = getattr(joint_position_ooc_r_base_pose11_AU1_AU2_0, "x_joint_position_1")
                # x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave                    

                # # folder_path = str(b98_sampling_path)\joint_positions\joint_position_ooc_r"
                # # sys.path.insert(0, folder_path)
                # # # Call the function and get the combined x_joint_position_1
                # # x_joint_combined = combine_x_joint_positions_from_folder(folder_path)
                
                # # Example usage: print the combined length of the positions
                # print(f"Total length of combined x_ooc_r_joint: {len(x_joint_combined)}")

                

                # x_ooc_r_joint = x_joint_combined

                # # print("x_ooc_r_joint D1 L680", len(x_ooc_r_joint))
                # x_joint_first_wave = getattr(joint_position_frontalis_base_0, "x_joint_position_1")
                # x_joint_second_wave = getattr(joint_position_frontalis_base_AU1_0, "x_joint_position_1")
                # x_joint_third_wave = getattr(joint_position_frontalis_base_AU1_AU2_0, "x_joint_position_1")
                
                # x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave

                # # folder_path = str(b98_sampling_path)\joint_positions\joint_position_ooc_r"
                # # sys.path.insert(0, folder_path)
                # # # Call the function and get the combined x_joint_position_1
                # # x_joint_combined = combine_x_joint_positions_from_folder(folder_path)
                
                # # Example usage: print the combined length of the positions
                # print(f"Total length of combined x_frontalis_joint: {len(x_joint_combined)}")

                

                # x_frontalis_joint = x_joint_combined



                # to update the joint position to the AA pose, as the starting pose.
                def update_joint_position_to_AA_pose(x_ooc_l_joint, x_ooc_r_joint, x_frontalis_joint):
                    deltaBList_temp_file_folder = previous_pose_folder
                    with open(deltaBList_temp_file_folder + "/deltaB_temp.pkl", 'rb') as f:
                        data_t = pickle.load(f)
                        delta_b0_conca = data_t['delta_b0_conca']

                    x_joint_conca = [item for sublist in x_joint for item in sublist]
                    x_joint_rotation_conca = [item for sublist in x_jaw_joint_rotation for item in sublist]
                    # x_ooc_l_joint_conca = [item for sublist in x_ooc_l_joint for item in sublist]
                    # x_ooc_r_joint_conca = [item for sublist in x_ooc_r_joint for item in sublist]
                    # print("x_ooc_r_joint_conca D1 L715", len(x_ooc_r_joint_conca))
                    # x_frontalis_joint_conca = [item for sublist in x_frontalis_joint for item in sublist]

                    x_joint_init_conca = x_joint_conca
                    # x_joint_init = np.array(x_joint_init_conca)
                    x_joint_rotation_init_conca = x_joint_rotation_conca
                    x_joint_rotation_init = np.array(x_joint_rotation_init_conca)   

                    # x_ooc_l_joint_init_conca = x_ooc_l_joint_conca
                    # x_ooc_l_joint_init = np.array(x_ooc_l_joint_init_conca)   

                    # x_ooc_r_joint_init_conca = x_ooc_r_joint_conca
                    # x_ooc_r_joint_init = np.array(x_ooc_r_joint_init_conca)   

                    # print("x_ooc_r_joint_init D1 L790", len(x_ooc_r_joint_init))

                    # x_frontalis_joint_init_conca = x_frontalis_joint_conca
                    # x_frontalis_joint_init = np.array(x_frontalis_joint_init_conca)   

                    

                    # delta_b0_conca = deltaBList[i-1]
                    # x_joint_ = np.array(x_joint_init_conca)


                    n = len(x_joint_conca)
                    joint_num = int(n/3) #153
                    n_rotate = len(x_joint_rotation_conca)
                    joint_rotate_num = int(n_rotate/3) #1
                    joint_ooc_l_num = len(vertex_ooc_l_index)
                    joint_ooc_r_num = len(vertex_ooc_r_index)
                    joint_frontalis_num = len(vertex_frontalis_index)

                    
                    # x_joint_ = x_joint_init + delta_b0_conca[0:n]
                    x_joint_rotation_ = x_joint_rotation_init + delta_b0_conca[n : n + 3*joint_rotate_num]
                    # x_ooc_l_joint_ = x_ooc_l_joint_init + delta_b0_conca[3*joint_num + 3*joint_rotate_num : 3*joint_num + 3*joint_rotate_num + 3*joint_ooc_l_num]
                    
                    # x_ooc_r_joint_ = x_ooc_r_joint_init + delta_b0_conca[3*joint_num + 3*joint_rotate_num + 3*joint_ooc_l_num : 3*joint_num + 3*joint_rotate_num + 3*joint_ooc_l_num + 3*joint_ooc_r_num]
                    # print("x_ooc_r_joint_init D1 L754", len(x_ooc_r_joint_init))
                    # print("x_ooc_r_joint D1 L754", len(x_ooc_r_joint))

                    # x_frontalis_joint_ = x_frontalis_joint_init + delta_b0_conca[3*joint_num + 3*joint_rotate_num + 3*joint_ooc_l_num + 3*joint_ooc_r_num : 3*joint_num + 3*joint_rotate_num + 3*joint_ooc_l_num + 3*joint_ooc_r_num + 3*joint_frontalis_num]

                    multiple_previous_folder = 0
                    if multiple_previous_folder == 1:
                        deltaBList_temp_file_folder = previous_pose_folder_2
                        with open(deltaBList_temp_file_folder + "/deltaBList_temp.pkl", 'rb') as f:
                            data_t = pickle.load(f)
                            deltaBList = data_t['deltaBList']
            
                        delta_b0_conca = deltaBList[i-1]

                        x_joint_ = x_joint_ + delta_b0_conca[0:n]
                        x_joint_rotation_ = x_joint_rotation_ + delta_b0_conca[n : n + 3*joint_rotate_num]                            

                    # x_joint_non_concatenate = [x_joint_[i:i+3] for i in range(0, len(x_joint_), 3)]
                    x_joint_rotation_non_concatenate = [x_joint_rotation_[i:i+3] for i in range(0, len(x_joint_rotation_), 3)]
                    
                    # x_ooc_l_joint_non_concatenate = [x_ooc_l_joint_[i:i+3] for i in range(0, len(x_ooc_l_joint_), 3)]
                    # x_ooc_r_joint_non_concatenate = [x_ooc_r_joint_[i:i+3] for i in range(0, len(x_ooc_r_joint_), 3)]
                    # x_frontalis_joint_non_concatenate = [x_frontalis_joint_[i:i+3] for i in range(0, len(x_frontalis_joint_), 3)]

                    # x_joint_non_concatenate, x_joint_rotation_non_concatenate, x_ooc_l_joint_non_concatenate, x_ooc_r_joint_non_concatenate, x_frontalis_joint_non_concatenate
                    return x_joint_rotation_non_concatenate
                
                # AA_to_rest_flag = 1  # 0
                # bypass this process
                if neutral_to_other_pose_flag == 0: 
                    # x_joint, x_jaw_joint_rotation, x_ooc_l_joint, x_ooc_r_joint, x_frontalis_joint
                    x_jaw_joint_rotation = update_joint_position_to_AA_pose(x_ooc_l_joint, x_ooc_r_joint, x_frontalis_joint)
                
                # print("x_ooc_r_joint D1 L780", len(x_ooc_r_joint))

                s_next = x_bar

                s_jaw_next = x_jaw_bar

                s_ooc_l_next = x_ooc_l_bar

                s_ooc_r_next = x_ooc_r_bar
                s_frontalis_next = x_frontalis_bar                    

                x_joint_ = x_joint
                x_jaw_joint_ = x_jaw_joint # useless
                x_ooc_l_joint_ = x_ooc_l_joint
                x_ooc_r_joint_ = x_ooc_r_joint

                # debug_save_folder =str(results_path / "240423-150653 log files/AU2_right_as_rest_pose/mb_file_offline_from_d1")
                # generate_output(x_ooc_r_joint_, debug_save_folder)
                # print("Debug output generation complete. Exiting.")
                # sys.exit(0)
            
                x_frontalis_joint_ = x_frontalis_joint                    

                x_jaw_joint_rotation_ = x_jaw_joint_rotation
                job_time_stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
                # with open(save_folder + "/log" + str(i) +".txt", 'a') as f:
                # s_next_conca = [item for sublist in s_next for item in sublist]   
                # s_next_conca = [item for sublist in s_next for item in sublist]
                # print("s_next_conca", s_next_conca[0:3])
                # print(" in d1 line 117, s_next is  ", control_vertices_slice(s_next_conca, vertex_index))

                params = {
                    'i_func': i,
                    # 'j_func': i + 1,
                    'x_joint_interFunctional': x_joint_,
                    'x_ooc_l_joint_interFunctional': x_ooc_l_joint_,
                    'x_ooc_r_joint_interFunctional': x_ooc_r_joint_,
                    'x_frontalis_joint_interFunctional': x_frontalis_joint_,
                    # 'x_jaw_joint_interFunctional': x_jaw_joint_,
                    'x_jaw_joint_rotation_interFunctional': x_jaw_joint_rotation_,
                    's_next_interFunctional': s_next,
                    's_jaw_next_interFunctional': s_jaw_next,
                    's_ooc_l_next_interFunctional': s_ooc_l_next,
                    's_ooc_r_next_interFunctional': s_ooc_r_next,
                    's_frontalis_next_interFunctional': s_frontalis_next,
                    'log_path': log_path,
                    'job_time_stamp': job_time_stamp,
                    'action_mode': action_mode,
                    'previous_pose_folder': previous_pose_folder_2,
                    'current_threshold': current_threshold,
                    'phase_params': phase_params,
                    'contract_to_neutral_task': contract_to_neutral_task,
                    'this_task': this_task,
                    'maya_file_to_open': maya_file_to_open,
                    'estimated_steps': estimated_steps,
                    'estimated_completion_steps_ooc_l': estimated_completion_steps_ooc_l,
                    'estimated_completion_steps_ooc_r': estimated_completion_steps_ooc_r,
                    **args
                }

                result = a2_fromFrameGetDeltaB.a2_fromFrameGetDeltaB(**params)   
        
                deltaB = result["deltaB"]
                x_joint_ = result["x_joint_"]
                x_ooc_r_joint_ = result["x_ooc_r_joint_"]
                x_frontalis_joint_ = result["x_frontalis_joint_"]
                x_ooc_l_joint_ = result["x_ooc_l_joint_"]                                        
                s_next = result["s_next"]
                s_ooc_r_next = result["s_ooc_r_next"]
                s_frontalis_next = result["s_frontalis_next"]
                s_ooc_l_next = result["s_ooc_l_next"]                                        
                save_folder = result["save_folder"]                                     
                # x_jaw_joint_ = result["x_joint_"]
                x_jaw_joint_rotation_ = result["x_joint_rotation_"]
                s_jaw_next = result["s_jaw_next"]
                # x_joint_rotation_ = result["x_jaw_joint_rotation_"]               


                deltaBList.append(deltaB) # i is frame index.

                # with open(save_folder + "/deltaBList_temp.pkl", 'wb') as f:
                #     pickle.dump({'deltaBList': deltaBList}, f)       

            # with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\log.txt", 'w') as f:
            #     print("deltaBList: \n \n", deltaBList, file = f)  

            # with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files" + "/deltaBList_temp.pkl", 'wb') as f:
            #     pickle.dump({'deltaBList': deltaBList}, f)


# Run the profiler

cProfile.run('main(x_bar)', 'output.prof')


# Create a Stats object
p = pstats.Stats('output.prof')

import datetime

saveFolder =str(results_path / "240423-150653 log files/computingTime_analysis")
filename = saveFolder + r"\computingTime_analysis_file"+str("_")+ datetime.datetime.now().strftime("%y%m%d-%H%M%S") #  +".mv"
os.makedirs(saveFolder, exist_ok=True)

# Open a text file for writing
with open(filename + ".txt", 'w') as f:
    # Redirect the output to the text file
    p.stream = f
    # Sort and print statistics to the text file
    p.sort_stats('cumulative').print_stats(60)

print("Profiling results saved to output.txt")
