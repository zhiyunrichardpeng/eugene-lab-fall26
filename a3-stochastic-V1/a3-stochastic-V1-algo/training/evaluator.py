import time
import numpy as np
import torch
import math
import os
from os import path
from pathlib import Path
from utils.initialization import create_env
from utils.common_utils import set_seed

import sys
# --- Resolve REPO_ROOT (a3-stochastic-git-V1) dynamically ---

# ==============================================================================
# Dynamic Repository Root Detection (Works via Shift+Alt+M commandPort)
# ==============================================================================
REPO_NAME = "a3-stochastic-Rich-local-V1"
REPO_ROOT = None

# 1. Check if 'paths' is already imported in this Maya session
if "paths" in sys.modules and hasattr(sys.modules["paths"], "__file__"):
    cached_path = Path(sys.modules["paths"].__file__).resolve().parent
    if (cached_path / "paths.py").is_file():
        REPO_ROOT = cached_path

# 2. Check if running from disk where __file__ is available
if not REPO_ROOT and "__file__" in globals() and __file__:
    script_path = Path(__file__).resolve()
    for p in [script_path] + list(script_path.parents):
        if (p / "paths.py").is_file():
            REPO_ROOT = p
            break

# 4. Standard lab & team clone fallbacks
if not REPO_ROOT:
    fallbacks = [
        Path(r"C:\keyLabLocal_D") / REPO_NAME,
        Path(r"D:\keyLabLocal_D") / REPO_NAME,
        Path.home() / "git" / REPO_NAME,
        Path.home() / "Documents" / "GitHub" / REPO_NAME,
    ]
    for fb in fallbacks:
        if (fb / "paths.py").is_file():
            REPO_ROOT = fb
            break

if not REPO_ROOT:
    raise RuntimeError(
        f"Could not locate '{REPO_NAME}' containing 'paths.py'. "
        "Verify your repository path."
    )

# Register repository root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import (
    a3_stochastic_path,
    a1_blendshape_path,
    backup_path,
)
sys.path.insert(0, str(a3_stochastic_path / "a3-stochastic-V1-algo"))
sys.path.insert(0, str(a3_stochastic_path / "a3-stochastic-V1-algo" / "example_train"))
sys.path.insert(0, str(a3_stochastic_path / "a3-stochastic-V1-algo" / "env_gym"))
sys.path.insert(0, str(a1_blendshape_path / "database"))
sys.path.insert(0, str(backup_path / "b31 dataV22 backup hub"))
sys.path.insert(0, str(a3_stochastic_path / "a3-stochastic-V1-algo"/ "training"/ "analysic_gradient"))

import importlib
from pytorch_analytical_gradient import AnalyticalGradientCalculator

import dataV22
importlib.reload(dataV22)

import dataV22_phoneme_AA

importlib.reload(dataV22_phoneme_AA)

import dataV22_phoneme_T

importlib.reload(dataV22_phoneme_T)

import dataV22_phoneme_IY

importlib.reload(dataV22_phoneme_IY)

import dataV22_phoneme_V

importlib.reload(dataV22_phoneme_V)

import dataV22_phoneme_N

importlib.reload(dataV22_phoneme_N)

import dataV22_phoneme_Y

importlib.reload(dataV22_phoneme_Y)

import dataV22_phoneme_UW

importlib.reload(dataV22_phoneme_UW)

from tqdm import tqdm
from scipy.spatial import cKDTree

from utils.distance_calculator import MeshDistanceCalculator
import importlib
from utils.shared_data import SharedData
import matplotlib.pyplot as plt
import csv
import pickle
from utils.phase_manager import PhaseManager

class Evaluator:
    def __init__(self, phase_manager=None, index=0, history_size=300, exploration_prob=0.2, **kwargs):
        print("Evaluator: ", __file__)
        for key, value in kwargs.items():
            setattr(self, key, value)         
        kwargs.update(
            {"reward_scale": None, "repeat_num": None}
        )  # evaluation don't need to scale reward

        # self.initial_ooc_l_distance = getattr(self, "initial_ooc_l_distance", 0.0)
        # self.initial_ooc_r_distance = getattr(self, "initial_ooc_r_distance", 0.0)
        # self.initial_frontalis_distance = getattr(self, "initial_frontalis_distance", 0.0)
        # self.initial_jaw_distance = getattr(self, "initial_jaw_distance", 0.0)

        if phase_manager is None:
            # Default to 2 phases if not specified
            self.phase_manager = PhaseManager(max_phases=2)
        else:
            self.phase_manager = phase_manager

        # Initialize phase-specific structures
        for phase in range(1, self.phase_manager.max_phases+1):
            self.phase_manager.phase_data[phase] = {
                'gradient_magnitudes': {},
                'term_history': {
                    'steps': [],
                    'distance': [],
                    'strain': [],
                    'muscle': []
                },
                'logs': {
                    'distance': [],
                    'step_counter': [],
                    # ... other logs
                }
            }


        self.distance_calculator = MeshDistanceCalculator()
        self.env = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        # file = __import__(alg_file_name)
        # ApproxContainer = getattr(file, "ApproxContainer")
        # self.networks = ApproxContainer(**kwargs)
        # self.render = kwargs["is_render"]
        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", False) #False
        self.vertex_index = kwargs["vertex_index"]
        # self.print_time = 0
        # self.print_iteration = -1
        self.print_every = kwargs["print_every"]
        self.print_Every_episode = kwargs["print_Every_episode"]
        self.step_counter_internal_max = kwargs["step_counter_internal_max"]
        self.debug_mode_vsCode = kwargs["debug_mode_vsCode"]
        self.n = kwargs["n"]
        self.tic = kwargs["tic"]
        self.x_joint_init = kwargs["x_joint_init"]
        self.i_func = kwargs["i_func"]
        self.j_func = kwargs["j_func"]
        # self.x_joint_interFunctional = kwargs["x_joint_interFunctional"]        
        # self.x_joint_rotation_interFunctional = kwargs["x_joint_rotation_interFunctional"]     
        self.joint_num = kwargs["joint_num"]
        self.vertices_num = kwargs["vertices_num"]
        self.vertices_jaw_num = kwargs["vertices_jaw_num"]
        self.joint_rotate_num = kwargs["joint_rotate_num"]
        self.x_joint_rotation_init = kwargs["x_joint_rotation_init"]
        self.x_bar_j = kwargs["x_bar_j"]
        self.action_mode = kwargs["action_mode"]
        self.keyframe_num = kwargs["keyframe_num"]        
        self.degrees_min = kwargs["degrees_min"]
        self.distance_min = kwargs["distance_min"]
        self.distance_threshold_percentage = kwargs["distance_threshold_percentage"]
        self.distance_jaw_min = kwargs["distance_jaw_min"]
        self.distance_jaw_threshold_percentage = kwargs["distance_jaw_threshold_percentage"]
        self.gradients_over_iterations = []
        self.gradients_over_iterations_dict = {}
        self.current_index = 0  # Track the current index to update
    
        self.prev_gradients = None
        self.recent_indices = []
        self.history_size = history_size
        self.exploration_prob = exploration_prob
        self.step_t1 = 0
        self.step_objective = 0
        self.count_diff_log = 0
        self.next_info = {}
        # self.phase_2_flag = 0
        self.phase_manager.current_phase = phase  # Set the current phase        
        """this is from main branch. WIll use as minimal viable product to test the analytical gradient calculator"""
        # 1. Instantiate the specific analytical calculators for all tissues
        self.analytical_calculator = AnalyticalGradientCalculator(
            str(a3_stochastic_path / "automation"/ "oor_bind_pose.json"),
            str(a3_stochastic_path / "automation"/ "oor_skin_weights.json"),
            vertex_index=self.vertex_index
        )
        print("Using analytical calculator")
        """Save these for later"""
        # self.analytical_calculator_jaw = AnalyticalGradientCalculator(
        #     bind_pose_json_path=str(a3_stochastic_path / "automation"/ "jaw_bind_pose.json"), 
        #     skin_weights_json_path=str(a3_stochastic_path / "automation"/ "jaw_skin_weights.json"), 
        #     vertex_index=self.vertex_jaw_index,
        #     joint_pivots_json_path=str(a3_stochastic_path / "automation"/ "jaw_joint_pivots.json")
        # )
        self.analytical_calculator_ooc_l = AnalyticalGradientCalculator(
            bind_pose_json_path=str(a3_stochastic_path / "automation"/ "ooc_l_bind_pose.json"),
            skin_weights_json_path=str(a3_stochastic_path / "automation"/ "ooc_l_skin_weights.json"),
            vertex_index=self.vertex_ooc_l_index
        )
        # self.analytical_calculator_ooc_r = AnalyticalGradientCalculator(
        #     bind_pose_json_path=str(a3_stochastic_path / "automation"/ "ooc_r_bind_pose.json"),
        #     skin_weights_json_path=str(a3_stochastic_path / "automation"/ "ooc_r_skin_weights.json"),
        #     vertex_index=self.vertex_ooc_r_index
        # )
        # self.analytical_calculator_frontalis = AnalyticalGradientCalculator(
        #     bind_pose_json_path=str(a3_stochastic_path / "automation"/ "frontalis_bind_pose.json"),
        #     skin_weights_json_path=str(a3_stochastic_path / "automation"/ "frontalis_skin_weights.json"),
        #     vertex_index=self.vertex_frontalis_index
        # )

        self.analytical_gradient_scale = 0.022

        self.next_info.update({

            'step_counter_internal': 0,

            # Add more variables as needed
        })

        self.count_diff_log_phase2 = 0

        self.term_histories = {
            1: {
                'forward': {'steps': [], 'distance': [], 'strain': [], 'muscle': []},
                'diff': {'steps': [], 'distance': [], 'strain': [], 'muscle': []},
                'count': {'value': 0}
            }
        }        

        self.gradient_magnitudes = {
            phase: {} for phase in range(1, self.phase_manager.max_phases + 1)
        }

        self.r_limit_sigmoid_1000_endList = []
        self.t_limit_sigmoid_1000_endList = []


        self.w1 = 0.34
        self.w2 = 0.33
        self.w3 = 0.33

        # pre-compute the selected observing vertices.
        self.x_bar_j_selected = self.control_vertices_slice(self.x_bar_j, self.vertex_index)
        
        if "both" == self.action_mode:
            if self.distance_min/self.distance_threshold_percentage < SharedData.smallest_distance and self.distance_jaw_min/self.distance_jaw_threshold_percentage < SharedData.smallest_jaw_distance:
                SharedData.smallest_distance = self.distance_min/self.distance_threshold_percentage
                SharedData.smallest_jaw_distance = self.distance_jaw_min/self.distance_jaw_threshold_percentage                                       
                # self.save_best_result(next_info)
                # print(f"New smallest distance: {SharedData.smallest_distance}")    
                


        debug_mode_vsCode = 1

        if debug_mode_vsCode == 1:
            import maya.standalone
            maya.standalone.initialize()    
            import maya.cmds as cmds
            import maya.OpenMaya as OpenMaya    

            import sys
            from PyQt5.QtCore import Qt, QCoreApplication
            from PyQt5.QtWidgets import QApplication, QLabel        
            
            # Set the attribute before creating the QApplication
            QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

            app = QApplication(sys.argv)


        self.cmds = cmds
        self.OpenMaya = OpenMaya        

    def load_state_dict(self, state_dict):
        # self.networks.load_state_dict(state_dict)
        pass

    def run_evaluation(self, phase=None, payload=None):

        """Generic evaluation function that handles all phases"""
        if phase is None:
            phase = self.phase_manager.current_phase
            current_phase = phase
        
        exit_flag = 0
        obs_list = []
        action_list = []
        action_real_list = []
        reward_list = []
        distance_list = []
        distance_jaw_list = []
        distance_ooc_l_list = []
        distance_ooc_r_list = []        
        success_flag_list = []
        
        # Initialize environment based on phase
        if phase == 1:
            obs, info = self.env.reset(return_info=True)
        else:  # Phase 2
            info = payload["next_info"]
            obs = payload["next_obs"]
            # Setup phase 2 specific parameters
            if phase == 2:
                target_state_temp = getattr(dataV22_phoneme_AA, "x_middleres_OrbicularisOris" + str(1))      
                x_jaw = getattr(dataV22_phoneme_AA, "x_middleres_Jaw" + str(1))
                elements = dataV22_phoneme_AA.x_middleres_OrbicularisOris_face_0
                this_task = "N to AA"

            if phase == 3:
                target_state_temp = getattr(dataV22_phoneme_T, "x_middleres_OrbicularisOris" + str(1))      
                x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
                elements = dataV22_phoneme_T.x_middleres_OrbicularisOris_face_0
                this_task = "AA to T"

            if phase == 4:
                target_state_temp = getattr(dataV22_phoneme_IY, "x_middleres_OrbicularisOris" + str(1))      
                x_jaw = getattr(dataV22_phoneme_IY, "x_middleres_Jaw" + str(1))
                elements = dataV22_phoneme_IY.x_middleres_OrbicularisOris_face_0
                this_task = "T to IY"

            if phase == 5:
                target_state_temp = getattr(dataV22_phoneme_V, "x_middleres_OrbicularisOris" + str(1))      
                x_jaw = getattr(dataV22_phoneme_V, "x_middleres_Jaw" + str(1))
                elements = dataV22_phoneme_V.x_middleres_OrbicularisOris_face_0
                this_task = "IY to V"

            if phase == 6:
                target_state_temp = getattr(dataV22_phoneme_N, "x_middleres_OrbicularisOris" + str(1))      
                x_jaw = getattr(dataV22_phoneme_N, "x_middleres_Jaw" + str(1))
                elements = dataV22_phoneme_N.x_middleres_OrbicularisOris_face_0
                this_task = "V to N"

            if phase == 7:
                target_state_temp = getattr(dataV22_phoneme_Y, "x_middleres_OrbicularisOris" + str(1))      
                x_jaw = getattr(dataV22_phoneme_Y, "x_middleres_Jaw" + str(1))
                elements = dataV22_phoneme_Y.x_middleres_OrbicularisOris_face_0
                this_task = "N to Y"

            if phase == 8:
                target_state_temp = getattr(dataV22_phoneme_UW, "x_middleres_OrbicularisOris" + str(1))      
                x_jaw = getattr(dataV22_phoneme_UW, "x_middleres_Jaw" + str(1))
                elements = dataV22_phoneme_UW.x_middleres_OrbicularisOris_face_0
                this_task = "Y to UW"



            x_conca = [item for sublist in target_state_temp for item in sublist]            
            x_jaw_conca = [item for sublist in x_jaw for item in sublist]
            target_state = np.asarray(x_conca)
            target_jaw_state = np.asarray(x_jaw_conca)
            x_bar_j = payload["s_next"]
            x_jaw_bar_j = payload["s_jaw_next"]
            x_bar_j_conca = np.asarray([item for sublist in x_bar_j for item in sublist])
            x_jaw_bar_j_conca = np.asarray([item for sublist in x_jaw_bar_j for item in sublist])
            x_bar_j = x_bar_j_conca
            x_jaw_bar_j = x_jaw_bar_j_conca
            self.step_objective = 0
            
            # Calculate initial distances
            method = 'euclidean'
            initial_distance = self.distance_calculator.calculate_distance(
                method, 
                self.control_vertices_slice(target_state, self.vertex_index), 
                self.control_vertices_slice(x_bar_j, self.vertex_index)
            )           
            initial_jaw_distance = self.distance_calculator.calculate_distance(
                method, 
                self.control_vertices_slice(target_jaw_state, self.vertex_jaw_index), 
                self.control_vertices_slice(x_jaw_bar_j, self.vertex_jaw_index)
            )           


            info['distance_jaw'] = initial_jaw_distance
            
            # Set phase 2 specific parameters
            distance_min = initial_distance * self.distance_threshold_percentage
            distance_jaw_min = initial_jaw_distance * self.distance_jaw_threshold_percentage
            expected_total_steps = 5200
            # this_task = "N to AA"
            # elements = dataV22_phoneme_AA.x_middleres_OrbicularisOris_face_0
            
            with open(self._get_log_filename(True), 'a') as f:
                print(f"distance_min: {distance_min}", file=f)        
                print(f"distance_jaw_min: {distance_jaw_min}", file=f)        
                print(f"this_task: {this_task}", file=f)        

        done = 0
        info["TimeLimit.truncated"] = False
        self.step_t1 = 0
        from tqdm import tqdm
        count = self.step_counter_internal_max
        pbar = tqdm(total=count)
        update_every = 200
        # self.phase_2_flag = 1 if phase == 2 else 0
        self.phase_manager.current_phase = phase  # Set the current phase
        step_counter_internal = 0 
        abc_execution_count = 0
        MAX_ABC_EXECUTIONS = self.overlapping_length

        landmark_trajectory = []
        vertex_trajectory = []

        while not (done or info["TimeLimit.truncated"]):
            self.step_t1 += 1
            if self.step_t1 % 3*self.joint_num == 0:
                self.recent_indices = []
                        
            # Extract positions and targets from observation
            positions = obs[3*self.joint_num + 3*self.joint_rotate_num : 3*self.joint_rotate_num + 3*self.joint_num + 3*self.vertices_num]
            positions_jaw = obs[3*self.joint_rotate_num + 3*self.joint_num + 3*self.vertices_num : 3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num]
            positions_lip_joint = obs[0 : 3*self.joint_num]
            """ From main branch"""
            landmark_trajectory.append(np.copy(positions_lip_joint[0:3]))
            vertex_trajectory.append(np.copy(positions[0:3]))
            positions_ooc_l = obs[3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num : 
                                                3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
                                                3*self.joint_ooc_l_num]
            positions_ooc_r = obs[3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num +
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num +
                                                3*self.joint_ooc_l_num :
                                                3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num +
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num +
                                                3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num]
            # positions_frontalis = obs[3*self.joint_num + 
            #                                     3*self.joint_rotate_num + 3*self.vertices_num + 
            #                                     3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
            #                                     3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num + 
            #                                     3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num + 3*self.vertices_ooc_r_num + 
            #                                     3*self.joint_frontalis_num : 
            #                                     3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 
            #                                     3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
            #                                     3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num + 
            #                                     3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num + 3*self.vertices_ooc_r_num + 
            #                                     3*self.joint_frontalis_num + 3*self.vertices_frontalis_num]

            targets = obs[3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num : 
                        3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 3*self.vertices_num]
            # targets_jaw = obs[3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 3*self.vertices_num : 
            #                 3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num]
            
            """ save for later """
            targets_ooc_l = obs[3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
                                                3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num  + 
                                                3*self.vertices_ooc_r_num : 3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
                                                3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num  + 
                                                3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num]

            targets_ooc_r = obs[3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num +
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num +
                                                3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num  +
                                                3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num : 3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num +
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num +
                                                3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num
                                                + 3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num  + 3*self.vertices_ooc_r_num]

            # positions_frontalis = obs[3*self.joint_num + 
            #                                     3*self.joint_rotate_num + 3*self.vertices_num + 
            #                                     3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
            #                                     3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num  + 
            #                                     3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num  + 3*self.vertices_ooc_r_num + 3*self.joint_frontalis_num : 3*self.joint_num + 
            #                                     3*self.joint_rotate_num + 3*self.vertices_num + 
            #                                     3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
            #                                     3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num  + 
            #                                     3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num  + 3*self.vertices_ooc_r_num + 3*self.joint_frontalis_num + 3*self.vertices_frontalis_num]  # this is for jaw

            # targets_frontalis = obs[3*self.joint_num + 
            #                                     3*self.joint_rotate_num + 3*self.vertices_num + 
            #                                     3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
            #                                     3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num  + 
            #                                     3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num  + 3*self.vertices_ooc_r_num + 3*self.joint_frontalis_num + 3*self.vertices_frontalis_num : 3*self.joint_num + 
            #                                     3*self.joint_rotate_num + 3*self.vertices_num + 
            #                                     3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
            #                                     3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num  + 
            #                                     3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num  + 3*self.vertices_ooc_r_num + 3*self.joint_frontalis_num  + 3*self.vertices_frontalis_num  + 3*self.vertices_frontalis_num]



            # # 2. Extract joints and vertices dynamically based on the exact offset in `obs`

            # # Base offset (skipping lip/jaw joints and target states)
            offset = 3*self.joint_num + 3*self.joint_rotate_num + 6*self.vertices_num + 6*self.vertices_jaw_num

            # # --- OOC_L Joints ---
            positions_ooc_l_joint = obs[offset : offset + 3*self.joint_ooc_l_num]
            offset += 3*self.joint_ooc_l_num

            # # --- OOC_R Joints ---
            positions_ooc_r_joint = obs[offset : offset + 3*self.joint_ooc_r_num]
            offset += 3*self.joint_ooc_r_num

            # # --- OOC_L Vertices ---
            positions_ooc_l_vertex = obs[offset : offset + 3*self.vertices_ooc_l_num]
            offset += 3*self.vertices_ooc_l_num

            # # --- OOC_R Vertices ---
            positions_ooc_r_vertex = obs[offset : offset + 3*self.vertices_ooc_r_num]
            offset += 3*self.vertices_ooc_r_num

            # # Skip OOC_L and OOC_R target states
            offset += 3*self.vertices_ooc_l_num + 3*self.vertices_ooc_r_num

            # # --- Frontalis ---
            # positions_frontalis_joint = obs[offset : offset + 3*self.joint_frontalis_num]
            # offset += 3*self.joint_frontalis_num

            # positions_frontalis_vertex = obs[offset : offset + 3*self.vertices_frontalis_num]


            # For phase 2, get previous targets
            if phase >= 2:
                targets_previous = payload["next_obs"][3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num : 
                                                    3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 3*self.vertices_num]
                targets_jaw_previous = payload["next_obs"][3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 3*self.vertices_num : 
                                                        3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num]
            # Initialize iteration tracker if it doesn't exist
            if not hasattr(self, 'action_iteration_count'):
                self.action_iteration_count = 0
                self.max_action_iterations = 1e7
            # Compute gradients

            object_name = "oor"
            if self.initial_distance <= 0.001:
                gradient = np.zeros(3 * self.joint_num)
            else:
                # For OOR, positions_lip_joint acts as your active joints
                gradient = self.compute_single_gradient(
                    positions,
                    targets,
                    positions_lip_joint,
                    self.step_t1,
                    object_name,
                )
            self.oor_dense_numerical_gradient = np.copy(self.prev_gradients)

            # RECORD IT HERE!
            self.record_gradient_step("oor", gradient, self.phase_manager.current_phase)
            # RECORD THE MAGNITUDE HERE!
            self.record_gradient_magnitude_step("oor", gradient, self.phase_manager.current_phase)

            rotation_values = 3 * self.joint_rotate_num * [0.0]
            gradient_jaw = np.zeros(3 * self.joint_rotate_num)
            gradient_ooc_r = np.zeros(3 * self.joint_ooc_r_num)
            gradient_frontalis = np.zeros(3 * self.joint_frontalis_num)
            # not yet
            # object_name = "jaw"
            # if self.initial_jaw_distance <= 0.001:
            #     gradient_jaw = np.zeros(len(rotation_values))
            # else:
            #     gradient_jaw = self.compute_analytical_gradient_jaw(rotation_values, targets_jaw, obs)

            # --- 2. OOC_L --- test this
            object_name = "ooc_l"
            expected_ooc_l_joint_values = 3 * self.joint_ooc_l_num
            expected_ooc_l_vertex_values = 3 * self.vertices_ooc_l_num
            if len(positions_ooc_l_joint) != expected_ooc_l_joint_values:
                raise ValueError(
                    f"OOC-L joint slice has {len(positions_ooc_l_joint)} values; "
                    f"expected {expected_ooc_l_joint_values}."
                )
            if (
                len(positions_ooc_l_vertex) != expected_ooc_l_vertex_values
                or len(targets_ooc_l) != expected_ooc_l_vertex_values
            ):
                raise ValueError(
                    "OOC-L observed and target vertex slices must each contain "
                    f"{expected_ooc_l_vertex_values} values; got "
                    f"{len(positions_ooc_l_vertex)} and {len(targets_ooc_l)}."
                )
            if (
                self.analytical_calculator_ooc_l.K != self.joint_ooc_l_num
                or self.analytical_calculator_ooc_l.N != self.vertices_ooc_l_num
            ):
                raise ValueError(
                    "OOC-L analytical calculator dimensions do not match the "
                    "configured joint and vertex counts."
                )
            if self.initial_ooc_l_distance <= 0.001:
                gradient_ooc_l = np.zeros(expected_ooc_l_joint_values)
            else:
                gradient_ooc_l = self.compute_single_gradient(
                    positions_ooc_l_vertex,
                    targets_ooc_l,
                    positions_ooc_l_joint,
                    self.step_t1,
                    object_name,
                )
            if len(gradient_ooc_l) != expected_ooc_l_joint_values:
                raise ValueError(
                    f"OOC-L gradient has {len(gradient_ooc_l)} values; "
                    f"expected {expected_ooc_l_joint_values}."
                )

            # RECORD IT HERE!
            self.record_gradient_step("ooc_l", gradient_ooc_l, self.phase_manager.current_phase)
            # RECORD THE MAGNITUDE HERE!
            self.record_gradient_magnitude_step("ooc_l", gradient_ooc_l, self.phase_manager.current_phase)
            # --- 3. OOC_R --- not yet
            # object_name = "ooc_r"
            # if self.initial_ooc_r_distance <= 0.001:
            #     gradient_ooc_r = np.zeros(len(positions_ooc_r))
            # else:
            #     gradient_ooc_r = self.compute_single_gradient(
            #         joints=positions_ooc_r_joint, 
            #         target=targets_ooc_r, 
            #         initial_distance=self.initial_ooc_r_distance, 
            #         observed_position=positions_ooc_r_vertex, 
            #         calculator=self.analytical_calculator_ooc_r,
            #         object_name=object_name
            #     )
            # # RECORD IT HERE!
            # self.record_gradient_step("ooc_r", gradient_ooc_r, self.phase_manager.current_phase)
            # # RECORD THE MAGNITUDE HERE!
            # self.record_gradient_magnitude_step("ooc_r", gradient_ooc_r, self.phase_manager.current_phase)

            # # --- 4. FRONTALIS ---
            # object_name = "frontalis"
            # if self.initial_frontalis_distance <= 0.001:
            #     gradient_frontalis = np.zeros(len(positions_frontalis))
            # else:
            #     gradient_frontalis = self.compute_single_gradient(
            #         joints=positions_frontalis_joint, 
            #         target=targets_frontalis, 
            #         initial_distance=self.initial_frontalis_distance, 
            #         observed_position=positions_frontalis_vertex, 
            #         calculator=self.analytical_calculator_frontalis,
            #         object_name=object_name
            #     )

            # # RECORD IT HERE!
            # self.record_gradient_step("frontalis", gradient_frontalis, self.phase_manager.current_phase)                
            # # RECORD THE MAGNITUDE HERE!
            # self.record_gradient_magnitude_step("frontalis", gradient_frontalis, self.phase_manager.current_phase)

            self.current_index = (self.current_index + 1) % len(positions)    
            # For phase 2, compute previous gradients
            if phase >= 2 and abc_execution_count < MAX_ABC_EXECUTIONS:
                gradient_previous = self.compute_single_gradient(
                    positions,
                    targets_previous,
                    positions_lip_joint,
                    self.step_t1,
                    "oor",
                )

                gradient_jaw_previous = self.compute_gradient_rotation(rotation_values, targets_jaw_previous)

                action_previous = np.concatenate([
                    gradient_previous,
                    gradient_jaw_previous,
                    gradient_ooc_l,
                    gradient_ooc_r,
                    gradient_frontalis,
                ], axis=0)
            else:
                action_previous = -9999

            action = np.concatenate([
                gradient,
                gradient_jaw,
                gradient_ooc_l,
                gradient_ooc_r,
                gradient_frontalis,
            ], axis=0)
             # Increment the counter
            self.action_iteration_count += 1

            # Only plot every 100 iterations
            if self.action_iteration_count % 100 == 0:
                # Plot gradients
                #object_names = ["oor", "ooc_l","ooc_r", "frontalis"]  # "oor", ooc_l
                object_names = ["oor", "ooc_l"]  # "oor", ooc_l
                for obj_name in object_names:
                    self.plot_gradient_magnitudes(obj_name, self.phase_manager.current_phase)
                    self.plot_gradients_over_iterations(obj_name, self.phase_manager.current_phase)

                # Plot landmark vs vertex trajectory (1st oor landmark ↔ 1st vertex, 1-to-1 mapping)
                self.plot_trajectories(landmark_trajectory, vertex_trajectory, self.phase_manager.current_phase)

            # Prepare parameters for environment step
            base_params = {
                'action': action,                
                # 'phase_2_flag': self.phase_2_flag,
                'current_phase': self.phase_manager.current_phase,  # Changed to use current_phase
                'r_limit_sigmoid_1000_endList': getattr(self, 'r_limit_sigmoid_1000_endList', []),
                't_limit_sigmoid_1000_endList': getattr(self, 't_limit_sigmoid_1000_endList', []),
                'step_counter_internal': step_counter_internal,
                'run_abc': abc_execution_count < MAX_ABC_EXECUTIONS if phase >= 2 else False,
            }

            # Add phase 2 specific parameters
            if phase >= 2:
                base_params.update({
                    'target_state': target_state,
                    'target_jaw_state': target_jaw_state,
                    'x_bar_j': x_bar_j,
                    'x_jaw_bar_j': x_jaw_bar_j,
                    'distance_threshold_percentage': self.distance_threshold_percentage,
                    'distance_jaw_threshold_percentage': self.distance_jaw_threshold_percentage,
                    'distance_min': distance_min,
                    'distance_jaw_min': distance_jaw_min,
                    'expected_total_steps': expected_total_steps,
                    'this_task': this_task,
                    'action_previous': action_previous,
                    'elements': elements,
                    'abc_execution_count': abc_execution_count,
                })

            # Execute environment step

            next_obs, reward, done, next_info, r_limit_sigmoid_1000_endList, t_limit_sigmoid_1000_endList = self.env.unwrapped.step(**base_params)
            self.r_limit_sigmoid_1000_endList = r_limit_sigmoid_1000_endList
            self.t_limit_sigmoid_1000_endList = t_limit_sigmoid_1000_endList
            
            # Update counters and track results
            if base_params['run_abc']:
                abc_execution_count += 1

            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            info = next_info
            self.next_info = next_info
            
            action_real_list.append(next_info["base_action_prime"])
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
                
            reward_list.append(reward)
            distance_list.append(next_info["distance"])
            distance_jaw_list.append(next_info["distance_jaw"])
            step_counter_internal = next_info["step_counter_internal"]

            # Update progress bar
            if self.step_t1 % update_every == 0:
                pbar.set_description(f"Step: {self.step_t1}")
                pbar.update(update_every)            



        # Post-episode processing
        ep_rs_sum = sum(reward_list)
        running_reward = ep_rs_sum

        # self.log_episode_info(self.phase_2_flag, running_reward=running_reward, next_info=next_info)
        self.log_episode_info(self.phase_manager.current_phase, running_reward=running_reward, next_info=next_info)
        
        # Format and log results
        formatted_reward_list = self.format_list(reward_list)
        formatted_distance_list = self.format_list(distance_list)
        formatted_distance_jaw_list = self.format_list(distance_jaw_list)

        if "both" == self.action_mode:
            with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
                print("** evaluator episode: \n accumulated reward:  ", running_reward, 
                    " \n   reward list : \n ", formatted_reward_list, 
                    " \n self.distance_list: \n ", formatted_distance_list, 
                    " \n self.distance_jaw_list: \n ", formatted_distance_jaw_list, 
                    file = f)   

        success_flag_list.append(next_info["success_flag"])

        # Handle success case
        if next_info["success_flag"] == 1:
            print("success in evaluator, distance is:", next_info["distance"])
            final_epoch_action_list = action_real_list
            delta_b0j = final_epoch_action_list
            delta_b0j_conca = delta_b0j
            self.log_action_list(self.phase_manager.current_phase, final_epoch_action_list=final_epoch_action_list)

            delta_b0_conca = [0] * 3*self.joint_num + [0] * 3 * self.joint_rotate_num + [0] * 3 *self.joint_ooc_l_num + [0] * 3 * self.joint_ooc_r_num + [0] * 3 * self.joint_frontalis_num
            
            if self.debug_mode_vsCode == 1:
                for i in range(len(delta_b0j_conca)):
                    delta_b0_conca = [delta_b0_conca[j] + delta_b0j_conca[i][j] for j in range(3*self.joint_num + 3*self.joint_rotate_num + 3 *self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num)]
                
                delta_b0 = [delta_b0_conca[i:i+3] for i in range(0, len(delta_b0_conca), 3)]
                self.log_delta_b0(self.phase_manager.current_phase, delta_b0=delta_b0)

                if "both" == self.action_mode:
                    x_joint_ = self.x_joint_init + delta_b0_conca[0:3*self.joint_num]
                    x_joint_rotation_ = self.x_joint_rotation_init + delta_b0_conca[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]
                    x_ooc_l_joint_ = self.x_ooc_l_joint_init + delta_b0_conca[3*self.joint_num + 3*self.joint_rotate_num : 3*self.joint_num + 3*self.joint_rotate_num + 3*self.joint_ooc_l_num]
                    x_ooc_r_joint_ = self.x_ooc_r_joint_init + delta_b0_conca[3*self.joint_num + 3*self.joint_rotate_num + 3*self.joint_ooc_l_num : 3*self.joint_num + 3*self.joint_rotate_num + 3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num]
                    x_frontalis_joint_ = self.x_frontalis_joint_init + delta_b0_conca[3*self.joint_num + 3*self.joint_rotate_num + 3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num : 3*self.joint_num + 3*self.joint_rotate_num + 3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num + 3*self.joint_frontalis_num]

                self.log_computation_time(self.phase_manager.current_phase)

                # Calculate and log Maya distance
                if phase == 1:
                    target_state_temp = getattr(dataV22, "x_middleres_OrbicularisOris" + str(self.j_func))
                if phase == 2:
                    target_state_temp = getattr(dataV22_phoneme_AA, "x_middleres_OrbicularisOris" + str(self.j_func))
                if phase == 3:
                    target_state_temp = getattr(dataV22_phoneme_T, "x_middleres_OrbicularisOris" + str(self.j_func))
                if phase == 4:
                    target_state_temp = getattr(dataV22_phoneme_IY, "x_middleres_OrbicularisOris" + str(self.j_func))
                if phase == 5:
                    target_state_temp = getattr(dataV22_phoneme_V, "x_middleres_OrbicularisOris" + str(self.j_func))
                if phase == 6:
                    target_state_temp = getattr(dataV22_phoneme_N, "x_middleres_OrbicularisOris" + str(self.j_func))
                if phase == 7:
                    target_state_temp = getattr(dataV22_phoneme_Y, "x_middleres_OrbicularisOris" + str(self.j_func))
                if phase == 8:
                    target_state_temp = getattr(dataV22_phoneme_UW, "x_middleres_OrbicularisOris" + str(self.j_func))

                x_conca = np.asarray([item for sublist in target_state_temp for item in sublist])
                target_state = x_conca
                s_next = next_info['s_next']
                distance = self.distance_calculator.calculate_distance('euclidean', self.control_vertices_slice(target_state, self.vertex_index), s_next)        
                distance_in_MayaFile = distance
                print("distance_in_MayaFile: ", distance_in_MayaFile)
                self.log_maya_distance(self.phase_manager.current_phase, distance_in_MayaFile=distance_in_MayaFile)
                
                print("Game over since hit one success action sequence. Exiting the program early.")
                s_next_full, s_jaw_next_full, s_ooc_l_next_full, s_ooc_r_next_full, s_frontalis_next_full = self.env.suc_position_save()


                # Prepare data for saving
                x_joint_non_concatenate = [x_joint_[i:i+3] for i in range(0, len(x_joint_), 3)]
                x_joint_rotation_non_concatenate = [x_joint_rotation_[i:i+3] for i in range(0, len(x_joint_rotation_), 3)]
                x_ooc_r_joint_non_concatenate = [x_ooc_r_joint_[i:i+3] for i in range(0, len(x_ooc_r_joint_), 3)]
                x_ooc_l_joint_non_concatenate = [x_ooc_l_joint_[i:i+3] for i in range(0, len(x_ooc_l_joint_), 3)]
                x_frontalis_joint_non_concatenate = [x_frontalis_joint_[i:i+3] for i in range(0, len(x_frontalis_joint_), 3)]
                s_next_non_concatenate = [s_next_full[i:i+3] for i in range(0, len(s_next_full), 3)]       
                s_ooc_l_next_non_concatenate = [s_ooc_l_next_full[i:i+3] for i in range(0, len(s_ooc_l_next_full), 3)]         
                s_ooc_r_next_non_concatenate = [s_ooc_r_next_full[i:i+3] for i in range(0, len(s_ooc_r_next_full), 3)]         
                s_frontalis_next_non_concatenate = [s_frontalis_next_full[i:i+3] for i in range(0, len(s_frontalis_next_full), 3)]         
                s_jaw_next_non_concatenate = [s_jaw_next_full[i:i+3] for i in range(0, len(s_jaw_next_full), 3)]            
                
                # Save results
                import pickle
                
                # Usage:
                filename = self.get_phase_filename("deltaB_temp", phase)

                with open(self.save_folder + filename, 'wb') as f:
                    pickle.dump({
                        'delta_b0_conca': delta_b0_conca, 
                        'x_joint_': x_joint_non_concatenate, 
                        'x_ooc_l_joint_': x_ooc_l_joint_non_concatenate, 
                        'x_ooc_r_joint_': x_ooc_r_joint_non_concatenate, 
                        'x_frontalis_joint_': x_frontalis_joint_non_concatenate, 
                        's_next': s_next_non_concatenate, 
                        's_jaw_next': s_jaw_next_non_concatenate,
                        's_ooc_l_next': s_ooc_l_next_non_concatenate,
                        's_ooc_r_next': s_ooc_r_next_non_concatenate,
                        's_frontalis_next': s_frontalis_next_non_concatenate,
                        'x_joint_rotation_': x_joint_rotation_non_concatenate
                    }, f)

                exit_flag = 1
                pickle_package = {
                    'delta_b0_conca': delta_b0_conca, 
                    'x_joint_': x_joint_non_concatenate, 
                    'x_ooc_l_joint_': x_ooc_l_joint_non_concatenate, 
                    'x_ooc_r_joint_': x_ooc_r_joint_non_concatenate,                     
                    'x_frontalis_joint_': x_frontalis_joint_non_concatenate,                     
                    's_next': s_next_non_concatenate,
                    's_jaw_next': s_jaw_next_non_concatenate,
                    's_ooc_l_next': s_ooc_l_next_non_concatenate,
                    's_ooc_r_next': s_ooc_r_next_non_concatenate,                    
                    's_frontalis_next': s_frontalis_next_non_concatenate,                    
                    'x_joint_rotation_': x_joint_rotation_non_concatenate,
                    'next_obs': next_obs, #  if phase == 1 else None
                    'next_info': next_info, #  if phase == 1 else None
                }
                
                self.plot_objective_terms_diff(self.phase_manager.current_phase)
                # self.plot_objective_terms_forward(self.phase_2_flag)
                self.plot_objective_terms_forward(self.phase_manager.current_phase)

                
                return pickle_package # if phase != self.phase_manager.max_phases else exit_flag
        
        # Handle failure case
        print("game is failed, did not find a trajectory")
        # if phase == 1:
        print("aborted at phase:", phase)
        sys.exit(0)  # Exit with status code 0 for phase 1
        return exit_flag  # Return exit_flag for phase 2
    
    
    def control_vertices_slice(self, target_state, vertex_index):
        combined_slices = []
        
        for index in vertex_index:
            slice_part = target_state[index*3:index*3+3]
            combined_slices.extend(slice_part)
        
        return combined_slices    

        
    def format_list(self, lst):
        # Format each number to 3 decimal places and join them with ', '
        return ", ".join([f"{x:.3f}" for x in lst])    

    def log_computation_time(self, current_phase):
        toc = time.time()
        elapsed_minutes = round((toc - self.tic) / 60, 1)
        
        with open(self._get_log_filename(current_phase), 'a') as f:
            print("\n ----- Computation time = " + str(elapsed_minutes) + "m", file=f)

    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    def reward_function(self, current_position, target):
        return -self.euclidean_distance(current_position, target)  # Negative distance as reward



    def translation_reward_function_forward(self, current_position, target, landmark_displacements, landmark_velocities, muscle_forces, strain_distance, distance_remain, w1=1.0, w2=0.1, w3=0.1):

        """
        Objective function that balances distance, strain energy, and muscle work.
        
        Args:
            current_position: Current position of the lip/jaw.
            target: Target position.
            landmark_displacements: Displacements of the landmarks from their rest positions.
            landmark_velocities: Velocities of the landmarks.
            muscle_forces: Forces generated by the muscles.
            w1, w2, w3: Weights for each term.
        
        Returns:
            Objective value.
        """
        # Distance term
        # distance = self.euclidean_distance(current_position, target)

        # obj_full_version = 0
        if self.obj_full_version == 1:        
            # Strain energy term
            strain_energy = 0.5 * self.k_tissue * strain_distance**2 # np.sum(strain_distance**2).  strain_distance = sqrt (^2 +  ^2 + ... )
            # k_tissue = 10000  # Stiffness of soft tissues (N/m)
            # assume the stiffness is same for oor and the softtissue used in elastic energy in jaw rotation.
            
            # Muscle work term
            muscle_work = np.sum(muscle_forces * landmark_displacements)  # For lips
            # now have changed to, muscle work for only one vertex (or landmark).
            # calculate the muscle work done, after completing that step, that landmark movement in this step. so the landmark displacement is the total translation. 
            # and the muscle force is the estimated of the muscle force at this step, but use it to estimate the whole work done by muscle. simplified model.
            # muscle_work = np.sum(muscle_forces * rotation_angle)  # For jaw
            
            # Normalize the distance 
            normalized_distance = distance_remain / (self.initial_distance )  # * self.scaling_factor
            
            # Normalize the accumulated work
            normalized_strain_energy = strain_energy / self.max_strain_energy

            # Normalize the muscle work

            normalized_muscle_work = muscle_work / self.max_muscle_work        

            # Calculate weighted contributions
            distance_contribution = w1 * normalized_distance
            strain_contribution = w2 * normalized_strain_energy
            muscle_contribution = w3 * normalized_muscle_work
            

            # Calculate the objective
            objective = w1 * normalized_distance
                        #w2 * normalized_strain_energy + 
                        #w3 * normalized_muscle_work)
            
        else:
            # scaling_factor = 9.62e-4
            normalized_distance = distance_remain / (self.initial_distance)

            # Strain energy term
            scaling_factor = 9.62e-4 # 9.62e-4
            strain_energy = 0.5 * self.k_tissue * (strain_distance * scaling_factor )**2 # np.sum(strain_distance**2).  strain_distance = sqrt (^2 +  ^2 + ... )
            # k_tissue = 10000  # Stiffness of soft tissues (N/m)
            # assume the stiffness is same for oor and the softtissue used in elastic energy in jaw rotation.
            
            # Normalize the accumulated work
            normalized_strain_energy = strain_energy / self.max_strain_energy


            # Muscle work term
            enlarge_contribution_factor = 10e2
            normalized_muscle_work = np.sum(muscle_forces * landmark_displacements  * (enlarge_contribution_factor * scaling_factor/ self.max_muscle_work))  # For lips
            
            muscle_work = normalized_muscle_work
            # calculate the muscle work done, after completing that step, that landmark movement in this step. so the landmark displacement is the total translation. 
            # and the muscle force is the estimated of the muscle force at this step, but use it to estimate the whole work done by muscle. simplified model.
            # muscle_work = np.sum(muscle_forces * rotation_angle)  # For jaw

            # Normalize the muscle work

            # normalized_muscle_work = muscle_work   # / self.max_muscle_work        

            distance_contribution = w1 * normalized_distance
            strain_contribution = w2 * normalized_strain_energy
            muscle_contribution = w3 * normalized_muscle_work
            

            # In phase 1 evaluation:
            self.log_forward_objective_terms(
                phase=self.phase_manager.current_phase,
                distance_contribution=distance_contribution,
                strain_contribution=strain_contribution,
                muscle_contribution=muscle_contribution,
                muscle_forces=muscle_forces,
                landmark_displacements=landmark_displacements,
                muscle_work=muscle_work,
                enlarge_contribution_factor=enlarge_contribution_factor
            )


            objective = distance_contribution + strain_contribution + muscle_contribution   # may only use several of them.

            scale_factor_objective = self.initial_distance
            scaled_objective = objective * scale_factor_objective
            
        # return objective
        reward = - scaled_objective   # Negative distance as reward

        # Log the values to a file
        log_objective_flag = 0
        if log_objective_flag == 1:
            save_folder = self.save_folder
            with open(os.path.join(save_folder, "translation_reward_function_log.txt"), 'a') as f:
                f.write(f"Step: {self.step_objective}\n")
                f.write(f"Strain Energy: {strain_energy:.4f}\n")
                f.write(f"Normalized Strain Energy: {normalized_strain_energy:.4f}\n")
                f.write(f"Muscle Work: {muscle_work:.4f}\n")
                f.write(f"Normalized Muscle Work: {normalized_muscle_work:.4f}\n")
                f.write(f"Distance Remain: {distance_remain:.4f}\n")
                f.write(f"Normalized Distance: {normalized_distance:.4f}\n")
                f.write(f"Objective: {objective:.4f}\n")
                f.write(f"Reward: {reward:.4f}\n")
                f.write(f"strain_distance: {strain_distance:.4f}\n")
                f.write(f"landmark_displacements: {landmark_displacements:.4f}\n")
                f.write(f"muscle_forces: {muscle_forces:.4f}\n")

                
                f.write("-" * 40 + "\n")  # Separator for readability
            
        return reward, distance_contribution, strain_contribution, muscle_contribution

    def translation_reward_function_backward(self, current_position, target, landmark_displacements, landmark_velocities, muscle_forces, strain_distance, distance_remain, w1=1.0, w2=0.1, w3=0.1):


        """
        Objective function that balances distance, strain energy, and muscle work.
        
        Args:
            current_position: Current position of the lip/jaw.
            target: Target position.
            landmark_displacements: Displacements of the landmarks from their rest positions.
            landmark_velocities: Velocities of the landmarks.
            muscle_forces: Forces generated by the muscles.
            w1, w2, w3: Weights for each term.
        
        Returns:
            Objective value.
        """
        # Distance term
        # distance = self.euclidean_distance(current_position, target)

        # obj_full_version = 0
        if self.obj_full_version == 1:        
            # Strain energy term
            strain_energy = 0.5 * self.k_tissue * strain_distance**2 # np.sum(strain_distance**2).  strain_distance = sqrt (^2 +  ^2 + ... )
            # k_tissue = 10000  # Stiffness of soft tissues (N/m)
            # assume the stiffness is same for oor and the softtissue used in elastic energy in jaw rotation.
            
            # Muscle work term
            muscle_work = np.sum(muscle_forces * landmark_displacements)  # For lips
            # calculate the muscle work done, after completing that step, that landmark movement in this step. so the landmark displacement is the total translation. 
            # and the muscle force is the estimated of the muscle force at this step, but use it to estimate the whole work done by muscle. simplified model.
            # muscle_work = np.sum(muscle_forces * rotation_angle)  # For jaw
            
            # Normalize the distance 
            normalized_distance = distance_remain / (self.initial_distance )  # * self.scaling_factor
            
            
            # Normalize the accumulated work
            normalized_strain_energy = strain_energy / self.max_strain_energy

            # Normalize the muscle work

            normalized_muscle_work = muscle_work / self.max_muscle_work        

            # Calculate weighted contributions
            distance_contribution = w1 * normalized_distance
            strain_contribution = w2 * normalized_strain_energy
            muscle_contribution = w3 * normalized_muscle_work
            

            # Calculate the objective
            objective = w1 * normalized_distance
                        #w2 * normalized_strain_energy + 
                        #w3 * normalized_muscle_work)
            
        else:


            # scaling_factor = 9.62e-4
            normalized_distance = distance_remain / (self.initial_distance)

            # Strain energy term
            scaling_factor = 9.62e-4 # 9.62e-4
            strain_energy = 0.5 * self.k_tissue * (strain_distance * scaling_factor )**2 # np.sum(strain_distance**2).  strain_distance = sqrt (^2 +  ^2 + ... )
            # k_tissue = 10000  # Stiffness of soft tissues (N/m)
            # assume the stiffness is same for oor and the softtissue used in elastic energy in jaw rotation.
            
            # Normalize the accumulated work
            normalized_strain_energy = strain_energy / self.max_strain_energy


            # Muscle work term
            enlarge_contribution_factor = 10e2
            normalized_muscle_work = np.sum(muscle_forces * landmark_displacements  * (enlarge_contribution_factor * scaling_factor/ self.max_muscle_work))  # For lips
            
            
            muscle_work = normalized_muscle_work
            
            # calculate the muscle work done, after completing that step, that landmark movement in this step. so the landmark displacement is the total translation. 
            # and the muscle force is the estimated of the muscle force at this step, but use it to estimate the whole work done by muscle. simplified model.
            # muscle_work = np.sum(muscle_forces * rotation_angle)  # For jaw

            # Normalize the muscle work

            # normalized_muscle_work = muscle_work / self.max_muscle_work        

            distance_contribution = w1 * normalized_distance
            strain_contribution = w2 * normalized_strain_energy
            muscle_contribution = w3 * normalized_muscle_work
  
            trace_and_log_obj_contribution = 1
            # have not maintain this backward logging. 25-6-10. just let it be there and not use it.

            if trace_and_log_obj_contribution == 1:
                # Store contributions for plotting
                if not hasattr(self, 'term_history_backward'):
                    self.term_history_backward = {
                        'steps': [],
                        'distance': [],
                        'strain': [],
                        'muscle': []
                    }
                
                # Log every 200 steps or at convergence
                if (self.step_objective % self.fowrad_gradient_log_interval == 0):
                    
                    self.term_history_backward['steps'].append(self.step_objective)
                    self.term_history_backward['distance'].append(distance_contribution)
                    self.term_history_backward['strain'].append(strain_contribution)
                    self.term_history_backward['muscle'].append(muscle_contribution/enlarge_contribution_factor)
                    
                                
                    # Check if file exists to determine if we need headers
                    file_exists = os.path.exists(os.path.join(self.save_folder, "objective_terms_log_backward.csv"))


            objective = distance_contribution  + strain_contribution  + muscle_contribution # + strain_contribution + muscle_contribution

            scale_factor_objective = self.initial_distance
            scaled_objective = objective * scale_factor_objective
            
        # return objective
        reward = - scaled_objective   # Negative distance as reward

        # Log the values to a file
        log_objective_flag = 0
        if log_objective_flag == 1:
            save_folder = self.save_folder
            with open(os.path.join(save_folder, "translation_reward_function_log.txt"), 'a') as f:
                f.write(f"Step: {self.step_objective}\n")
                f.write(f"Strain Energy: {strain_energy:.4f}\n")
                f.write(f"Normalized Strain Energy: {normalized_strain_energy:.4f}\n")
                f.write(f"Muscle Work: {muscle_work:.4f}\n")
                f.write(f"Normalized Muscle Work: {normalized_muscle_work:.4f}\n")
                f.write(f"Distance Remain: {distance_remain:.4f}\n")
                f.write(f"Normalized Distance: {normalized_distance:.4f}\n")
                f.write(f"Objective: {objective:.4f}\n")
                f.write(f"Reward: {reward:.4f}\n")
                f.write(f"strain_distance: {strain_distance:.4f}\n")
                f.write("-" * 40 + "\n")  # Separator for readability
            
        return reward, distance_contribution, strain_contribution, muscle_contribution


    def rotation_reward_function(self, current_position, target, accumulated_work, w1=1.0, w2=0.1):
        """
        Reward function that balances distance and accumulated work.
        
        Args:
            current_position: Current position of the jaw.
            target: Target position.
            accumulated_work: Total work done by the muscles so far.
            w1: Weight for the distance term.
            w2: Weight for the accumulated work term.
        
        Returns:
            Reward value.
        """
        # Calculate the Euclidean distance
        distance = self.euclidean_distance(current_position, target)
        
        # Normalize the distance (optional)
        normalized_distance = distance / self.initial_distance
        
        # Normalize the accumulated work (optional)
        normalized_work = accumulated_work / self.max_expected_work
        
        # Calculate the reward
        reward = -w1 * normalized_distance - w2 * normalized_work
        
        return reward
    
    def compute_analytical_gradient(self, positions_lip_joint, targets, initial_distance, position, calculator):    
            # note that "positions_lip_joint" can be other joints, (i.e. landmarks) for other muscles
            import torch

            # 1. Convert inputs to PyTorch Tensors
            translations = torch.tensor(positions_lip_joint, dtype=torch.float32).view(-1, 3)
            translations.requires_grad_(True)
            targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 3)
            positions_tensor = torch.tensor(position, dtype=torch.float32).view(-1, 3)

            # 2. Forward pass USING THE PASSED CALCULATOR
            y_e = calculator.maya_lbs(translations, positions_tensor)

            # 3. Compute Distance (geometric vertex distance to target in Maya units)
            # Added a small epsilon (1e-12) to prevent NaN gradients at zero distance
            distance_remain = torch.sqrt(torch.sum((targets_tensor - y_e)**2) + 1e-12)

            # 4. Compute Strain Energy matching refence.py formulation
            scaling_factor = 9.62e-4
            k_tissue = getattr(calculator, 'k_tissue', 10000.0)
            strain_energy = 0.5 * k_tissue * ((distance_remain * scaling_factor) ** 2)

            # 5. Compute Muscle Work matching refence.py formulation
            temp_joint_num = translations.shape[0]
            # enlarge_contribution_factor = 10e2
            enlarge_contribution_factor = 0.1

            # Pull parameters dynamically or fall back to defaults
            max_muscle = getattr(calculator, 'max_muscle_work', 10.0)
            max_strain = getattr(calculator, 'max_strain_energy', 1.0)
            w1 = getattr(calculator, 'w1', self.w1)
            w2 = getattr(calculator, 'w2', self.w2)
            w3 = getattr(calculator, 'w3', self.w3)

            # Physiologically constrained parameters
            F_max_new = 17.0 / 300.0
            expected_total_steps = 5500.0
            v_max_new = 1.0 / ((35.0 / 30.0) / expected_total_steps) * 9.62e-4 * 100.0
            K_new = 0.18
            a_new = 0.12
            b_new = v_max_new * a_new / (F_max_new * K_new)

            activation = 0.8
            original_length = 2.0e-2
            l_0 = 2.0e-2
            sigma = 0.4
            k_PEE = 0.5
            k_SEE = 0.3
            l_tendon = 0.05

            # Compute translations per vertex for the muscle model
            landmark_displacements = torch.sqrt(torch.sum(translations**2, dim=1) + 1e-12) * scaling_factor
            distance_remain_per_vertex = distance_remain / temp_joint_num
            trans_cm = distance_remain_per_vertex * 9.62e-4 * 100.0

            # Epsilon-based velocity conversion from refence.py
            epsilon = 1e-3
            vel = epsilon / ((35.0 / 30.0) / expected_total_steps)
            vel_cm = vel * 9.62e-4 * 100.0

            # Hill Muscle Length Calculation
            l = (original_length * 100.0) + trans_cm
            l_0_cm = l_0 * 100.0
            normalized_length_diff = (l - l_0_cm) / l_0_cm
            exponent = -(normalized_length_diff ** 2) / (2.0 * (sigma ** 2))
            F_length = F_max_new * torch.exp(exponent)

            # Force-velocity Relationship
            if vel_cm >= 0:
                F_velocity = ((F_max_new + a_new) * b_new) / (vel_cm + b_new) - a_new
            else:
                F_velocity = 1.4 - 0.4 * (vel_cm / v_max_new)

            # Passive Forces (torch.clamp prevents negative stretch values)
            F_PEE = k_PEE * torch.clamp(l - l_0_cm, min=0.0)
            F_SEE = k_SEE * l_tendon

            # Total force generation (clamped to physiological maximum)
            active_force = activation * F_length * F_velocity + F_PEE + F_SEE
            muscle_force = torch.clamp(active_force, max=F_max_new)

            # 6. Normalize Components
            init_dist = initial_distance if (initial_distance is not None and initial_distance > 0) else 1.0
            normalized_distance = distance_remain / init_dist
            normalized_strain_energy = strain_energy / (max_strain if max_strain > 0 else 1.0)

            # Sum muscle work over all landmarks utilizing the computed muscle_force tensor
            muscle_work_term = muscle_force * landmark_displacements * (enlarge_contribution_factor * scaling_factor / (max_muscle if max_muscle > 0 else 10.0))
            normalized_muscle_work = torch.sum(muscle_work_term)

            # Calculate final contributions
            distance_contrib = w1 * normalized_distance
            strain_contrib = w2 * normalized_strain_energy
            muscle_contrib = w3 * normalized_muscle_work

            # Store term contributions on calculator for logging
            calculator.dist_grad_mag = distance_contrib.item()
            calculator.strain_grad_mag = strain_contrib.item()
            calculator.muscle_grad_mag = muscle_contrib.item()
            calculator.total_grad_mag = distance_remain.item()

            # 7. Compute weighted scaled loss matching reference.py
            objective = distance_contrib + strain_contrib + muscle_contrib
            scaled_objective = objective * init_dist

            # # Formulate as negative objective for reward maximization
            # reward = -scaled_objective

            # --- THE FIX: GLOBAL SPEED BOOST ---
            # We scaled down the muscle term by ~10,000x to balance the terms, 
            # which shrank the overall gradients. We multiply the final reward 
            # to restore the gradient magnitude the optimizer expects.
            global_speed_boost = 3000.0 

            # Formulate as negative objective for reward maximization
            reward = -scaled_objective * global_speed_boost

            reward.backward()

            # gradients_flat = translations.grad.view(-1).detach().numpy()

            # return gradients_flat * self.analytical_gradient_scale

            # --- Adam Optimizer Implementation ---
            # We store the Adam state on the calculator object to ensure 
            # independent momentum tracking for each muscle/landmark group.

            # NOTE: Adam normalizes gradients, making it invariant to 'global_speed_boost'.
            # The 'adam_lr' here becomes the new primary speed controller.
            # If the optimization is too slow, increase adam_lr (e.g., to 10.0, 50.0, or 100.0).
            adam_lr = 5.0

            if not hasattr(calculator, '_adam_param') or calculator._adam_param.shape != translations.view(-1).shape:
                calculator._adam_param = torch.nn.Parameter(torch.zeros_like(translations.view(-1)))
                calculator._adam_opt = torch.optim.Adam([calculator._adam_param], lr=adam_lr)
            else:
                # Update learning rate in case you change it dynamically
                for param_group in calculator._adam_opt.param_groups:
                    param_group['lr'] = adam_lr

            calculator._adam_opt.zero_grad()

            # Adam minimizes by default, but we want to maximize the reward.
            # Setting grad to -translations.grad makes Adam step towards the maximum.
            calculator._adam_param.grad = -translations.grad.view(-1).clone()

            old_param = calculator._adam_param.data.clone()
            calculator._adam_opt.step()

            # The difference is our Adam-adjusted gradient step
            adam_step = (calculator._adam_param.data - old_param).detach().numpy()

            return adam_step * self.analytical_gradient_scale

    """
    Discovery: In main branch (newest version), the analytical gradient is used whereas here we use a numerical gradient
    I have to use the analytical gradient in the functions below to match the main branch.

    This will become fallback function if the analytical gradient fails to compute.
    """
    def compute_single_gradient(self, position, target, positions_lip_joint, step_t1, 
                                object_name, batch_size=1, epsilon=1e-3, threshold=1e-5, clip_value=1.0):
        # TEST: early return with analytical gradient
        if object_name in ("oor", "ooc_l"):
            if object_name == "oor":
                calculator = self.analytical_calculator
                initial_distance = self.initial_distance
            else:
                calculator = self.analytical_calculator_ooc_l
                initial_distance = self.initial_ooc_l_distance
            calculator.w1 = getattr(self, "w1", 0.34)
            calculator.w2 = getattr(self, "w2", 0.33)
            calculator.w3 = getattr(self, "w3", 0.33)
            calculator.k_tissue = getattr(self, "k_tissue", 10000.0)
            calculator.max_strain_energy = getattr(self, "max_strain_energy", 1.0)
            calculator.max_muscle_work = getattr(self, "max_muscle_work", 1.0)
            calculator.expected_total_steps = getattr(self, "expected_total_steps", 5500)
            calculator.object_name = object_name
            gradient = self.compute_analytical_gradient(
                positions_lip_joint=positions_lip_joint,
                targets=target,
                initial_distance=initial_distance,
                position=position,
                calculator=calculator,
            )

            if object_name == "oor":
                self.prev_gradients = gradient
            return gradient

        # Initialize a cumulative gradient to sum over all steps
        cumulative_gradient = np.zeros_like(position, dtype=np.float32)        
        # Get current phase
        current_phase = self.phase_manager.current_phase
        # Ensure the gradient_magnitudes dictionary has the object_name key
        phase_gradients = self.phase_manager.phase_data[current_phase]['gradient_magnitudes']

        # Initialize the appropriate dictionary based on phase
        
        if object_name not in self.gradient_magnitudes[self.phase_manager.current_phase]:
            self.gradient_magnitudes[self.phase_manager.current_phase][object_name] = []

        # get previous gradient of full 
        def get_full_gradient(position, target, positions_lip_joint, batch_size=1, epsilon=1e-3, threshold=1e-5, clip_value=1.0):
            # position here is the chosen indexed vertices, not landmark's position. but is one-one mapping related.
            gradient = np.zeros_like(position, dtype=np.float32)
            for i in range(len(position)):

                self.step_objective = self.step_objective + 1
                
                pos_forward = np.copy(position)
                pos_backward = np.copy(position)
                pos_forward[i] += epsilon
                pos_backward[i] -= epsilon


                method = 'euclidean'
                scaling_factor = 1 # 9.62e-4  # 1 # 9.62e-4
                """
                assume the velocity is the current step's this current landmark's velocity, not the average from the past.
                final unit: m/s
                scaling_factor = 0.0962  # 1 Maya unit = 0.0962 cm   = 0.000962 m

                originally, strain_distance = ... this equation is to calculate AA to rest process. In this case, target pose is "neutral pose".
                so when the current pose is at the begining status, the strain distance is maximum. and distance_remain = strain_distance.
                at the end, the strain distance is minimum.

                after some exploration, we assume, strain energy at all of the target pose, be it "neutral pose", "smile", or any other pose, is the minimum strain energy.
                this allows the gradient, 3 composnents of the gradient, work together to converge, in a general way.

                Case study 2: from neutral pose to smile. The strain energy, we assume it is very large at the neutral pose. and when it deforms to the target pose,
                the strain energy reaches the minimum. So "distance_remain = strain_distance" still holds.
                """
                strain_distance = self.distance_calculator.calculate_distance(method, target, pos_forward)  * scaling_factor         
                strain_distance_back = self.distance_calculator.calculate_distance(method, target, pos_backward)   * scaling_factor        
                
                distance_remain = strain_distance
                distance_remain_back = strain_distance_back

                scaling_factor_adhoc = 1 # 9.62e-4
                landmark_velocities = epsilon * scaling_factor_adhoc / (  (35/30) /self.expected_total_steps)   #  epsilon per step.  convert to epsilon per s , 

                if self.obj_full_version == 1:   
                    landmark_displacements = self.distance_calculator.calculate_distance(method, self.x_bar_j_selected[i], pos_forward[i])   * scaling_factor         
                    landmark_displacements_back = self.distance_calculator.calculate_distance(method, self.x_bar_j_selected[i], pos_backward[i])  * scaling_factor         
                    scale_factor_reward = 1e8

                else:
                    landmark_displacements= abs( epsilon )  * scaling_factor  
                    landmark_displacements_back= abs( epsilon )  * scaling_factor  
                    scale_factor_reward = 1      

                    
                def calculate_muscle_force(activation, original_length, landmark_translations, landmark_velocities, 
                                        F_max, l_0, sigma, v_max, K, a, b, k_PEE, k_SEE, l_tendon):

                    # Viktor: This was commented out but it may be useful?
                    if self.debug_mode_vsCode == 1:
                        # Debug file setup
                        debug_file = os.path.join(self.save_folder, "muscle_force_debug.txt")
                        
                        with open(debug_file, 'a') as f:
                            f.write(f"\n=== Muscle Force Calculation Debug ===\n")
                            f.write(f"self.step_objective: {self.step_objective}\n")
                            
                            # Original debug info
                            f.write(f"Original landmark_translations (before scaling): {landmark_translations}\n")
                            f.write(f"Type: {type(landmark_translations)}, Shape: {np.array(landmark_translations).shape if hasattr(landmark_translations, 'shape') else 'N/A'}\n")
                            
                            # Convert all units to cm
                            scaling_factor_adhoc = 9.62e-4
                            landmark_translations = np.array(landmark_translations) * scaling_factor_adhoc * 100  
                            landmark_velocities = np.array(landmark_velocities) * scaling_factor_adhoc * 100  
                            original_length = original_length * 100 # m to cm
                            l_0 = l_0 * 100

                            # Debug scaled values
                            f.write(f"Scaled landmark_translations: {landmark_translations}\n")
                            
                            # Try both approaches and compare
                            try:
                                # Original approach with np.mean()
                                avg_with_mean = np.mean(landmark_translations)
                                f.write(f"Average with np.mean(): {avg_with_mean}\n")
                                
                                # Direct approach (assuming scalar)
                                avg_direct = landmark_translations # landmark_translations[0] if len(landmark_translations) == 1 else landmark_translations
                                f.write(f"Average direct access: {avg_direct}\n")
                                
                                # Compare results
                                if np.allclose(avg_with_mean, avg_direct):
                                    f.write("Results MATCH - safe to remove np.mean()\n")
                                    avg_translation = avg_direct  # Use the faster approach
                                else:
                                    f.write("Results DIFFER - keep np.mean()\n")
                                    avg_translation = avg_with_mean
                                    
                            except Exception as e:
                                f.write(f"Error during comparison: {str(e)}\n")
                                avg_translation = np.mean(landmark_translations)  # Fallback to original
                                
                            # Continue with velocity calculation
                            try:
                                avg_velocity = landmark_velocities  # landmark_velocities[0] if len(landmark_velocities) == 1 else np.mean(landmark_velocities)
                                f.write(f"Velocity average: {avg_velocity}\n")
                            except Exception as e:
                                f.write(f"Error in velocity calculation: {str(e)}\n")
                                avg_velocity = np.mean(landmark_velocities)
                                
                            f.write("------------------------\n")
                            

                    # Convert all units to cm
                    scaling_factor_adhoc = 9.62e-4
                    landmark_translations = np.array(landmark_translations) * scaling_factor_adhoc * 100  
                    landmark_velocities = np.array(landmark_velocities) * scaling_factor_adhoc * 100  
                    original_length = original_length * 100 # m to cm
                    l_0 = l_0 * 100
                                            
                    avg_translation = landmark_translations
                    avg_velocity = landmark_velocities

                    # Current muscle length
                    l = original_length + avg_translation

                    # Debug print before F_length calculation
                    # print(f"\n--- Debug ---")
                    # print(f"original_length: {original_length} cm")
                    # print(f"avg_translation: {avg_translation} cm")
                    # print(f"l_0: {l_0} cm")
                    # print(f"Current l: {l} cm")
                    # print(f"Difference (l - l_0): {l - l_0} cm")

                    # Force-Length Relationship
                    normalized_length_diff = (l - l_0) / l_0
                    exponent = -(normalized_length_diff ** 2) / (2 * (sigma ** 2))
                    #     # refer: Zajac, Felix E. "Muscle and tendon: properties, models, scaling, and application to biomechanics and motor control." Critical reviews in biomedical engineering 17.4 (1989): 359-411.
                    #     # PDF page 7 Fig. 5 A. a simplified approximation of the classic sarcomere length-tension curve. The Gaussian form is more common in computational muscle models for smoothness.
                                            
                    F_length = F_max * np.exp(exponent)
                    # F_length: the only term that will affect the gradient difference
                    
                    # print(f"normalized_length_diff: {normalized_length_diff}")
                    # print(f"sigma: {sigma}")
                    # print(f"exponent value: {exponent}")
                    # print(f"exp(exponent): {np.exp(exponent)}")
                    # print(f"F_length: {F_length} N")

                    # Force-Velocity Relationship
                    # assume the Force-Velocity Relationship, we only use the right side of the plot, which is "Eccentric", lengthenting. 
                    # because so far we did not involve any expression or pose that need to squeeze the muscle, or "concentric" such that length will be
                    # smaller than the original length.
                    if avg_velocity >= 0:  # should be Concentric, but we see it as "Eccentric". So we assume the avg_velocity always greater than 0.
                        F_velocity = ((F_max + a) * b) / (avg_velocity + b) - a
                        # source: P161, eq 1. Hill, A. V. (1938). "The heat of shortening and the dynamic constants of muscle." Proceedings of the Royal Society of London. Series B, Biological Sciences, 126(843), 136-195.

                    else:  # Eccentric
                        F_velocity = 1.4 - 0.4 * (avg_velocity / v_max)
                        # source: P161, eq 1. Hill, A. V. (1938). "The heat of shortening and the dynamic constants of muscle." Proceedings of the Royal Society of London. Series B, Biological Sciences, 126(843), 136-195.
                        # we have not considered the negative velocity case, where the lip muscle is shortening. The AA to rest case is when lip "shortens" from a already lengthed pose, not shortening beyond its original length.
                        
                    # Passive forces
                    F_PEE = k_PEE * max(l - l_0, 0)
                    F_SEE = k_SEE * l_tendon
                    
                    # Total force (clamped to not exceed F_max)
                    muscle_force = min(activation * F_length * F_velocity + F_PEE + F_SEE, F_max)

                    log_muscle_force_flag = 0
                    if log_muscle_force_flag == 1:                    
                        save_folder = self.save_folder
                        # Log the important variables
                        if save_folder:
                            with open(os.path.join(save_folder, "muscle_force_log.txt"), 'a') as f:
                                f.write(f"Force-Length (F_length): {F_length:.4f} N\n")
                                f.write(f"Force-Velocity (F_velocity): {F_velocity:.4f}\n")
                                f.write(f"Passive Parallel Elastic Force (F_PEE): {F_PEE:.4f} N\n")
                                f.write(f"Passive Series Elastic Force (F_SEE): {F_SEE:.4f} N\n")
                                f.write(f"Total Muscle Force: {muscle_force:.4f} N\n")
                                f.write("-" * 40 + "\n")  # Separator for readability


                    return muscle_force

                # Inputs

                landmark_translations_per_vertax = landmark_displacements / 1 # self.joint_num
                distance_remain_per_vertax = distance_remain / self.joint_num
                #  landmark_displacements is all landmark translation sum.
                # landmark_translations is each landmark translation value.

                distance_remain_back_per_vertax = distance_remain_back  / self.joint_num
                landmark_translations_back_per_vertax = landmark_displacements_back  / 1 # self.joint_num
                
                # New parameters (physiologically constrained)
                # make the output muscle force below 17 / 300 newton (because I have 300 landmarks, and the whole lip can generate 17 N accourding to literature)
                # 17 N is from: McHenry, Monica A., et al. "Age‐related changes in orofacial force generation in women." The Laryngoscope 109.5 (1999): 827-830.
                # lower 6 N, upper 11 N, (6+11=17 N) for age group 2: 18 - 25 yr arounds, females.
                F_max_new = 17 / 300  # 0.0567 N per landmark
                expected_total_steps = 5500 # empirical value.
                
                # empirical value. set as 1 unit per step.  35 frames, 30 frames per second,   100 is to convert m to cm. 9.62e-4 m = 1 maya unit
                v_max_new = 1 / ((35 / 30) / expected_total_steps) * 9.62e-4 * 100  # ~0.235 cm/s
                
                # Adjusted Hill parameters to keep force < F_max_new
                K_new = 0.18       # Lower K reduces force drop at high velocities. empirical value.
                a_new = 0.12       # Smaller a flattens the force-velocity curve. empirical value.
                b_new = v_max_new * a_new / (F_max_new * K_new)  # Maintains curve shape. empirical value.

                # Common parameters
                activation = 0.8  # degree of activation of muscle. empirical value.  from 0 to 1.
                original_length = 2.0e-2  # m assume                

                
                l_0 = 2.0e-2  # m
                sigma = 0.4  # Narrower length-tension curve  empirical value.
                k_PEE = 0.5  # Reduced passive stiffness  empirical value.
                k_SEE = 0.3    # empirical value.
                l_tendon = 0.05  # Smaller tendon stretch  empirical value.

                if self.obj_full_version == 1:   
                    muscle_forces = calculate_muscle_force(activation, original_length, distance_remain_per_vertax, landmark_velocities)
                    muscle_forces_back = calculate_muscle_force(activation, original_length, distance_remain_back_per_vertax, landmark_velocities)
                # print(f"Computed Muscle Force: {muscle_forces:.2f} N")
                else:
                    # Here is the second strong assumption, coresponding to line 1590. We assume, the big idea is, that the remaining distance = the translated muscle distance.
                    # This assumption is correct in "AA to neutral" task. At AA pose, the lip has maximum muscle force. At neutral pose, the lip has minimum muscle force.
                    # and remaining distance = the translated muscle distance.

                    # This assumption is not correct in "neutral to AA" task. But we assume at neutral pose, the lip has maximum muscle force;
                    # at AA pose, the lip has minimum muscle force. 
                    # and assume remaining distance = the translated muscle distance. So we used the "full distance - translated distance" as the computing "translated distance".
                    # Same reason with line 1590's.

                    # the remaining distance per vertax 
                    trans = distance_remain_per_vertax
                    vel = landmark_velocities
                    muscle_forces = calculate_muscle_force(activation, original_length, trans, vel,
                                                            F_max_new, l_0, sigma, v_max_new, K_new, a_new, b_new, 
                                                            k_PEE, k_SEE, l_tendon
                                                            )

                    # muscle_forces_back = calculate_muscle_force(activation, original_length, distance_remain_back_per_vertax, landmark_velocities)        
                    trans = distance_remain_back_per_vertax
                    muscle_forces_back = calculate_muscle_force(activation, original_length, trans, vel,
                                                            F_max_new, l_0, sigma, v_max_new, K_new, a_new, b_new, 
                                                            k_PEE, k_SEE, l_tendon
                                                            )           

                # Calculate muscle force
                # muscle_force = calculate_muscle_force(activation, original_length, landmark_displacements, landmark_velocities)

                reward_forward, distance_contribution_f, strain_contribution_f, muscle_contribution_f = self.translation_reward_function_forward(pos_forward, target, landmark_translations_per_vertax, landmark_velocities, muscle_forces, strain_distance , distance_remain, w1=self.w1, w2=self.w2, w3=self.w3)
                reward_backward, distance_contribution_b, strain_contribution_b, muscle_contribution_b  = self.translation_reward_function_backward(pos_backward, target, landmark_translations_back_per_vertax, landmark_velocities, muscle_forces_back, strain_distance_back, distance_remain_back, w1=self.w1, w2=self.w2, w3=self.w3)
                # reward_backward = self.reward_function(pos_backward, target)

                reward_diff = reward_forward - reward_backward

                reward_diff = reward_diff * scale_factor_reward

                scale_factor_objective = self.initial_distance

                reward_diff_distance_contribution = ( distance_contribution_f * scale_factor_objective - distance_contribution_b * scale_factor_objective  ) *  scale_factor_reward
                reward_diff_strain_contribution = ( strain_contribution_f * scale_factor_objective - strain_contribution_b * scale_factor_objective  ) *  scale_factor_reward
                reward_diff_muscle_contribution = ( muscle_contribution_f * scale_factor_objective - muscle_contribution_b * scale_factor_objective  ) *  scale_factor_reward                
            

                self.log_objective_terms(self.phase_manager.current_phase, reward_diff=reward_diff, 
                       reward_diff_distance_contribution=reward_diff_distance_contribution,
                       reward_diff_strain_contribution=reward_diff_strain_contribution,
                       reward_diff_muscle_contribution=reward_diff_muscle_contribution,
                       muscle_contribution_f=muscle_contribution_f,
                       muscle_contribution_b=muscle_contribution_b)
                # reward_diff = round(reward_forward - reward_backward, 4) 
                if abs(reward_diff) < threshold:
                    gradient[i] = 0.0
                else:
                    gradient[i] = reward_diff / (2 * epsilon)
                
                # Clip gradient to avoid large updates
                
                gradient[i] = np.clip(gradient[i], -clip_value, clip_value)
                
            # Reshape the gradient to have shape (number_of_landmarks, 3)
            num_landmarks = len(position) // 3
            gradient = gradient.reshape((num_landmarks, 3))

            # Flatten the gradient array
            flattened_gradient = gradient.flatten() 

            return flattened_gradient           
        
        
        self.prev_gradients = get_full_gradient(position, target, positions_lip_joint)        
        
        num_params = len(position)
        self.num_params = num_params
        gradient = np.zeros_like(position, dtype=np.float32)

        def select_indices(batch_size):
            # Record the gradients
            
            # Calculate indices to avoid
            avoid_indices = set(self.recent_indices)
            
            # Choose indices based on exploration probability
            if np.random.rand() < self.exploration_prob:
                # Exploration: Select random indices not recently chosen
                available_indices = list(set(range(self.num_params)) - avoid_indices)
                selected_indices = np.random.choice(available_indices, batch_size, replace=False)
            else:
                # Exploitation: Choose largest gradients, avoiding recent ones
                sorted_indices = np.argsort(np.abs(self.prev_gradients))
                selected_indices = [idx for idx in sorted_indices if idx not in avoid_indices][-batch_size:]
            
            # Update recent indices history
            self.recent_indices.extend(selected_indices)
            if len(self.recent_indices) > self.history_size:
                self.recent_indices = self.recent_indices[-self.history_size:]
            
            return selected_indices 

        selected_indices = select_indices(batch_size)       


        for idx in selected_indices:
            gradient[idx] = self.prev_gradients[idx]
        
        num_landmarks = len(position) // 3
        gradient = gradient.reshape((num_landmarks, 3))

        # Flatten the gradient array
        flattened_gradient = gradient.flatten()

        # Add the current step's gradient to the cumulative gradient
        cumulative_gradient += flattened_gradient
    
        # Optionally, print or log the cumulative gradient to analyze later
        # with open(self.save_folder + "/log" + str(self.i_func) + ".txt", 'a') as f:
        #     print(f"Cumulative Gradient after step {step_t1}: {cumulative_gradient}", file=f)

        # Calculate the magnitude of the cumulative gradient to check for cancellations
        gradient_magnitude = np.linalg.norm(cumulative_gradient)
        # print(f"Gradient Magnitude: {gradient_magnitude}")
        
        # Store the gradient magnitude with respect to step_t1 in the specific object's list
        # self.gradient_magnitudes_dict[object_name].append((step_t1, gradient_magnitude))      
        
        # if self.phase_2_flag == 1:            
        #     self.gradient_magnitudes_dict_phase2[object_name].append((step_t1, gradient_magnitude))            
        # else:
        #     self.gradient_magnitudes_dict[object_name].append((step_t1, gradient_magnitude))            
        
        # Store in the current phase's dictionary
        if object_name not in self.gradient_magnitudes[self.phase_manager.current_phase]:
            self.gradient_magnitudes[self.phase_manager.current_phase][object_name] = []
            
        self.gradient_magnitudes[self.phase_manager.current_phase][object_name].append(
            (step_t1, gradient_magnitude)
        )

        self.gradients_over_iterations.append(flattened_gradient[idx])
        
        return flattened_gradient

    # adam
    def compute_gradient_rotation(self, rotation_values, target, epsilon=1e-3, threshold=1e-4):
        gradient = np.zeros_like(rotation_values, dtype=np.float32)

        for i in range(len(rotation_values)):
            rot_forward = np.copy(rotation_values)
            rot_backward = np.copy(rotation_values)

            rot_forward[i] += epsilon
            rot_backward[i] -= epsilon


            # s_next_full, s_jaw_next_full, s_ooc_l_next_full, s_ooc_r_next_full, s_frontalis_next_full = self.env.suc_position_save()
            # if self.phase_manager.current_phase == 3:
            #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
            #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
            #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
            #     method = 'euclidean'
            #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_full, self.vertex_jaw_index))
            #     print("e1 line 2741, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)

            new_coords_jaw_forward, accumulated_work_at_last_forward = self.env.return_coordinates_adhoc(rot_forward, self.phase_manager.current_phase)

            # s_next_full, s_jaw_next_full, s_ooc_l_next_full, s_ooc_r_next_full, s_frontalis_next_full = self.env.suc_position_save()
            # if self.phase_manager.current_phase == 3:
            #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
            #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
            #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
            #     method = 'euclidean'
            #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_full, self.vertex_jaw_index))
            #     print("e1 line 2751, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)

            new_coords_jaw_backward, accumulated_work_at_last_backward = self.env.return_coordinates_adhoc(rot_backward, self.phase_manager.current_phase)

            # s_next_full, s_jaw_next_full, s_ooc_l_next_full, s_ooc_r_next_full, s_frontalis_next_full = self.env.suc_position_save()
            # if self.phase_manager.current_phase == 3:
            #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
            #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
            #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
            #     method = 'euclidean'
            #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_full, self.vertex_jaw_index))
            #     print("e1 line 2761, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)


            if self.add_energy_to_jaw_objective_flag == 1:
                reward_forward = self.rotation_reward_function(new_coords_jaw_forward, target, accumulated_work_at_last_forward, w1=1.0, w2=0.1)
                reward_backward = self.rotation_reward_function(new_coords_jaw_backward, target, accumulated_work_at_last_backward, w1=1.0, w2=0.1)
            else:
                reward_forward = self.reward_function(new_coords_jaw_forward, target)
                reward_backward = self.reward_function(new_coords_jaw_backward, target)
                
            reward_diff = reward_forward - reward_backward
            if abs(reward_diff) < threshold:
                gradient[i] = 0.0
            else:
                gradient[i] = reward_diff / (2 * epsilon)

        return gradient.flatten()
    
    def plot_gradient_magnitudes(self, object_name, current_phase=1):
        """Plot gradient magnitudes with professional academic style.
        
        Args:
            object_name (str): Name of the object to plot
            current_phase (int): Current phase number (1, 2, 3, etc.)
        """
        # Initialize phase storage if it doesn't exist
        if current_phase not in self.gradient_magnitudes:
            self.gradient_magnitudes[current_phase] = {}
        
        # Check if object_name exists in the dictionary
        if object_name not in self.gradient_magnitudes[current_phase]:
            print(f"No gradient data available for object {object_name} in phase {current_phase}")
            return
        
        # Get the gradient data
        gradient_data = self.gradient_magnitudes[current_phase][object_name]
        
        if not gradient_data:
            print(f"No gradient data points available for {object_name} in phase {current_phase}")
            return
            
        # Rest of the plotting code remains the same...
        steps, magnitudes = zip(*gradient_data)        
        
        # Create plot with enhanced styling
        plt.plot(steps, magnitudes, 
                marker='o', 
                markersize=5,
                markeredgewidth=0.5,
                markerfacecolor='none',
                color='#1f77b4',  # Professional blue
                linewidth=1.5,
                linestyle='-',
                )

        # Axis labels with increased font size
        plt.xlabel('Optimization step', fontsize=14)
        plt.ylabel('Gradient magnitude', fontsize=14)
        
        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Configure tick parameters
        ax.tick_params(axis='both', which='major', 
                    direction='out', 
                    length=6, 
                    width=1.2,
                    labelsize=12)

        # Add gridlines
        plt.grid(True, linestyle='--', alpha=0.4)
        # Determine filename based on phase
        filename = f'Gradient_Magnitudes_{object_name}'
        if current_phase > 1:  # Only add phase suffix if not phase 1 (the default)
            filename += f'_phase{current_phase}'
        filename += '.png'

        # Save with tight layout and high DPI
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder, filename), 
                bbox_inches='tight', 
                dpi=300,
                transparent=False)
        plt.close()

    def plot_gradients_over_iterations(self, object_name, current_phase=1, indices=None):
        """Plot the chosen flattened_gradient coordinates over optimization steps.
        
        Args:
            object_name (str): Name of the object to plot
            current_phase (int): Current phase number (1, 2, 3, etc.)
            indices (list, optional): List of specific coordinate indices of the flattened gradient to plot.
                                      If None, auto-selects top 5 indices with the highest variance.
        """
        # Initialize dictionary if not exist
        if not hasattr(self, 'gradients_over_iterations_dict'):
            self.gradients_over_iterations_dict = {}
            
        if current_phase not in self.gradients_over_iterations_dict:
            self.gradients_over_iterations_dict[current_phase] = {}
            
        if object_name not in self.gradients_over_iterations_dict[current_phase]:
            print(f"No gradient history available to plot for {object_name} in phase {current_phase}")
            return
            
        data = self.gradients_over_iterations_dict[current_phase][object_name]
        if not data:
            print(f"Empty gradient history for {object_name} in phase {current_phase}")
            return
            
        grads = np.array(data)  # shape: (num_steps, num_params)

        num_steps, num_params = grads.shape

        # Automatically generate step indices (0, 1, 2, ..., num_steps - 1)
        steps = np.arange(num_steps)

        if indices is None:
            # Dynamically choose the top-5 coordinates with the highest variance
            # so we visualize the most active parameters/dimensions
            variances = np.var(grads, axis=0)
            num_to_pick = min(5, num_params)
            indices = np.argsort(variances)[::-1][:num_to_pick].tolist()
            
        # Set up professional plotting style
        plt.figure(figsize=(9, 6), dpi=300)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2

        # Plot each selected coordinate
        for idx in indices:
            if idx < 0 or idx >= num_params:
                print(f"Warning: index {idx} is out of bounds for gradient of size {num_params}")
                continue
            plt.plot(steps, grads[:, idx], 
                     linewidth=1.5,
                     label=f'Coordinate {idx}')

        # Plot the mean absolute gradient as a background reference
        mean_abs_grad = np.mean(np.abs(grads), axis=1)
        plt.plot(steps, mean_abs_grad,
                 color='black',
                 linewidth=2.0,
                 linestyle='--',
                 alpha=0.6,
                 label='Mean Absolute Gradient')

        # Axis labels with increased font size
        plt.xlabel('Optimization step', fontsize=14)
        plt.ylabel('Gradient Value', fontsize=14)
        
        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Configure tick parameters
        ax.tick_params(axis='both', which='major', 
                    direction='out', 
                    length=6, 
                    width=1.2,
                    labelsize=12)

        # Add gridlines
        plt.grid(True, linestyle='--', alpha=0.4)

        plt.legend(frameon=True, loc='best', fontsize=10)

        # Determine filename based on phase
        filename = f'Gradients_Over_Iterations_{object_name}'
        if current_phase > 1:
            filename += f'_phase{current_phase}'
        filename += '.png'

        # Save with tight layout and high DPI
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder, filename), 
                bbox_inches='tight', 
                dpi=300,
                transparent=False)
        plt.close()

    def plot_trajectories(self, landmark_trajectory, vertex_trajectory, current_phase=1):
        """Plot the 3D and 2D trajectories of the first landmark and vertex.
        
        Args:
            landmark_trajectory (list): List of 3D coordinates for the landmark.
            vertex_trajectory (list): List of 3D coordinates for the vertex.
            current_phase (int): The active phase.
        """
        if not landmark_trajectory or not vertex_trajectory:
            print("No trajectory data to plot.")
            return

        # Local import to ensure 3D plotting is supported
        from mpl_toolkits.mplot3d import Axes3D

        landmark_trajectory = np.array(landmark_trajectory) # shape: (T, 3)
        vertex_trajectory = np.array(vertex_trajectory) # shape: (T, 3)
        steps = np.arange(len(landmark_trajectory))

        # Set up professional plotting style
        fig = plt.figure(figsize=(14, 6), dpi=300)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 11

        # Panel 1: 3D Trajectory
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot trajectories
        ax1.plot(landmark_trajectory[:, 0], landmark_trajectory[:, 1], landmark_trajectory[:, 2],
                 color='#1f77b4', linewidth=2, label='1st Landmark')
        ax1.plot(vertex_trajectory[:, 0], vertex_trajectory[:, 1], vertex_trajectory[:, 2],
                 color='#ff7f0e', linewidth=2, linestyle='--', label='1st Vertex')

        # Highlight start and end points
        ax1.scatter(landmark_trajectory[0, 0], landmark_trajectory[0, 1], landmark_trajectory[0, 2],
                    color='green', marker='o', s=50, label='Start')
        ax1.scatter(landmark_trajectory[-1, 0], landmark_trajectory[-1, 1], landmark_trajectory[-1, 2],
                    color='red', marker='x', s=50, label='End')
        ax1.scatter(vertex_trajectory[0, 0], vertex_trajectory[0, 1], vertex_trajectory[0, 2],
                    color='green', marker='o', s=50)
        ax1.scatter(vertex_trajectory[-1, 0], vertex_trajectory[-1, 1], vertex_trajectory[-1, 2],
                    color='red', marker='x', s=50)

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Trajectory Comparison')
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Panel 2: 2D Coordinates over steps
        ax2 = fig.add_subplot(122)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = ['X', 'Y', 'Z']
        
        for i in range(3):
            ax2.plot(steps, landmark_trajectory[:, i], 
                     color=colors[i], linewidth=1.5, label=f'Landmark {labels[i]}')
            ax2.plot(steps, vertex_trajectory[:, i], 
                     color=colors[i], linewidth=1.5, linestyle='--', label=f'Vertex {labels[i]}')

        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Coordinate Value')
        ax2.set_title('Coordinate Alignment Over Steps')
        ax2.legend(loc='best', ncol=2)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Remove top and right spines on 2D plot
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Determine filename based on phase
        filename = 'Trajectory_Comparison'
        if current_phase > 1:
            filename += f'_phase{current_phase}'
        filename += '.png'

        plt.suptitle(f'Movement Trajectory Analysis (Phase {current_phase})', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder, filename), 
                    bbox_inches='tight', 
                    dpi=300,
                    transparent=False)
        plt.close()

    def plot_objective_terms_forward(self, current_phase=1):
        """Plot forward gradient objective terms with error handling"""
        data = self._get_term_history(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase > 1 else ""
        
        # Check if we have data to plot
        if not data['term_history_forward'].get('steps'):
            print(f"Warning: No forward term history data for phase {current_phase}")
            return

        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

        try:
            # Get all data with safety checks
            steps = data['term_history_forward'].get('steps', [])
            distances = data['term_history_forward'].get('distance', [])
            strains = data['term_history_forward'].get('strain', [])
            muscles = data['term_history_forward'].get('muscle', [])
            
            # Ensure all arrays have same length
            min_length = min(len(steps), len(distances), len(strains), len(muscles))
            if min_length == 0:
                print("No data available for plotting")
                return
                
            steps = steps[:min_length]
            distances = distances[:min_length]
            strains = strains[:min_length]
            muscles = muscles[:min_length]

            # Sampling logic
            total_points = len(steps)
            if total_points > 20:
                sample_indices = np.linspace(0, total_points-1, 20, dtype=int)
                sampled_steps = [steps[i] for i in sample_indices]
                sampled_distance = [distances[i] for i in sample_indices]
                sampled_strain = [strains[i] for i in sample_indices]
                sampled_muscle = [muscles[i] for i in sample_indices]
                use_markers = True
            else:
                sampled_steps = steps
                sampled_distance = distances
                sampled_strain = strains
                sampled_muscle = muscles
                use_markers = False
        
            # Plot configuration
            plot_kwargs = {'linewidth': 2, 'linestyle': '-'}
            if use_markers:
                plot_kwargs.update({'marker': 'o', 'markersize': 5})

            # Plot each term
            plt.plot(sampled_steps, sampled_distance, 
                    label=f'Distance (w1={self.w1:.3f})', 
                    color='blue', **plot_kwargs)
            plt.plot(sampled_steps, sampled_strain, 
                    label=f'Strain Energy (w2={self.w2:.3f})', 
                    color='green', **plot_kwargs)
            plt.plot(sampled_steps, sampled_muscle, 
                    label=f'Muscle Work (w3={self.w3:.3f})', 
                    color='red', **plot_kwargs)
            
            # Mark convergence if applicable
            if hasattr(self, 'step_objective'):
                conv_step = self.step_objective
                if conv_step in sampled_steps:
                    conv_idx = sampled_steps.index(conv_step)
                    plt.scatter(conv_step, sampled_distance[conv_idx], 
                            color='black', marker='o', s=100, zorder=5,
                            label=f'Convergence (step {conv_step})')

            # Styling
            plt.xlabel('Optimization Steps', fontsize=14)
            plt.ylabel('Term Contribution Value\nForward Gradient', fontsize=14)
            plt.gca().spines[['top', 'right']].set_visible(False)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

            # Save phase-specific plot
            plt.savefig(os.path.join(self.save_folder, f'objective_terms_plot_forward{phase_suffix}.png'), 
                    bbox_inches='tight', dpi=300)
            plt.close()
        except KeyError as e:
            print(f"Error accessing plotting data: {e}")
        except Exception as e:
            print(f"Unexpected error during plotting: {e}")


    def plot_objective_terms_backward(self, current_phase=1):
        """Plot backward gradient objective terms"""
        data = self._get_term_history(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase > 1 else ""
        
        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

        # Sampling logic
        total_points = len(data['term_history_backward']['steps'])
        if total_points > 20:
            sample_indices = np.linspace(0, total_points-1, 20, dtype=int)
            sampled_steps = [data['term_history_backward']['steps'][i] for i in sample_indices]
            sampled_distance = [data['term_history_backward']['distance'][i] for i in sample_indices]
            sampled_strain = [data['term_history_backward']['strain'][i] for i in sample_indices]
            sampled_muscle = [data['term_history_backward']['muscle'][i] for i in sample_indices]
            use_markers = True
        else:
            sampled_steps = data['term_history_backward']['steps']
            sampled_distance = data['term_history_backward']['distance']
            sampled_strain = data['term_history_backward']['strain']
            sampled_muscle = data['term_history_backward']['muscle']
            use_markers = False

        # Plot configuration
        plot_kwargs = {'linewidth': 2, 'linestyle': '-'}
        if use_markers:
            plot_kwargs.update({'marker': 'o', 'markersize': 5})

        # Plot each term
        plt.plot(sampled_steps, sampled_distance, 
                label=f'Distance (w1={self.w1:.3f})', 
                color='blue', **plot_kwargs)
        plt.plot(sampled_steps, sampled_strain, 
                label=f'Strain Energy (w2={self.w2:.3f})', 
                color='green', **plot_kwargs)
        plt.plot(sampled_steps, sampled_muscle, 
                label=f'Muscle Work (w3={self.w3:.3f})', 
                color='red', **plot_kwargs)
        
        # Mark convergence if applicable
        if hasattr(self, 'step_objective'):
            conv_step = self.step_objective
            if conv_step in sampled_steps:
                conv_idx = sampled_steps.index(conv_step)
                plt.scatter(conv_step, sampled_distance[conv_idx], 
                        color='black', marker='o', s=100, zorder=5,
                        label=f'Convergence (step {conv_step})')

        # Styling
        plt.xlabel('Optimization Steps', fontsize=14)
        plt.ylabel('Term Contribution Value\nBackward Gradient', fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

        # Save phase-specific plot
        plt.savefig(os.path.join(self.save_folder, f'objective_terms_plot_backward{phase_suffix}.png'), 
                bbox_inches='tight', dpi=300)
        plt.close()


    def plot_objective_terms_diff(self, current_phase=1):
        """Plot difference in objective terms"""
        data = self._get_term_history(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase > 1 else ""
                
        # Verify lengths match
        lengths = {
            'steps': len(data['term_history_diff']['steps']),
            'distance': len(data['term_history_diff']['distance']),
            'strain': len(data['term_history_diff']['strain']),
            'muscle': len(data['term_history_diff']['muscle'])
        }

        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

        # Get the data to plot
        steps = data['term_history_diff']['steps']
        distances = data['term_history_diff']['distance']
        strains = data['term_history_diff']['strain']
        muscles = data['term_history_diff']['muscle']
        # Add type verification
        # Sampling logic
        total_points = len(data['term_history_diff']['steps'])
        if total_points > 20:
            sample_indices = np.linspace(0, total_points-1, 20, dtype=int)
            sampled_steps = [steps[i] for i in sample_indices]
            sampled_distance = [distances[i] for i in sample_indices]
            sampled_strain = [strains[i] for i in sample_indices]
            sampled_muscle = [muscles[i] for i in sample_indices]
            use_markers = True
        else:
            sampled_steps = steps
            sampled_distance = distances
            sampled_strain = strains
            sampled_muscle = muscles
            use_markers = False

        # Plot configuration
        plot_kwargs = {'linewidth': 2, 'linestyle': '-'}
        if use_markers:
            plot_kwargs.update({'marker': 'o', 'markersize': 5})

        # Plot each term
        plt.plot(sampled_steps, sampled_distance, 
                label=f'Distance (w1={self.w1:.3f})', 
                color='blue', **plot_kwargs)
        plt.plot(sampled_steps, sampled_strain, 
                label=f'Strain Energy (w2={self.w2:.3f})', 
                color='green', **plot_kwargs)
        plt.plot(sampled_steps, sampled_muscle, 
                label=f'Muscle Work (w3={self.w3:.3f})', 
                color='red', **plot_kwargs)
        
        # Mark convergence if applicable
        if hasattr(self, 'step_objective'):
            conv_step = self.step_objective
            if conv_step in sampled_steps:
                conv_idx = sampled_steps.index(conv_step)
                plt.scatter(conv_step, sampled_distance[conv_idx], 
                        color='black', marker='o', s=100, zorder=5,
                        label=f'Convergence (step {conv_step})')

        # Styling
        plt.xlabel('Optimization Steps', fontsize=14)
        plt.ylabel('Gradient Term Contribution Value\nDifference Value', fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

        # Save phase-specific plot
        # plt.savefig(os.path.join(self.save_folder, f'objective_terms_plot_diff{phase_suffix}.png'), 
        #         bbox_inches='tight', dpi=300)
        # plt.close()
        plt.savefig(os.path.join(self.save_folder, 
                f'objective_terms_plot_diff{phase_suffix}.png'), 
                bbox_inches='tight', dpi=300)
        plt.close()


    def _get_term_history(self, current_phase):
        """Unified term history accessor with guaranteed structure"""
        if current_phase not in self.term_histories:
            # print(f"Initializing term history for phase {current_phase}")
            self.term_histories[current_phase] = {
                'forward': {
                    'steps': [],
                    'distance': [],
                    'strain': [],
                    'muscle': []
                },
                'diff': {
                    'steps': [],
                    'distance': [],
                    'strain': [],
                    'muscle': []
                },
                'count': {'value': 0}
            }
        
        # Verify structure exists
        if 'steps' not in self.term_histories[current_phase]['forward']:
            self.term_histories[current_phase]['forward']['steps'] = []

        return {
            'term_history_forward': self.term_histories[current_phase]['forward'],
            'term_history_diff': self.term_histories[current_phase]['diff'],
            'count_diff_log': self.term_histories[current_phase]['count']['value']
        }    

    def log_objective_terms(self, current_phase=1, reward_diff=None, 
                        reward_diff_distance_contribution=None,
                        reward_diff_strain_contribution=None,
                        reward_diff_muscle_contribution=None,
                        muscle_contribution_f=None,
                        muscle_contribution_b=None):
        
        # Get phase-specific data containers
        data = self._get_term_history(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase > 1 else ""
        
        trace_and_log_obj_contribution = 1
        if trace_and_log_obj_contribution == 1:
            # Initialize term history diff if it doesn't exist
            if 'steps' not in data['term_history_diff']:
                data['term_history_diff'].update({
                    'steps': [],
                    'distance': [],
                    'strain': [],
                    'muscle': []
                })
            
            check_interval = math.floor(self.count_diff_log_interval)  # 1500 default
            threshold = 1e-6  # Define your threshold here

            count = data['count_diff_log'].get('value', 0) if isinstance(data['count_diff_log'], dict) else data['count_diff_log']
            if abs(reward_diff) >= threshold:
                if count % check_interval == 0:            
            # if abs(reward_diff) >= threshold and count % check_interval == 0:
                # Store the data
                    data['term_history_diff']['steps'].append(self.step_objective)
                    data['term_history_diff']['distance'].append(reward_diff_distance_contribution)
                    data['term_history_diff']['strain'].append(reward_diff_strain_contribution)
                    data['term_history_diff']['muscle'].append(reward_diff_muscle_contribution)
                    
                    # Write to phase-specific CSV file
                    filename = f"objective_terms_data_diff{phase_suffix}.csv"
                    file_exists = os.path.exists(os.path.join(self.save_folder, filename))
                    try:
                        with open(os.path.join(self.save_folder, filename), 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            
                            if not file_exists:
                                writer.writerow(["Step", "Distance Contribution", 
                                            "Strain Contribution", "Muscle Contribution",
                                            "muscle_contribution_f", "muscle_contribution_b"])
                            
                            writer.writerow([
                                self.step_objective,
                                f"{reward_diff_distance_contribution:.6f}",
                                f"{reward_diff_strain_contribution:.6f}", 
                                f"{reward_diff_muscle_contribution:.6f}",
                                f"{muscle_contribution_f:.6f}",
                                f"{muscle_contribution_b:.6f}"
                            ])
                    except Exception as e:
                        print(f"Error writing to CSV: {e}")

                self.term_histories[current_phase]['count']['value'] = count + 1

    def log_forward_objective_terms(self, phase=None, 
                                distance_contribution=None,
                                strain_contribution=None,
                                muscle_contribution=None,
                                muscle_forces=None,
                                landmark_displacements=None,
                                muscle_work=None,
                                enlarge_contribution_factor=1.0):
    #     """Log forward objective terms for any phase"""
    #     if phase is None:
    #         phase = self.phase_manager.current_phase
        
    #     data = self._get_term_history(phase)

    # def log_forward_objective_terms(self, phase=None, ...):
        if phase is None:
            phase = self.phase_manager.current_phase
        
        # print(f"\n=== DEBUG: Logging for phase {phase} ===")
        # print(f"Current term_histories structure: {self.term_histories}")
        
        # # Force initialization if needed
        # if phase not in self.term_histories:
        #     self._get_term_history(phase)



        data = self._get_term_history(phase)
        # print(f"Returned data structure: {data}")
        
        # Ensure lists exist
        for key in ['steps', 'distance', 'strain', 'muscle']:
            if key not in data['term_history_forward']:
                data['term_history_forward'][key] = []

        phase_suffix = f"_phase{phase}" if phase > 1 else ""
        
        # fowrad_gradient_log_interval = 4e3
        if self.step_objective % self.fowrad_gradient_log_interval == 0:
            try:
                data['term_history_forward']['steps'].append(self.step_objective)
                data['term_history_forward']['distance'].append(distance_contribution)
                data['term_history_forward']['strain'].append(strain_contribution)
                data['term_history_forward']['muscle'].append(muscle_contribution/enlarge_contribution_factor)
            except Exception as e:
                print(f"Error logging forward terms for phase {phase}: {str(e)}")
                print(f"Current data structure: {data}")

            # Write to CSV
            filename = f"objective_terms_data_forward{phase_suffix}.csv"
            file_exists = os.path.exists(os.path.join(self.save_folder, filename))

            with open(os.path.join(self.save_folder, filename), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                if not file_exists:
                    writer.writerow(["Step", "Distance Contribution", 
                                "Strain Contribution", "Muscle Contribution",
                                "muscle_forces", "landmark_displacements", 
                                "muscle_work"])
                
                writer.writerow([
                    self.step_objective,
                    f"{distance_contribution:.6f}",
                    f"{strain_contribution:.6f}", 
                    f"{muscle_contribution:.6f}",
                    f"{muscle_forces:.12f}",
                    f"{landmark_displacements:.12f}",
                    f"{muscle_work:.12f}"
                ])

    def _get_log_filename(self, current_phase):
        """Helper function to get phase-specific log filename"""
        phase_suffix = f"_phase{current_phase}" if current_phase > 1 else ""
        return f"{self.save_folder}/log{str(self.i_func)}{phase_suffix}.txt"
    
    def log_episode_info(self, current_phase=1, running_reward=None, next_info=None):
        """Log episode information"""
        with open(self._get_log_filename(current_phase), 'a') as f:
            print(f"** evaluator phase {current_phase} episode: reward: {running_reward}, done at No. {next_info['step_counter_internal']} step, success flag {next_info['success_flag']}", file=f)

    def log_delta_b0(self, current_phase=1, delta_b0=None):
        """Log delta_b0 values"""
        pass  # Implementation remains the same, just uses current_phase

    def log_maya_distance(self, current_phase=1, distance_in_MayaFile=None):
        """Log Maya distance information"""
        with open(self._get_log_filename(current_phase), 'a') as f:
            print(f"\n\ndistance_in_MayaFile: {distance_in_MayaFile}\n\n", file=f)    
                                                    
    def log_action_list(self, current_phase=1, final_epoch_action_list=None):
        """Log action list information"""
        with open(self._get_log_filename(current_phase), 'a') as f:
            print(f"final_epoch_action_list length: {len(final_epoch_action_list)}", file=f)

        phase_suffix = f"_phase{current_phase}" if current_phase > 1 else ""
        filename = f"final_epoch_action_list{phase_suffix}.pkl"
        with open(os.path.join(self.save_folder, filename), 'wb') as f:
            pickle.dump({'final_epoch_action_list': final_epoch_action_list}, 
                    f, protocol=pickle.HIGHEST_PROTOCOL)                                                    
            
    def _get_phase_params(self, phase, payload=None):
        """Returns parameters specific to each phase with proper validation"""
        # Base params common to all phases
        base_params = {
            'action': self.current_action,
            'phase_flag': phase,
            'step_counter': self.step_counter_internal,
            'run_abc': False,
            'r_limit_sigmoid_1000_endList': getattr(self, 'r_limit_sigmoid_1000_endList', None),
            't_limit_sigmoid_1000_endList': getattr(self, 't_limit_sigmoid_1000_endList', None)
        }
        
        params = {}
            
        if phase >= 2 and payload is not None:
            # Prepare phase 2 parameters
            params.update({
                'target_state': payload.get('target_state'),
                'target_jaw_state': payload.get('target_jaw_state'),
                'x_bar_j': payload.get('x_bar_j'),
                'x_jaw_bar_j': payload.get('x_jaw_bar_j'),
                'distance_threshold_percentage': self.distance_threshold_percentage,
                'distance_jaw_threshold_percentage': self.distance_jaw_threshold_percentage,
                'distance_min': payload.get('distance_min'),
                'distance_jaw_min': payload.get('distance_jaw_min'),
                'expected_total_steps': payload.get('expected_total_steps'),
                'this_task': payload.get('this_task'),
                'action_previous': payload.get('action_previous'),
                'elements': payload.get('elements')
            })

            # Only include ABC params if applicable
            if hasattr(self, 'abc_execution_count'):
                params['run_abc'] = (self.abc_execution_count < self.MAX_ABC_EXECUTIONS)
            
            base_params.update(params)
        
        # Add payload from previous phase if available
        if payload and phase > 1:
            base_params['previous_payload'] = payload
        
        return base_params
    
    def get_phase_filename(self, base_name, phase):
        """Generate phase-specific filenames"""
        if phase == 1:
            return f"/{base_name}.pkl"
        return f"/{base_name}_phase{phase}.pkl"
    def record_gradient_step(self, object_name, gradient_array, current_phase):
        """Saves the flat gradient array at the current iteration for plotting later."""
        if not hasattr(self, 'gradients_over_iterations_dict'):
            self.gradients_over_iterations_dict = {}
            
        if current_phase not in self.gradients_over_iterations_dict:
            self.gradients_over_iterations_dict[current_phase] = {}
            
        if object_name not in self.gradients_over_iterations_dict[current_phase]:
            self.gradients_over_iterations_dict[current_phase][object_name] = []
            
        # Append a copy of the flat numpy array gradient
        self.gradients_over_iterations_dict[current_phase][object_name].append(np.copy(gradient_array))

    def record_gradient_magnitude_step(self, object_name, gradient_array, current_phase):
        """Computes the L2 norm (magnitude) of the gradient and saves it with the current step."""
        if not hasattr(self, 'gradient_magnitudes'):
            self.gradient_magnitudes = {}
            
        if current_phase not in self.gradient_magnitudes:
            self.gradient_magnitudes[current_phase] = {}
            
        if object_name not in self.gradient_magnitudes[current_phase]:
            self.gradient_magnitudes[current_phase][object_name] = []
            
        # Calculate the magnitude (L2 norm) of the gradient array
        magnitude = float(np.linalg.norm(gradient_array))
        
        # Save as a tuple of (step, magnitude) so zip(*gradient_data) can unpack it safely
        current_step = self.action_iteration_count
        self.gradient_magnitudes[current_phase][object_name].append((current_step, magnitude))
        
def create_evaluator(phase_manager=None, **kwargs):
    evaluator = Evaluator(phase_manager=phase_manager, **kwargs)
    print("Create evaluator successfully!")
    return evaluator
