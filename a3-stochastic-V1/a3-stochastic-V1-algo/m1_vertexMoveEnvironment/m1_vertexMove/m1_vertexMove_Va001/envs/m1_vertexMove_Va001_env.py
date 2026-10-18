__credits__ = ["Carlos Luis"]
import os
from typing import Optional
from os import path
import sys
import numpy as np

# line 2335 is the convergence line, or search this: "if distance_jaw <= self.distance_jaw_min + epsilon_min"

# NumPy 2.0+ compatibility patch
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_  # bool_ is the modern equivalent


import pygame
from pygame import gfxdraw

import gym
from gym import spaces
from gym.utils import seeding


import math
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
from matplotlib.colors import Normalize  # Import Normalize from matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation 
# from IPython import display
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
import math


from scipy.spatial import cKDTree
from scipy.integrate import cumulative_trapezoid
import datetime
import csv

import importlib
from paths import a3_stochastic_relative_path

sys.path.insert(0, str(a3_stochastic_relative_path / "a1-blendshape-guided-animation-pipeline"/ "database"))
sys.path.insert(0, str(a3_stochastic_relative_path /"b98-sampling-joint"/ "300joints_suite_good_old"))
sys.path.insert(0, r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub")

import dataV22
importlib.reload(dataV22)

import dataV22_phoneme_T
importlib.reload(dataV22_phoneme_T)

import pickle



import joint_position
import joint_position_first_wave
import joint_position_third_wave
importlib.reload(joint_position)
importlib.reload(joint_position_first_wave)
importlib.reload(joint_position_third_wave)

# class RewardFunction:
#     def __init__(self, distance_threshold, distance_jaw_threshold, initial_w_dist=5, initial_w_dist_jaw=5, max_history_size=1000):
#         self.distance_threshold = distance_threshold
#         self.distance_jaw_threshold = distance_jaw_threshold
#         self.w_dist = initial_w_dist
#         self.w_dist_jaw = initial_w_dist_jaw
#         self.distance_history = []
#         self.angle_history = []
#         self.initial_w_dist = initial_w_dist
#         self.max_history_size = max_history_size

    
#     def compute_combined_reward(self,
#             dist_to_target, 
#             dist_to_target_jaw, 
#             dist_to_target_ooc_l, 
#             dist_to_target_ooc_r, 
#             # w_direction_translation,
#             # reward_direction_translation,
#             # w_differential_translation,
#             # reward_differential_translation,
#             # w_differential_rotation,
#             # reward_differential_rotation,
#             fix_flag_ship,
#             action_mode
#         ):

#         self.action_mode = action_mode
     
#         self.fix_flag_ship = fix_flag_ship
        
#         # Define maximum weights
#         MAX_W_DIST = 10.0  # Set a reasonable maximum value for w_dist
#         MAX_W_DIST_JAW = 3.0  # Set a reasonable maximum value for w_dist_jaw

#         if "both" == self.action_mode:

#             # Dynamic Weight Adjustment
#             if len(self.distance_history) > 1:
#                 if self.distance_history[-1] > self.distance_history[-2]:
#                     self.w_dist = min(self.w_dist * (1 + 0.0001), MAX_W_DIST)  # Increase the weight for distance if not improving, capped at MAX_W_DIST
#                 elif self.distance_history[-1] == self.distance_history[-2]:
#                     self.w_dist *= 1.0  # No change if distance is the same
#                 else:
#                     self.w_dist = max(self.w_dist * (1 - 0.0001), self.initial_w_dist)  # Decrease the weight but not below the initial

#             if len(self.angle_history) > 1:
#                 if self.angle_history[-1] > self.angle_history[-2]:
#                     self.w_dist_jaw = min(self.w_dist_jaw * (1 + 0.0001), MAX_W_DIST_JAW)  # Increase the weight for angle if not improving, capped at MAX_W_DIST_JAW
#                 elif self.angle_history[-1] == self.angle_history[-2]:
#                     self.w_dist_jaw *= 1.0  # No change if angle is the same
#                 else:
#                     self.w_dist_jaw = max(self.w_dist_jaw * (1 - 0.0001), self.initial_w_dist)  # Decrease the weight but not below the initial


#         self.distance_history.append(dist_to_target)
#         if len(self.distance_history) > self.max_history_size:
#             self.distance_history.pop(0)

#         if dist_to_target < self.distance_threshold:
#             reward_dist = 50  # Large bonus for reaching the threshold
#         else:
#             reward_dist = self.w_dist * math.exp(-dist_to_target**2)


#         self.angle_history.append(dist_to_target_jaw)
#         if len(self.angle_history) > self.max_history_size:
#             self.angle_history.pop(0)

#         # Calculate angle component
#         if dist_to_target_jaw < self.distance_jaw_threshold:
#             reward_dist_jaw = 50  # Bonus for achieving the angle threshold
#         else:
#             reward_dist_jaw = self.w_dist_jaw * math.exp(-dist_to_target_jaw**2)

class RewardFunction:
    def __init__(self, distance_threshold, distance_jaw_threshold, 
                 distance_ooc_l_threshold, distance_ooc_r_threshold, distance_frontalis_threshold, initial_w_dist=5, initial_w_dist_jaw=5, 
                 initial_w_dist_ooc_l=3, initial_w_dist_ooc_r=3, initial_w_dist_frontalis=3, max_history_size=1000):
        self.distance_threshold = distance_threshold
        self.distance_jaw_threshold = distance_jaw_threshold
        self.distance_ooc_l_threshold = distance_ooc_l_threshold
        self.distance_ooc_r_threshold = distance_ooc_r_threshold
        self.distance_frontalis_threshold = distance_frontalis_threshold
        self.w_dist = initial_w_dist
        self.w_dist_jaw = initial_w_dist_jaw
        self.w_dist_ooc_l = initial_w_dist_ooc_l
        self.w_dist_ooc_r = initial_w_dist_ooc_r
        self.w_dist_frontalis = initial_w_dist_frontalis
        self.distance_history = []
        self.angle_history = []
        self.ooc_l_history = []
        self.ooc_r_history = []
        self.frontalis_history = []

        self.initial_w_dist = initial_w_dist
        self.max_history_size = max_history_size

    def compute_combined_reward(self,
            dist_to_target, 
            dist_to_target_jaw, 
            dist_to_target_ooc_l, 
            dist_to_target_ooc_r, 
            dist_to_target_frontalis, 
            fix_flag_ship,
            action_mode
        ):

        self.action_mode = action_mode
        self.fix_flag_ship = fix_flag_ship

        # Define maximum weights
        MAX_W_DIST = 10.0  # Set a reasonable maximum value for w_dist
        MAX_W_DIST_JAW = 3.0  # Set a reasonable maximum value for w_dist_jaw
        MAX_W_DIST_OOC = 5.0  # Set a maximum value for w_dist_ooc_l and w_dist_ooc_r
        MAX_W_DIST_FRONTALIS = 5.0  # Set a maximum value for w_dist_ooc_l and w_dist_ooc_r

        if "both" == self.action_mode:

            # Dynamic Weight Adjustment for dist_to_target
            if len(self.distance_history) > 1:
                if self.distance_history[-1] > self.distance_history[-2]:
                    self.w_dist = min(self.w_dist * (1 + 0.0001), MAX_W_DIST)
                elif self.distance_history[-1] == self.distance_history[-2]:
                    self.w_dist *= 1.0
                else:
                    self.w_dist = max(self.w_dist * (1 - 0.0001), self.initial_w_dist)

            # Dynamic Weight Adjustment for dist_to_target_jaw
            if len(self.angle_history) > 1:
                if self.angle_history[-1] > self.angle_history[-2]:
                    self.w_dist_jaw = min(self.w_dist_jaw * (1 + 0.0001), MAX_W_DIST_JAW)
                elif self.angle_history[-1] == self.angle_history[-2]:
                    self.w_dist_jaw *= 1.0
                else:
                    self.w_dist_jaw = max(self.w_dist_jaw * (1 - 0.0001), self.initial_w_dist)

            # Dynamic Weight Adjustment for dist_to_target_ooc_l
            if len(self.ooc_l_history) > 1:
                if self.ooc_l_history[-1] > self.ooc_l_history[-2]:
                    self.w_dist_ooc_l = min(self.w_dist_ooc_l * (1 + 0.0001), MAX_W_DIST_OOC)
                elif self.ooc_l_history[-1] == self.ooc_l_history[-2]:
                    self.w_dist_ooc_l *= 1.0
                else:
                    self.w_dist_ooc_l = max(self.w_dist_ooc_l * (1 - 0.0001), self.initial_w_dist)

            # Dynamic Weight Adjustment for dist_to_target_ooc_r
            if len(self.ooc_r_history) > 1:
                if self.ooc_r_history[-1] > self.ooc_r_history[-2]:
                    self.w_dist_ooc_r = min(self.w_dist_ooc_r * (1 + 0.0001), MAX_W_DIST_OOC)
                elif self.ooc_r_history[-1] == self.ooc_r_history[-2]:
                    self.w_dist_ooc_r *= 1.0
                else:
                    self.w_dist_ooc_r = max(self.w_dist_ooc_r * (1 - 0.0001), self.initial_w_dist)

            # Dynamic Weight Adjustment for dist_to_target_frontalis
            if len(self.frontalis_history) > 1:
                if self.frontalis_history[-1] > self.frontalis_history[-2]:
                    self.w_dist_frontalis = min(self.w_dist_frontalis * (1 + 0.0001), MAX_W_DIST_FRONTALIS)
                elif self.frontalis_history[-1] == self.frontalis_history[-2]:
                    self.w_dist_frontalis *= 1.0
                else:
                    self.w_dist_frontalis = max(self.w_dist_frontalis * (1 - 0.0001), self.initial_w_dist)

        # Update histories
        self.distance_history.append(dist_to_target)
        if len(self.distance_history) > self.max_history_size:
            self.distance_history.pop(0)

        self.angle_history.append(dist_to_target_jaw)
        if len(self.angle_history) > self.max_history_size:
            self.angle_history.pop(0)

        self.ooc_l_history.append(dist_to_target_ooc_l)
        if len(self.ooc_l_history) > self.max_history_size:
            self.ooc_l_history.pop(0)

        self.ooc_r_history.append(dist_to_target_ooc_r)
        if len(self.ooc_r_history) > self.max_history_size:
            self.ooc_r_history.pop(0)

        self.frontalis_history.append(dist_to_target_frontalis)
        if len(self.frontalis_history) > self.max_history_size:
            self.frontalis_history.pop(0)

        # Calculate rewards
        epsilon_min = 0.01
        if dist_to_target < self.distance_threshold + epsilon_min:
            reward_dist = 50
        else:
            reward_dist = self.w_dist * math.exp(-dist_to_target**2)

        if dist_to_target_jaw < self.distance_jaw_threshold + epsilon_min:
            reward_dist_jaw = 50
        else:
            reward_dist_jaw = self.w_dist_jaw * math.exp(-dist_to_target_jaw**2)

        if dist_to_target_ooc_l < self.distance_ooc_l_threshold + epsilon_min:
            reward_dist_ooc_l = 50
        else:
            reward_dist_ooc_l = self.w_dist_ooc_l * math.exp(-dist_to_target_ooc_l**2)

        if dist_to_target_ooc_r < self.distance_ooc_r_threshold + epsilon_min:
            reward_dist_ooc_r = 50
        else:
            reward_dist_ooc_r = self.w_dist_ooc_r * math.exp(-dist_to_target_ooc_r**2)

        if dist_to_target_frontalis < self.distance_frontalis_threshold + epsilon_min:
            reward_dist_frontalis = 50
        else:
            reward_dist_frontalis = self.w_dist_frontalis * math.exp(-dist_to_target_frontalis**2)

        # Combine rewards
        # total_reward = reward_dist + reward_dist_jaw + reward_dist_ooc_l + reward_dist_ooc_r
        # return total_reward

 

        reward = 0

        # addition_distance_jaw = np.tanh(1 - dist_to_target_jaw/self.distance_jaw_threshold) * 1 # 10 #  50
        
        # addition_distance = np.tanh(1 - dist_to_target/self.distance_threshold) * 1 # 10 #  50
        

        reward_jaw_part = reward_dist_jaw # + \
        # w_differential_rotation * reward_differential_rotation  + addition_distance_jaw     

        reward_lip_part = reward_dist  #  + \
        # w_direction_translation * reward_direction_translation + w_differential_translation * reward_differential_translation  + addition_distance     

        reward_ooc_l_part = reward_dist_ooc_l
        reward_ooc_r_part = reward_dist_ooc_r
        reward_frontalis_part = reward_dist_frontalis

        if "both" == self.action_mode:

            if self.fix_flag_ship[0] == 0: # if lip joint is not locked.
                reward = reward + reward_lip_part
            else:
                # lip joint is locked. give it's 50 reward.
                reward = reward + 50




            if self.fix_flag_ship[1] == 0:
                reward = reward + reward_jaw_part
            else:
                # jaw joint is locked. give it's 50 reward.
                reward = reward + 50

            if self.fix_flag_ship[2] == 0:
                reward = reward + reward_ooc_l_part
            else:
                # jaw joint is locked. give it's 50 reward.
                reward = reward + 50

            if self.fix_flag_ship[3] == 0:
                reward = reward + reward_ooc_r_part
            else:
                # jaw joint is locked. give it's 50 reward.
                reward = reward + 50                

            if self.fix_flag_ship[4] == 0:
                reward = reward + reward_frontalis_part
            else:
                # jaw joint is locked. give it's 50 reward.
                reward = reward + 50       

        # if "translation" == self.action_mode:
        #     reward = reward + reward_lip_part

        # if "rotation" == self.action_mode:
        #     reward = reward + reward_jaw_part

        reward = np.clip(reward, -1e3, 1e3)
        
        return reward, self.w_dist, self.w_dist_jaw, self.w_dist_ooc_l, self.w_dist_ooc_r, self.w_dist_frontalis, self.fix_flag_ship # , addition_distance, addition_distance_jaw


class MeshDistanceCalculator:
    def __init__(self):
        pass

    def euclidean_distance(self, point1, point2):
        # Convert points to numpy arrays
        point1 = np.array(point1)
        point2 = np.array(point2)
        # print("point1", point1)
        # print("point2", point2)
        # Reshape points if necessary
        # if point1.ndim == 1:
        vertices_num = int(len(point1) / 3)
        point1 = point1.reshape((vertices_num, 3))
        point2 = point2.reshape((vertices_num, 3))

        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((point1 - point2)**2))

        return distance

    # def euclidean_distance(self, point1, point2):
    #     vertices_num = int(len(point1) / 3)
    #     point1 = np.reshape(point1, (vertices_num, 3))
    #     point2 = np.reshape(point2, (vertices_num, 3))

    #     distance = np.linalg.norm(point1 - point2)

    #     # distance_ship = np.sqrt((point1 - point2) ** 2)

    #     # distance_ship_vertices = np.linalg.norm(point1 - point2, axis=1)

    #     return distance  # , distance_ship, distance_ship_vertices

    def chamfer_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        tree1 = cKDTree(point1)
        tree2 = cKDTree(point2)

        distances1, _ = tree1.query(point2)
        distances2, _ = tree2.query(point1)

        chamfer_dist = np.mean(distances1) + np.mean(distances2)
        return chamfer_dist

    def bulldozer_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        tree1 = cKDTree(point1)
        distances, _ = tree1.query(point2)
        bulldozer_dist = np.max(distances)
        return bulldozer_dist

    def hausdorff_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        tree1 = cKDTree(point1)
        tree2 = cKDTree(point2)

        distances1, _ = tree1.query(point2)
        distances2, _ = tree2.query(point1)

        hausdorff_dist = max(np.max(distances1), np.max(distances2))
        return hausdorff_dist

    def rms_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        squared_diff = np.sum((point1 - point2) ** 2, axis=1)
        rms_dist = np.sqrt(np.mean(squared_diff))
        return rms_dist

    def closest_point_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        tree1 = cKDTree(point1)
        tree2 = cKDTree(point2)

        distances1, _ = tree1.query(point2)
        distances2, _ = tree2.query(point1)

        closest_dist1 = np.mean(distances1)
        closest_dist2 = np.mean(distances2)
        return closest_dist1, closest_dist2

    def calculate_distance(self, method, point1, point2):
        if method == 'euclidean':
            return self.euclidean_distance(point1, point2)
        elif method == 'chamfer':
            return self.chamfer_distance(point1, point2)
        elif method == 'bulldozer':
            return self.bulldozer_distance(point1, point2)
        elif method == 'hausdorff':
            return self.hausdorff_distance(point1, point2)
        elif method == 'rms':
            return self.rms_distance(point1, point2)
        elif method == 'closest':
            return self.closest_point_distance(point1, point2)
        else:
            raise ValueError("Unknown distance method: {}".format(method))

class m1_vertexMove_Va001(gym.Env):
    """
       ### Description
       to add.

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, g=10.0, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)        

        # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
        #     print(" in env init, self.x_bar_j is  ", self.control_vertices_slice(self.x_bar_j, self.vertex_index), file = f)

        # Add these variables to your class initialization (__init__ method)
        # self.iteration_count = 0
        # self.max_iterations = 200
        # self.coordinate_differences = []  # To store the differences for each iteration

        self.point1_list = []
        self.point_jaw_list = []
        self.norm_distances_list = []
        self.distance_ship_list = []
        self.previous_cbar = None
        self.expected_total_steps_initial = self.expected_total_steps
        self.remaining_steps = 10e6
        self.distance_calculator = MeshDistanceCalculator()
        self.reward_function = RewardFunction(distance_threshold=self.distance_min, distance_jaw_threshold=self.distance_jaw_min, distance_ooc_l_threshold=self.distance_ooc_l_min, distance_ooc_r_threshold=self.distance_ooc_r_min, distance_frontalis_threshold=self.distance_frontalis_min)
        
        self.step_count = 0
        self.max_steps = 10  
        self.plot_counter = 0           
        self.plot_debug_every = 1000
        
        self.start_time = None

        self.time_limit = float('inf')  # This sets the time limit to positive infinity


        # Initialize lists to store distances and timestamps

        self.phase_logs = {
            # Each phase will store its own set of logs
            1: {  # Phase 1
                'distance': [],
                'step_counter_internal': [],
                'time_for_energy': [],
                'rotational_kinetic_energy': [],
                'time_for_rotational_kinetic_energy': [],
                'elastic_energy': [],
                'rotation_velocity': [],
                'base_action_prime': [],
                'angles': [],  # Replaces accumulated_rotated_angle
                'action_real_list': [],
                'base_action_prime_log': [],
            },
            # Phase 2 will be added dynamically when needed
        }
        # self.action_real_list = []

        # self.elastic_energy = []        

        # self.distance_log = []
        # self.distance_log_phase2 = []  # For phase 2
        # # self.time_log = []
        # self.time_for_energy_log = []

        # self.strain_energy = []
        # self.rotation_velocity = []
        # self.rotational_kinetic_energy = []
        # self.time_for_rotational_kinetic_energy_log = []

        # self.step_counter_internal_log = []
        # self.step_counter_internal_log_phase2 = []  # For phase 2
        # self.accumulated_rotated_angle = []

        # Initialize phase angle storage
        # replace this " self.accumulated_rotated_angle_phase2"
        self.phase_angle_storage = {
            1: [],  # Phase 1 angles
            2: [],  # Phase 2 angles
            # Additional phases can be added dynamically
        }

        # Initialize a persistent accumulator as a NumPy array
        self._delta_b0_conca_accumulator = np.zeros(
            3 * self.joint_num + 
            3 * self.joint_rotate_num + 
            3 * self.joint_ooc_l_num + 
            3 * self.joint_ooc_r_num + 
            3 * self.joint_frontalis_num,
            dtype=np.float32  # Use float32 if precision permits
        )        

        # self.action_real_list = []

        # self.elastic_energy = []

        # # Existing logs
        # self.distance_log = []
        # self.step_counter_internal_log = []
        # self.strain_energy = []
        # self.time_for_energy_log = []
        # self.rotational_kinetic_energy = []
        # self.time_for_rotational_kinetic_energy_log = []
        # self.elastic_energy = []
        # self.rotation_velocity = []
        # self.accumulated_rotated_angle = []
        
        # # Phase 2 logs
        # self.distance_log_phase2 = []
        # self.step_counter_internal_log_phase2 = []
        # self.strain_energy_phase2 = []
        # self.time_for_energy_log_phase2 = []
        # self.rotational_kinetic_energy_phase2 = []
        # self.time_for_rotational_kinetic_energy_log_phase2 = []
        # self.elastic_energy_phase2 = []
        # self.rotation_velocity_phase2 = []
        # self.accumulated_rotated_angle_phase2 = []

        # self.base_action_prime_log = []
        # self.base_action_prime_log_phase2 = []

        #BK1

        # Start the timer
        # start_time = time.time()

        # print('l90')
        if self.debug_mode_vsCode == 0:
            import maya.cmds as cmds
            import maya.OpenMaya as OpenMaya        
        
        if self.debug_mode_vsCode == 1:
            import maya.standalone
            maya.standalone.initialize()    
            import maya.cmds as cmds
            import maya.OpenMaya as OpenMaya    

            
            from PyQt5.QtCore import Qt, QCoreApplication
            from PyQt5.QtWidgets import QApplication, QLabel        
            
            # Set the attribute before creating the QApplication
            QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

            app = QApplication(sys.argv)

            # cmds.loadPlugin("Mayatomr") # Load all plugins you might need

            # Have a namespace if you want (recommended)
            # self.namespace = "ref" 

            # full path to your Maya file to OPEN
            # self.maya_file_to_open = r"C:\keyLabLocal_D\c99-maya-ziva-workshop\experiment_files\g10-10 rest pose 1200 landmarks v9.mb"
            # self.maya_file_to_open

            if not os.path.exists(self.maya_file_to_open):
                raise RuntimeError(f"File not found: {self.maya_file_to_open}")


            # Check for unsaved changes and handle them
            if cmds.file(query=True, modified=True):
                # Save or discard changes as needed
                print("Unsaved changes detected.")
                # save_changes = input("You have unsaved changes. Do you want to save them before opening the new file? (yes/no): ").strip().lower()
                if 1==0:  # save_changes in ['yes', 'y']:
                    cmds.file(save=True)
                    print("Changes saved.")
                else:
                    cmds.file(newFile=True, force=True)  # Discard changes

            # Open the Maya file
            try:
                self.opened_file = cmds.file(self.maya_file_to_open, open=True, force=True)
                print("Opened file:", self.opened_file)


                cmds.playbackOptions(min=1, max=50, animationStartTime=1, animationEndTime=50)

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

            except Exception as e:
                print(f"Failed to open Maya file: {e}")

            solver = cmds.ziva(s=True)[0]
            ziva_nodes = cmds.ls(type='zSolverTransform')
            print(f"Ziva solver nodes: {ziva_nodes}")        


            # self.maya_file_to_open = maya_file_to_open
            # # cmds.file(maya_file_to_open, r=True )
            # # Open your file
            # self.opened_file = cmds.file(self.maya_file_to_open, r=True, ns=self.namespace)
            # # print("self.opened_file", self.opened_file)
            

        self.cmds = cmds
        self.OpenMaya = OpenMaya

        # self.deform_rest_to_previous_pose()



        self.max_speed = 8
        self.max_torque = self.UNIT_PRIME # 0.001 # 0.0005 # 0.5 # 2.0
        self.max_torque_rotate = self.phase_params['rotation_anatomy_limit'][1] # 0.1 # 0.03 # 0.1 # 0.001 # 0.0005 # 0.5 # 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.screen = None
        self.clock = None
        self.isopen = True

        self.screen_dim = 500
        self.print_allow = 0
        self.step_counter_internal = 0
        # print("env l146 _init_ internal step ", self.step_counter_internal)
        # self.fix_flag_ship = [0] * self.joint_num * 3 + [0] * self.joint_rotate_num * 3
        self.i_func = kwargs["i_func"]
        self.j_func = kwargs["j_func"]
        self.fix_r_limit_flag = 0
        self.fixed_r_limit_sigmoid = 0
        self.x_bar_j_selected = self.control_vertices_slice(self.x_bar_j, self.vertex_index)
        self.abc_execution_count = 0
        self.MAX_ABC_EXECUTIONS = self.overlapping_length        

        # debug use 240527, only take one vertices.
        # self.fix_flag_ship = [0] * self.joint_num * 3 + [0] * self.joint_rotate_num * 3
        self.fix_flag_ship = [0] * (self.joint_num + self.joint_rotate_num  + self.joint_ooc_l_num + self.joint_ooc_r_num  + self.joint_frontalis_num )
        # only the self.fix_flag_ship[0:4] is really used in this code. not the whole array.
        secondary_boundary = 500
        # observation space:
        high = np.array([secondary_boundary] * (3*self.joint_num + 3*self.joint_rotate_num + 3*self.vertices_num + 
                                                3*self.vertices_jaw_num + 3*self.vertices_num + 3*self.vertices_jaw_num + 
                                                3*self.joint_ooc_l_num  + 3*self.joint_ooc_r_num + 3*self.vertices_ooc_l_num  + 
                                                3*self.vertices_ooc_r_num + 3*self.vertices_ooc_l_num + 3*self.vertices_ooc_r_num  + 3*self.joint_frontalis_num  + 3*self.vertices_frontalis_num  + 3*self.vertices_frontalis_num), dtype=np.float32)
        print("high size: \n \n", high.size)
        # action space
        high_torque = np.array([self.max_torque] * 3*self.joint_num + [self.max_torque_rotate] *3*self.joint_rotate_num  + [self.max_torque] * 3*self.joint_ooc_l_num  +  [self.max_torque] * 3*self.joint_ooc_r_num +  [self.max_torque] * 3*self.joint_frontalis_num, dtype=np.float32)
        print("high_torque size: \n \n", high_torque.size)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-high_torque, high=high_torque, shape=(3*self.joint_num + 3*self.joint_rotate_num + 3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num + 3*self.joint_frontalis_num,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        # print("line115 env")
        
        self.info = {}

        self.info.update({
            'success_flag': -1,
            # 'fix_flag_ship': self.fix_flag_ship,
            'step_counter': self.step_counter,
            'base_action_prime': np.array([0.0]*3*self.joint_num + [0.0]*3*self.joint_rotate_num + [0.0]*3*self.joint_ooc_l_num + [0.0]*3*self.joint_ooc_r_num + [0.0]*3*self.joint_frontalis_num),
            'x_joint': self.x_joint_init,
            'x_ooc_r_joint': self.x_ooc_r_joint_init,
            'x_ooc_l_joint': self.x_ooc_l_joint_init,
            'x_frontalis_joint': self.x_frontalis_joint_init,
            'x_joint_rotation': self.x_joint_rotation_init,
            'print_allow': self.print_allow,
            'i_count': self.i_count,
            'step_counter_internal': self.step_counter_internal,
            'target_state': self.target_state,
            'target_ooc_r_state': self.target_ooc_r_state,
            'target_ooc_l_state': self.target_ooc_l_state,
            'target_frontalis_state': self.target_frontalis_state,
            'distance_jaw': self.distance_jaw_min/self.distance_jaw_threshold_percentage,
            # 'distance': distance,

            # Add more variables as needed
        })

        obs, info = self.reset(return_info = True)
        self.info.update(info)
        self.obs = obs

        self.cmds.select(cl=True)
        self.cmds.currentCtx()
        # Close the reference                

        # self.cmds.file(self.opened_file, removeReference=True)  

        self.opened_file_status = False


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = True,  # False
        options: Optional[dict] = None, 
    ):
        super().reset(seed=seed)
        self._readback_cache = {}
        self.start_time = time.time()  # Record the start time
        
        self.point1_list = []
        self.norm_distances_list = []
        self.distance_ship_list = []        
        self.distance_log = []
        self.distance_log_phase2 = []  # For phase 2
        
        # self.time_log = []    
        self.time_for_energy_log = []
        self.strain_energy = []   
        self.step_counter_internal_log = []
        self.step_counter_internal_log_phase2 = []  # For phase 2
        self.rotational_kinetic_energy = []
        self.action_real_list = []
        self.elastic_energy = []



        self.rotation_velocity = []

        self.time_for_rotational_kinetic_energy_log = []


        self.accumulated_rotated_angle = []


        # Existing logs
        self.distance_log = []
        self.step_counter_internal_log = []
        self.strain_energy = []
        self.time_for_energy_log = []
        self.rotational_kinetic_energy = []
        self.time_for_rotational_kinetic_energy_log = []
        self.elastic_energy = []
        self.rotation_velocity = []
        self.accumulated_rotated_angle = []
        
        # Phase 2 logs
        self.distance_log_phase2 = []
        self.step_counter_internal_log_phase2 = []
        self.strain_energy_phase2 = []
        self.time_for_energy_log_phase2 = []
        self.rotational_kinetic_energy_phase2 = []
        self.time_for_rotational_kinetic_energy_log_phase2 = []
        self.elastic_energy_phase2 = []
        self.rotation_velocity_phase2 = []
        self.accumulated_rotated_angle_phase2 = []


        
        self.fix_flag_ship = [0] * (self.joint_num + self.joint_rotate_num  + self.joint_ooc_l_num + self.joint_ooc_r_num+ self.joint_frontalis_num)

        self.position = self.control_vertices_slice(self.x_bar_j, self.vertex_index)
        # print("self.vertex_index", self.vertex_index)
        # print("passed the test, now quit.")
        # sys.exit(0)  # Exit with status code 0          
        self.position_jaw = self.control_vertices_slice(self.x_jaw_bar_j, self.vertex_jaw_index)
        self.position_ooc_l =  self.control_vertices_slice(self.x_ooc_l_bar_j, self.vertex_ooc_l_index)
        self.position_ooc_r =  self.control_vertices_slice(self.x_ooc_r_bar_j, self.vertex_ooc_r_index)
        self.position_frontalis =  self.control_vertices_slice(self.x_frontalis_bar_j, self.vertex_frontalis_index)
        self.step_counter_internal = 0
        
        self.x_joint = self.x_joint_init
        self.x_joint_rotation = self.x_joint_rotation_init
        
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    # def _get_obs(self):
        
    #     return np.concatenate(
    #         [
    #             self.x_joint,
    #             self.x_joint_rotation,
    #             # self.control_vertices_slice(self.position, self.vertex_index),
    #             self.position,
    #             # self.control_vertices_slice(self.position_jaw, self.vertex_jaw_index),
    #             self.position_jaw,
    #             self.control_vertices_slice(self.target_state, self.vertex_index),
    #             self.control_vertices_slice(self.target_jaw_state, self.vertex_jaw_index),
    #         ]
    #     )

    def _get_obs(self):
        # Helper function to get the size of a numpy array or list
        def get_size(arr):
            return arr.size if isinstance(arr, np.ndarray) else len(arr)
        
        # Calculate individual sizes
        # x_joint_size = get_size(self.x_joint)
        # x_joint_rotation_size = get_size(self.x_joint_rotation)
        # position_size = get_size(self.position)
        # position_jaw_size = get_size(self.position_jaw)
        # control_vertices_slice_target_state_size = get_size(self.control_vertices_slice(self.target_state, self.vertex_index))
        # control_vertices_slice_target_jaw_state_size = get_size(self.control_vertices_slice(self.target_jaw_state, self.vertex_jaw_index))
        
        # # Print individual sizes
        # print("x_joint size: \n \n", x_joint_size)
        # print("x_joint_rotation size: \n \n", x_joint_rotation_size)
        # print("position size: \n \n", position_size)
        # print("position_jaw size: \n \n", position_jaw_size)
        # print("control_vertices_slice target_state size: \n \n", control_vertices_slice_target_state_size)
        # print("control_vertices_slice target_jaw_state size: \n \n", control_vertices_slice_target_jaw_state_size)

        # Print the sizes of the specific components
        # print("Size of x_frontalis_joint:", np.array(self.x_frontalis_joint).shape)
        # print("Size of position_frontalis:", np.array(self.position_frontalis).shape)
        # print(
        #     "Size of target_frontalis_state slice:",
        #     np.array(self.control_vertices_slice(self.target_frontalis_state, self.vertex_frontalis_index)).shape,
        # )

        # Quick debug to find the 2D array at index 7

        # print("len np.array(self.x_ooc_r_joint)", len(self.x_ooc_r_joint))
        # print("np.array(self.x_ooc_r_joint)", np.array(self.x_ooc_r_joint))
        # print("np.array(self.position_ooc_l),", np.array(self.position_ooc_l))
        # print("np.array(self.position_ooc_r),", np.array(self.position_ooc_r))


        # print("passed the test, now quit.")
        # import sys
        # sys.exit(0)  # Exit with status code 0  

        # Concatenate the observation
        obs = np.concatenate(
            [
                np.array(self.x_joint),
                np.array(self.x_joint_rotation),
                # self.control_vertices_slice(self.position, self.vertex_index),
                np.array(self.position),
                # self.control_vertices_slice(self.position_jaw, self.vertex_jaw_index),
                np.array(self.position_jaw),
                self.control_vertices_slice(self.target_state, self.vertex_index),
                self.control_vertices_slice(self.target_jaw_state, self.vertex_jaw_index),
                np.array(self.x_ooc_l_joint),
                np.array(self.x_ooc_r_joint),
                np.array(self.position_ooc_l),
                np.array(self.position_ooc_r),
                self.control_vertices_slice(self.target_ooc_l_state, self.vertex_ooc_l_index),
                self.control_vertices_slice(self.target_ooc_r_state, self.vertex_ooc_r_index),
                np.array(self.x_frontalis_joint),
                np.array(self.position_frontalis),
                self.control_vertices_slice(self.target_frontalis_state, self.vertex_frontalis_index),
            ]
        )

        
        # # Print the size of the concatenated observation
        # print("Concatenated observation size: \n \n", obs.size)

        return obs

    

    # def step(self, action, action_previous, phase_2_flag, r_limit_sigmoid_1000_endList):
    # def step(self, action, action_previous, phase_2_flag, r_limit_sigmoid_1000_endList, 
    #         run_abc=True, target_state=None, target_jaw_state=None, x_bar_j=None, x_jaw_bar_j=None):
    #     if phase_2_flag == 1 and target_state is not None and target_jaw_state is not None:
    #         self.target_state = target_state
    #         self.target_jaw_state = target_jaw_state    
    #         self.x_bar_j = x_bar_j
    #         self.x_jaw_bar_j = x_jaw_bar_j                


    def step(self, **kwargs): # action_previous

        # new_coords_jaw = self.returnCoordinates("middleres_Jaw", self.vertex_jaw_index)



        action = kwargs['action']
        current_phase = kwargs['current_phase']
        r_limit_sigmoid_1000_endList = kwargs['r_limit_sigmoid_1000_endList']
        t_limit_sigmoid_1000_endList = kwargs['t_limit_sigmoid_1000_endList']

        # if current_phase == 3:
        #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #     # s_jaw_next_adhoc = self.x_jaw_bar_j
        #     # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #     # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)
        #     method = 'euclidean'
        #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), new_coords_jaw)
        #     print("m1 line 919 before action, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)

        # step_counter_internal = kwargs['step_counter_internal']
        
    # def step(self, action, **kwargs): # action_previous
        # passed the test, now quit.

        
        """
        Execute one time step in the environment
        
        Args:
            action: Current action
            action_previous: Previous action
            current_phase: Flag indicating phase 2 status
            r_limit_sigmoid_1000_endList: List for tracking sigmoid limits
            **kwargs: Additional parameters that may include:
                - run_abc: Whether to run ABC function (default True)
                - target_state: Target state to update
                - target_jaw_state: Target jaw state to update
                - x_bar_j: Current x_bar_j value
                - x_jaw_bar_j: Current x_jaw_bar_j value
                - distance_threshold_percentage: Distance threshold percentage
                - distance_jaw_threshold_percentage: Jaw distance threshold percentage
                - distance_min: Minimum distance threshold
                - distance_jaw_min: Minimum jaw distance threshold
                - expected_total_steps: Expected total steps
                - this_task: Current task description
        """
        # Update environment parameters from kwargs
        self.run_abc = kwargs.get('run_abc') # , True
        
        # Phase 2 specific parameter updates
        if current_phase >= 2:
            if 'target_state' in kwargs:
                self.target_state = kwargs['target_state']
            if 'target_jaw_state' in kwargs:
                self.target_jaw_state = kwargs['target_jaw_state']
            if 'x_bar_j' in kwargs:
                self.x_bar_j = kwargs['x_bar_j']
            if 'x_jaw_bar_j' in kwargs:
                self.x_jaw_bar_j = kwargs['x_jaw_bar_j']

            if 'action_previous' in kwargs:
                self.action_previous = kwargs['action_previous']            
                action_previous = self.action_previous
            
            # Update evaluator parameters
            if 'distance_threshold_percentage' in kwargs:
                self.distance_threshold_percentage = kwargs['distance_threshold_percentage']
            if 'distance_jaw_threshold_percentage' in kwargs:
                self.distance_jaw_threshold_percentage = kwargs['distance_jaw_threshold_percentage']
            if 'distance_min' in kwargs:
                self.distance_min = kwargs['distance_min']
            if 'distance_jaw_min' in kwargs:
                self.distance_jaw_min = kwargs['distance_jaw_min']
            if 'expected_total_steps' in kwargs:
                self.expected_total_steps = kwargs['expected_total_steps']
            if 'this_task' in kwargs:
                self.this_task = kwargs['this_task']
            
            self.elements = kwargs['elements']

            self.abc_execution_count = kwargs['abc_execution_count']



        # if current_phase == 3:
        #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #     s_jaw_next_adhoc = self.x_jaw_bar_j

        #     # print("\n=== DEBUG: s_jaw_next_adhoc ===")
        #     # print(f"Type: {type(s_jaw_next_adhoc)}")
        #     # print(f"Shape: {s_jaw_next_adhoc.shape if hasattr(s_jaw_next_adhoc, 'shape') else 'N/A'}")
        #     # print(f"First 5 elements: {s_jaw_next_adhoc[:5] if isinstance(s_jaw_next_adhoc, (np.ndarray, list)) else s_jaw_next_adhoc}")

        #     # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #     # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)

        #     # distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_adhoc_conca, self.vertex_jaw_index))
        #     # print("\n=== DEBUG: Jaw Coordinate Shapes ===")
        #     # print(f"target_jaw_state shape: {self.target_jaw_state.shape}")
        #     # print(f"s_jaw_next_adhoc shape: {s_jaw_next_adhoc.shape}")
        #     # print(f"vertex_jaw_index: {self.vertex_jaw_index}")

        #     method = 'euclidean'
        #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(self.target_jaw_state, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_adhoc, self.vertex_jaw_index))
        #     print("m1 line 982, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)

        

        self.step_counter_internal = kwargs['step_counter_internal']

        if current_phase >= 2:
            pass
        else:

            real_stdout = sys.stdout

            # Redirect stdout and stderr to null
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

            try:
                # Check if the file is not open
                if not self.opened_file_status:
                    # self.opened_file = self.cmds.file(self.maya_file_to_open, r=True, ns=self.namespace)


                    # Check for unsaved changes and handle them
                    if self.cmds.file(query=True, modified=True):
                        # Save or discard changes as needed
                        print("Unsaved changes detected.")
                        # save_changes = input("You have unsaved changes. Do you want to save them before opening the new file? (yes/no): ").strip().lower()
                        if 1==0:  # save_changes in ['yes', 'y']:
                            self.cmds.file(save=True)
                            print("Changes saved.")
                        else:
                            self.cmds.file(newFile=True, force=True)  # Discard changes

                    # Open the Maya file
                    try:
                        self.opened_file = self.cmds.file(self.maya_file_to_open, open=True, force=True)
                        print("Opened file:", self.opened_file)

                        self.cmds.playbackOptions(min=1, max=50, animationStartTime=1, animationEndTime=50)

                        # Function to delete existing cache nodes
                        def delete_cache_nodes():
                            cache_nodes = self.cmds.ls(type='cacheFile')
                            for node in cache_nodes:
                                self.cmds.delete(node)

                        # Delete existing cache nodes
                        delete_cache_nodes()

                        self.cmds.cacheFile(createCacheNode=True, singleCache=True, startTime=1, endTime=50, fileName='myCache', dir='cacheDir')

                        # self.cmds.currentTime(15)
                        for frame_number in range(15):
                            self.cmds.currentTime(str(frame_number), edit=True)
                            

                    except Exception as e:
                        print(f"Failed to open Maya file: {e}")


                    self.opened_file_status = True  # Update the state to indicate the file is open
                else:
                    pass
                    # print("File is already open. Skipping the open operation.")

                # Perform the action
                # ...
            
            finally:
                # Restore stdout and stderr
                # sys.stdout = sys.__stdout__
                sys.stdout = real_stdout
                sys.stderr = sys.__stderr__


        # new_coords_jaw = self.returnCoordinates("middleres_Jaw", self.vertex_jaw_index)


        # if current_phase == 3:
        #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #     # s_jaw_next_adhoc = self.x_jaw_bar_j
        #     # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #     # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)
        #     method = 'euclidean'
        #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), new_coords_jaw)
        #     print("m1 line 1653 before action, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)


        reward = 0.0
        done = False

        learning_rate = 1  # 0.1 # 0.01
        action = learning_rate * action

        
        # # Add this debug code where you want to monitor the ooc_l actions

        # if not hasattr(self, 'action_iteration_count'):
        #     self.action_iteration_count = 0
        #     self.max_action_iterations = 200
            
        #     # Create debug file header (ONLY ONCE)
        #     action_debug_path = os.path.join(self.save_folder, "ooc_l_actions_debug.csv")
        #     with open(action_debug_path, 'w', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         # Create header with joint indices
        #         header = ['iteration']
        #         for i in range(3*self.joint_ooc_l_num):
        #             header.append(f'joint_ooc_l_action_{i}')
        #         writer.writerow(header)
        #     print(f"Created action debug file: {action_debug_path}")

        # # Extract and print the ooc_l actions
        # # start_idx = 3*self.joint_num + self.joint_rotate_num * 3 + 3*self.joint_ooc_l_num
        # # end_idx = 3*self.joint_num + self.joint_rotate_num * 3 + 3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num

        # start_idx = 3*self.joint_num + self.joint_rotate_num * 3
        # end_idx = 3*self.joint_num + self.joint_rotate_num * 3 + 3*self.joint_ooc_l_num

        # ooc_l_actions = action[start_idx:end_idx]

        # # Print to console for debugging
        # print(f"Iteration {self.action_iteration_count}: ooc_l actions = {ooc_l_actions}")

        # # Save to CSV file
        # action_debug_path = os.path.join(self.save_folder, "ooc_l_actions_debug.csv")
        # with open(action_debug_path, 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     row = [self.action_iteration_count]
        #     row.extend(ooc_l_actions)
        #     writer.writerow(row)

        # # Increment iteration counter
        # self.action_iteration_count += 1

        # # Check if we've reached the maximum iterations
        # if self.action_iteration_count >= self.max_action_iterations:
        #     print("\n=== Action Debug Test Completed ===")
        #     print(f"Recorded {self.max_action_iterations} iterations of ooc_l actions")
        #     print("Action debug completed, now exiting.")
        #     sys.exit(0)


        # def get_smooth_limit():
        #     """Double sigmoid that eases both at start and end"""
        #     total_growth_steps = self.overlapping_length   # 2000 for now.
            
        #     # Debug print inputs
        #     # print(f"\n=== get_smooth_limit() DEBUG ===")
        #     # print(f"Current step: {self.step_counter_internal}")
        #     # print(f"Total growth steps: {total_growth_steps}")
            
        #     t = min(self.step_counter_internal / total_growth_steps, 1.0)
        #     # print(f"Normalized progress t: {t:.4f}")
            
        #     eased_t = t**2 * (3 - 2*t)
        #     # print(f"Eased t: {eased_t:.4f}")
            
        #     t_limit = self.UNIT_PRIME * (0.25 + 0.75 * eased_t)
        #     # print(f"Raw t_limit: {t_limit:.6f}")
        #     # print(f"UNIT_PRIME: {self.UNIT_PRIME:.6f}")
            
        #     final_limit = min(t_limit, self.UNIT_PRIME)
        #     # print(f"Final limit: {final_limit:.6f}")
            
        #     return final_limit

        # t_limit_sigmoid = get_smooth_limit()
        # action[0:self.joint_num * 3] = np.clip(action[0:self.joint_num * 3], -t_limit_sigmoid, t_limit_sigmoid)
        # this is r prime   degree per step.



        action, t_limit_sigmoid = self.apply_smooth_limits_to_actions(action, current_phase)


        set_constant_threshold_percentage = 15/100
        adjust_factor_adhoc = 1.0 # 40/55
        # expected_total_steps_for_jaw = 3500  # 16000
        # cap_max = self.UNIT_PRIME_jaw_translate * 3000/max((self.expected_total_steps_for_jaw - 200), 3000) *  adjust_factor_adhoc
        # if current_phase >= 2:
        #     cap_max = self.UNIT_PRIME_jaw_translate_phase2
        #     T = self.expected_total_steps_for_jaw_phase2            
        # else:
        #     cap_max = self.UNIT_PRIME_jaw_translate
        #     T = self.expected_total_steps_for_jaw
        cap_max = self.phase_params['rotation_anatomy_limit'][current_phase]
        T = self.phase_params['expected_total_steps_for_jaw'][current_phase]
            # from :   params['expected_total_steps_for_jaw'][phase_num] = config['expected_steps']
            # params['rotation_anatomy_limit'][phase_num] = config['rotation_limit']        
        omega_max = cap_max
        
        # t1 = 0.25*T
        # t2 = 0.75*T
        # k = 0.002    # Sigmoid steepness    the 0.004 kind of corresponding to the omega_max (0.0024296)
        # 0.004 is after tuning, for 0.002 (too flat) (0.01, too steep).

        # new version for the more "linear velosity" during transition, and involve less overlapping of two controllers.
        t1 = 0.07 * T
        t2 = 0.93 * T
        k = 0.004  # Sigmoid steepness


        # print("\n=== VELOCITY PROFILE DEBUG (First 5 Steps) ===")
        # print(f"{'Step':<6} {'t':<8} {'sigmoid1':<12} {'sigmoid2':<12} {'s_diff':<12} {'env_part1':<12} {'env_part2':<12} {'envelope':<12} {'velocity':<12}")        

        # def omega(t):
        #     # print(f"\n=== omega() DEBUG at step {t} ===")
        #     # print(f"Input Parameters:")
        #     # print(f"T (total steps): {T}, t1: {t1}, t2: {t2}, k: {k}, omega_max: {omega_max}")
            
        #     def safe_exp(x, limit=700):
        #         """
        #         Safely compute exp(x) by clipping x to avoid overflow.
        #         """
        #         # print(f"safe_exp({x})")                
        #         return np.exp(np.clip(x, -limit, limit))
                    
        #     sigmoid1 = 1 / (1 + safe_exp(-k * (t - t1)))
        #     sigmoid2 = 1 / (1 + safe_exp(-k * (t - t2)))
            
        #     # print(f"Sigmoids - sigmoid1: {sigmoid1:.6f}, sigmoid2: {sigmoid2:.6f}")
        #     # print(f"Sigmoid difference: {(sigmoid1 - sigmoid2):.6f}")
            
        #     env_part1 = (1 - safe_exp(-k * t))
        #     env_part2 = (1 - safe_exp(-k * (T - t)))
        #     envelope = env_part1 * env_part2
            
        #     # print(f"Envelope - part1: {env_part1:.6f}, part2: {env_part2:.6f}, total: {envelope:.6f}")
            
        #     omega_value = omega_max * (sigmoid1 - sigmoid2) * envelope
        #     # print(f"Final omega_value: {omega_value:.6f} ({(omega_value/omega_max*100):.2f}% of omega_max)")
        #     # Print all values in aligned columns
        #     # print(f"{step:<6} {t/t_total:<8.4f} {sigmoid1:<12.6f} {sigmoid2:<12.6f} {s_diff:<12.6f} {env_part1:<12.6f} {env_part2:<12.6f} {envelope:<12.6f} {velocity:<12.6f}")            
        #     # Print key parameters
        #     # print("\n=== VELOCITY PARAMETERS ===")
        #     # print(f"{'Parameter':<25} {'Value':<15}")
        #     # print(f"{'UNIT_PRIME_jaw_translate':<25} {self.UNIT_PRIME_jaw_translate:<15.6f}")
        #     # print(f"{'expected_total_steps_for_jaw':<25} {self.expected_total_steps_for_jaw:<15}")
        #     # print(f"{'k (steepness)':<25} {k:<15.6f}")
        #     # print(f"{'t1 (7% of total)':<25} {0.07*self.expected_total_steps_for_jaw:<15.2f}")
        #     # print(f"{'t2 (93% of total)':<25} {0.93*self.expected_total_steps_for_jaw:<15.2f}")

        #     return omega_value



        t = self.step_counter_internal
        
        # velocity = omega(t)
        # Then where you call omega():
        # print("\n=== Before omega() call ===")
        # print(f"step_counter_internal: {self.step_counter_internal}")
        # print(f"expected_total_steps: {self.expected_total_steps}")
        # print(f"UNIT_PRIME_jaw_translate: {self.UNIT_PRIME_jaw_translate}")
        # print(f"adjust_factor_adhoc: {adjust_factor_adhoc}")


        # print("\n=== VELOCITY CALCULATION DEBUG ===")
        # print(f"Phase: {'2' if phase_2_flag else '1'}")
        # print(f"Current step: {self.step_counter_internal}")
        # print(f"expected_total_steps_for_jaw: {self.expected_total_steps_for_jaw}")
        # print(f"UNIT_PRIME_jaw_translate: {self.UNIT_PRIME_jaw_translate}")
        # print(f"cap_max: {cap_max}")
        # print(f"omega_max: {omega_max}")
        # print(f"t1: {t1}, t2: {t2}, k: {k}")


        # Calculate and print intermediate values
        initial_distance = self.distance_jaw_min/self.distance_jaw_threshold_percentage

        distance_need_to_travel = initial_distance - self.distance_jaw_min
        current_distance = self.info["distance_jaw"]
        distance_travelled = initial_distance - current_distance
        threshold_to_compare = distance_need_to_travel * set_constant_threshold_percentage

        left_pat = distance_need_to_travel - distance_travelled

        # r_limit_sigmoid = velocity




        angle_needed_to_rotate = self.angle_needed_rotate
        angle_threshold_to_compare = angle_needed_to_rotate * set_constant_threshold_percentage


        # At the start of your velocity calculation code:
        
        # debug_log = self._get_log_filename(phase_2_flag)
        # if False:
        #     with open(self._get_log_filename_velocity(phase_2_flag), 'a') as f:
        #         print("\n=== NEW VELOCITY CALCULATION ===", file=f)
        #         print(f"Step: {self.step_counter_internal}", file=f)
        #         print(f"Phase: {'2' if phase_2_flag else '1'}", file=f)
        #         print(f"UNIT_PRIME_jaw_translate: {self.UNIT_PRIME_jaw_translate}", file=f)
        #         print(f"expected_total_steps_for_jaw: {self.expected_total_steps_for_jaw}", file=f)
        #         print(f"t1: {t1} ({t1/T*100:.1f}% of total)", file=f)
        #         print(f"t2: {t2} ({t2/T*100:.1f}% of total)", file=f)
        #         print(f"k (steepness): {k}", file=f)

        # Inside omega() function:
        def omega(t):
            # if False:            
            #     with open(self._get_log_filename_velocity(phase_2_flag), 'a') as f:
            #         print(f"\nomega() calculation at t={t}", file=f)
            
            def safe_exp(x, limit=700):
                return np.exp(np.clip(x, -limit, limit))
                
            sigmoid1 = 1 / (1 + safe_exp(-k * (t - t1)))
            sigmoid2 = 1 / (1 + safe_exp(-k * (t - t2)))
            env_part1 = (1 - safe_exp(-k * t))
            env_part2 = (1 - safe_exp(-k * (T - t)))
            envelope = env_part1 * env_part2
            omega_value = omega_max * (sigmoid1 - sigmoid2) * envelope
            # if False:            
            #     with open(self._get_log_filename_velocity(phase_2_flag), 'a') as f:
            #         print(f"sigmoid1: {sigmoid1:.6f}", file=f)
            #         print(f"sigmoid2: {sigmoid2:.6f}", file=f)
            #         print(f"sigmoid diff: {(sigmoid1-sigmoid2):.6f}", file=f)
            #         print(f"envelope: {envelope:.6f}", file=f)
            #         print(f"raw omega_value: {omega_value:.6f}", file=f)
            
            return omega_value

        # After velocity calculation:
        velocity = omega(t)
        r_limit_sigmoid = velocity
        # if False:
        #     with open(self._get_log_filename_velocity(phase_2_flag), 'a') as f:
        #         print(f"\nVelocity result: {velocity:.6f}", file=f)
        #         print(f"Max velocity achieved: {getattr(self, 'max_velocity_achieved', 0):.6f}", file=f)

        # Condition checks
        condition1 = (self.info["distance_jaw"] <= self.distance_jaw_min + 
                    (initial_distance - self.distance_jaw_min) * set_constant_threshold_percentage)

        if not hasattr(self, 'max_velocity_achieved'):
            self.max_velocity_achieved = 0
        if velocity > self.max_velocity_achieved:
            self.max_velocity_achieved = velocity

        condition2 = (self.max_velocity_achieved > 0 and 
                    velocity < 0.5 * self.phase_params['rotation_anatomy_limit'][current_phase] and
                    velocity < 0.5 * self.max_velocity_achieved)
        # if False:
        #     with open(self._get_log_filename_velocity(phase_2_flag), 'a') as f:
        #         print(f"\nCondition checks:", file=f)
        #         print(f"Condition1 (distance): {condition1}", file=f)
        #         print(f"Condition2 (velocity decay): {condition2}", file=f)
        #         print(f"Current distance: {self.info['distance_jaw']:.6f}", file=f)
        #         print(f"Threshold distance: {self.distance_jaw_min + (initial_distance - self.distance_jaw_min) * set_constant_threshold_percentage:.6f}", file=f)
        #         print(f"Current velocity: {velocity:.6f}", file=f)
        #         print(f"0.5*max_velocity: {0.5*self.max_velocity_achieved:.6f}", file=f)

        # Final limit calculation
        if condition1 or condition2:
            if self.fix_r_limit_flag == 0:
                self.fix_r_limit_flag = 1
                if condition2:
                    self.fixed_r_limit_sigmoid = 0.5 * self.max_velocity_achieved
                else:
                    self.fixed_r_limit_sigmoid = r_limit_sigmoid * 0.5
                # if False:
                #     with open(self._get_log_filename_velocity(phase_2_flag), 'a') as f:
                #         print("\nLIMITER ACTIVATED!", file=f)
                #         print(f"Setting fixed_r_limit_sigmoid to: {self.fixed_r_limit_sigmoid:.6f}", file=f)
                #         print(f"Triggered by: {'condition2' if condition2 else 'condition1'}", file=f)

        if self.fix_r_limit_flag == 1:
            r_limit_sigmoid = self.fixed_r_limit_sigmoid

        MINIMUM_THRESHOLD = 1.62831122e-06
        final_r_limit = max(r_limit_sigmoid, MINIMUM_THRESHOLD)

        # with open(self._get_log_filename_velocity(phase_2_flag), 'a') as f:
        #             print(f"Step {step_counter},", file=f)
        # if False:
        #     with open(self._get_log_filename_velocity(phase_2_flag), 'a') as f:
        #         print(f"\nFinal limits:", file=f)
        #         print(f"Before minimum threshold: {r_limit_sigmoid:.6f}", file=f)
        #         print(f"After minimum threshold: {final_r_limit:.6f}", file=f)
        #         print(f"Minimum threshold: {MINIMUM_THRESHOLD:.6f}", file=f)
        #         print(f"Ratio to UNIT_PRIME: {final_r_limit/self.UNIT_PRIME_jaw_translate*100:.2f}%", file=f)

        r_limit_sigmoid = final_r_limit



        # velocity = omega(t)

        # print(f"calculated velocity: {velocity}")

        # if self.step_counter_internal >= 400:
        #     # Exit the program
        #     # import sys
        #     sys.exit(0)


        # r_limit_sigmoid = velocity

        # # set_to_constant_threshold = 3000









        # print("\n=== DEBUG: r_limit_sigmoid condition check ===")
        # print(f"angle_needed_to_rotate: {angle_needed_to_rotate}")
        # print(f"step_counter_internal: {self.step_counter_internal}")
        # print(f"travelled_angle: {travelled_angle}")
        # print(f"angle_threshold_to_compare: {angle_threshold_to_compare}")
        # print(f"Condition: {travelled_angle} > {angle_threshold_to_compare} ? {travelled_angle > angle_threshold_to_compare}")


        # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
            
        #     print(f'self.info[distance_jaw]: {self.info["distance_jaw"]}', file = f)
        #     print(f"self.distance_jaw_min : {self.distance_jaw_min}", file = f)
        #     print(f'(self.info[distance_jaw] - self.distance_jaw_min) * set_constant_threshold_percentage: {(self.info["distance_jaw"] - self.distance_jaw_min) * set_constant_threshold_percentage}', file = f)

        # Condition 1: Distance-based threshold (original condition)
        # condition1 = (self.info["distance_jaw"] <= self.distance_jaw_min + 
        #             (initial_distance - self.distance_jaw_min) * set_constant_threshold_percentage)

        # # Condition 2: Velocity has peaked and dropped below half of maximum
        # # Track maximum velocity achieved
        # if not hasattr(self, 'max_velocity_achieved'):
        #     self.max_velocity_achieved = 0
        # if velocity > self.max_velocity_achieved:
        #     self.max_velocity_achieved = velocity

        # condition2 = (self.max_velocity_achieved > 0 and 
        #             velocity < 0.5 * self.UNIT_PRIME_jaw_translate and
        #             velocity < 0.5 * self.max_velocity_achieved)

        # # Original condition with debug
        # # if left_pat < threshold_to_compare:
        # if condition1 or condition2:
        # # if self.info["distance_jaw"] <= self.distance_jaw_min + (initial_distance - self.distance_jaw_min) * set_constant_threshold_percentage :
        #     # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
        #     #     print("\nCONDITION TRIGGERED!", file = f)
        #     if self.fix_r_limit_flag == 0:
        #         # print("Setting fixed_r_limit_sigmoid for the first time")
        #         self.fix_r_limit_flag = 1
        #         if condition2 == True:
        #             self.fixed_r_limit_sigmoid = 0.5 * self.max_velocity_achieved
        #         else:
        #             self.fixed_r_limit_sigmoid = r_limit_sigmoid * 1/2
        #         # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                    
        #         #     print(f"velocity: {velocity}", file = f)
        #         #     print(f"step_counter_internal: {self.step_counter_internal}", file = f)
        #         #     print(f"fixed_r_limit_sigmoid set to: {self.fixed_r_limit_sigmoid}", file = f)
                    
        # else:
        #     # print("\nCondition NOT triggered")
        #     pass

        # if self.fix_r_limit_flag == 1:
        #     # print(f"Using fixed_r_limit_sigmoid: {self.fixed_r_limit_sigmoid} (original was: {r_limit_sigmoid})")
        #     r_limit_sigmoid = self.fixed_r_limit_sigmoid

        # # print(f" r_limit_sigmoid before line 1326: {r_limit_sigmoid}")



        # # Apply minimum threshold to the velocity cap
        # MINIMUM_THRESHOLD = 1.62831122e-06  #  1.2174288e-06
        # r_limit_sigmoid = max(r_limit_sigmoid, MINIMUM_THRESHOLD)  # if r_limit_sigmoid < MINIMUM_THRESHOLD, make r_limit_sigmoid = MINIMUM_THRESHOLD

        # print(f"MINIMUM_THRESHOLD: {MINIMUM_THRESHOLD}")
        # print(f"final r_limit_sigmoid: {max(velocity, MINIMUM_THRESHOLD)}")      

        # if self.step_counter_internal >= 50:
        #     sys.exit(0)

        def omega_rad(r_limit_sigmoid):
            return r_limit_sigmoid * np.pi / 180  # Convert °/s to rad/s

        velocity_rad = omega_rad(r_limit_sigmoid)


        # r_limit_sigmoid = self.UNIT_PRIME_jaw_translate * (3*(self.step_counter_internal/self.expected_total_steps)**2 - 2*(self.step_counter_internal/self.expected_total_steps)**3)
        if r_limit_sigmoid >= cap_max:
            r_limit_sigmoid = cap_max
        

        # scale_factor_r_limit = round(3/7, 3)
        # scale_factor_r_limit = round(1, 3)
        # r_limit_sigmoid = r_limit_sigmoid * scale_factor_r_limit
        action[self.joint_num * 3:self.joint_num * 3+self.joint_rotate_num * 3] = np.clip(action[self.joint_num * 3:self.joint_num * 3+self.joint_rotate_num * 3], -r_limit_sigmoid, r_limit_sigmoid)  # [0]


        # print(f"Full action array stats:")
        # print(f"Min: {np.min(action)}, Max: {np.max(action)}, Mean: {np.mean(action)}")
        # print(f"Rotation part stats (indices {self.joint_num*3} to {self.joint_num*3+self.joint_rotate_num*3}):")
        rotation_part = action[self.joint_num*3:self.joint_num*3+self.joint_rotate_num*3]
        # print(f"Min: {np.min(rotation_part)}, Max: {np.max(rotation_part)}, Mean: {np.mean(rotation_part)}")        


        # print("\n=== CLIPPING DEBUG ===")
        # print(f"Clipping bounds: [-{r_limit_sigmoid}, {r_limit_sigmoid}]")
        # print(f"Values before clipping: {action[self.joint_num*3:self.joint_num*3+self.joint_rotate_num*3]}")
        clipped = np.clip(action[self.joint_num*3:self.joint_num*3+self.joint_rotate_num*3], -r_limit_sigmoid, r_limit_sigmoid)
        # print(f"Values after clipping: {clipped}")


        # Extract the 3 elements
        elements = action[self.joint_num * 3 : self.joint_num * 3 + self.joint_rotate_num * 3]

        # Calculate the average
        rotational_velocity_average = sum(elements) # / len(elements)        

        # my code:
        s_prime = self.position
        s_jaw_prime = self.position_jaw
        
        reward = 0.0
        base_action_prime = np.array([0.0]*3*self.joint_num + [0.0]*3*self.joint_rotate_num + [0.0]*3*self.joint_ooc_l_num + [0.0]*3*self.joint_ooc_r_num + [0.0]*3*self.joint_frontalis_num)       

        # print("size of action", len(action))
        # print("size of base_action_prime", len(base_action_prime))

        # for translation
        for actionI in range(0,3*self.joint_num):           
            base_action_prime[int(actionI)] = action[int(actionI)]                       

        # for rotation
        for actionI in range(3*self.joint_num, 3*self.joint_num + self.joint_rotate_num * 3):    # 0,3*self.joint_num         
            base_action_prime[int(actionI)] = action[int(actionI)]

        # for translate joint_ooc_l_num
        for actionI in range(3*self.joint_num + self.joint_rotate_num * 3, 3*self.joint_num + self.joint_rotate_num * 3 + 3*self.joint_ooc_l_num ):    # 0,3*self.joint_num         
            base_action_prime[int(actionI)] = action[int(actionI)]



        # for translate joint_ooc_r_num
        for actionI in range(3*self.joint_num + self.joint_rotate_num * 3 + 3*self.joint_ooc_l_num , 3*self.joint_num + self.joint_rotate_num * 3 + 3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num):    # 0,3*self.joint_num         
            base_action_prime[int(actionI)] = action[int(actionI)]


        # # for translate joint_frontalis_num
        # for actionI in range(3*self.joint_num + self.joint_rotate_num * 3 + 3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num, 3*self.joint_num + self.joint_rotate_num * 3 + 3*self.joint_ooc_l_num + 3*self.joint_ooc_r_num + 3 * 3*self.joint_frontalis_num):    # 0,3*self.joint_num         
        #     base_action_prime[int(actionI)] = action[int(actionI)]

        # print("Size of base_action_prime:", len(base_action_prime) if isinstance(base_action_prime, list) else base_action_prime.shape)
        # print("Size of action:", len(action) if isinstance(action, list) else action.shape)

        # for translate joint_frontalis_num
        for actionI in range(3 * self.joint_num + self.joint_rotate_num * 3 + 
                            3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num, 
                            3 * self.joint_num + self.joint_rotate_num * 3 + 
                            3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 
                            3 * self.joint_frontalis_num):  # 0, 3 * self.joint_num
            # Print sizes of base_action_prime and action


            # Assign the action value
            base_action_prime[int(actionI)] = action[int(actionI)]


        # bk1
        if self.fix_flag_ship[0] == 1:
            base_action_prime[0:3*self.joint_num] = 0

        if self.fix_flag_ship[1] == 1:
            base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num] = 0
        
        if self.fix_flag_ship[2] == 1:
            base_action_prime[3*self.joint_num + 3*self.joint_rotate_num:3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num] = 0

        if self.fix_flag_ship[3] == 1:
            base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num] = 0

        if self.fix_flag_ship[4] == 1:
            base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num] = 0


        # s_prime_return_coordinates = self.returnCoordinates()

        # to bypass rotation for now
        # base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num] = 0
        # # to bypass oor for now
        # base_action_prime[0:3*self.joint_num] = 0
        # to bypass ooc for now
        # base_action_prime[3*self.joint_num + 3*self.joint_rotate_num:3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num] = 0
        
        # base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num] = 0

        # to bypass frontalis for now
        # base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num] = 0

        # new_coords_jaw = self.returnCoordinates("middleres_Jaw", self.vertex_jaw_index)


        # if current_phase == 3:
        #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #     # s_jaw_next_adhoc = self.x_jaw_bar_j
        #     # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #     # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)
        #     method = 'euclidean'
        #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), new_coords_jaw)
        #     print("m1 line 1528 before action, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)


        def run_previous_ending_actions():
            
            try:
                # Debug print before the problematic line
                # print("\n=== DEBUGGING ARRAY DIMENSIONS ===")
                # print(f"r_limit_sigmoid_1000_endList type: {type(r_limit_sigmoid_1000_endList)}, length: {len(r_limit_sigmoid_1000_endList)}")
                # print(f"r_limit_sigmoid_1000_endList content: {r_limit_sigmoid_1000_endList}")
                # print(f"action_previous type: {type(action_previous)}, shape: {np.asarray(action_previous).shape}")
                # print(f"action_previous content (first 20 elements): {action_previous[:20]}")
                # print(f"self.abc_execution_count: {self.abc_execution_count}")
                # print(f"self.joint_num: {self.joint_num}")
                # print(f"self.joint_rotate_num: {self.joint_rotate_num}")
                # print(f"self.joint_num * 3: {self.joint_num * 3}")
                # print(f"self.joint_rotate_num * 3: {self.joint_rotate_num * 3}")
                # print(f"self.joint_num * 3 + self.joint_rotate_num * 3: {self.joint_num * 3 + self.joint_rotate_num * 3}")
                # print(f"len(action_previous): {len(action_previous)}")
                
                # Check if indices are within bounds
                start_idx = self.joint_num * 3
                end_idx = self.joint_num * 3 + self.joint_rotate_num * 3
                if end_idx > len(action_previous):
                    print(f"ERROR: end_idx {end_idx} exceeds action_previous length {len(action_previous)}")
                if self.abc_execution_count >= len(r_limit_sigmoid_1000_endList):
                    print(f"ERROR: abc_execution_count {self.abc_execution_count} exceeds r_limit_sigmoid_1000_endList length {len(r_limit_sigmoid_1000_endList)}")
                
                # The original operation with additional checks
                clip_min = -r_limit_sigmoid_1000_endList[self.abc_execution_count]
                clip_max = r_limit_sigmoid_1000_endList[self.abc_execution_count]
                # print(f"Clip range: [{clip_min}, {clip_max}]")
                
                action_previous[start_idx:end_idx] = np.clip(
                    action_previous[start_idx:end_idx],
                    clip_min,
                    clip_max
                )

                # clip for lip action of the previous one.
                start_idx = 0
                end_idx = self.joint_num * 3
                # if end_idx > len(action_previous):
                #     print(f"ERROR: end_idx {end_idx} exceeds action_previous length {len(action_previous)}")
                if self.abc_execution_count >= len(t_limit_sigmoid_1000_endList):
                    print(f"ERROR: abc_execution_count {self.abc_execution_count} exceeds r_limit_sigmoid_1000_endList length {len(t_limit_sigmoid_1000_endList)}")
                
                # The original operation with additional checks
                clip_min = -t_limit_sigmoid_1000_endList[self.abc_execution_count]
                clip_max = t_limit_sigmoid_1000_endList[self.abc_execution_count]
                # print(f"Clip range: [{clip_min}, {clip_max}]")
                
                action_previous[start_idx:end_idx] = np.clip(
                    action_previous[start_idx:end_idx],
                    clip_min,
                    clip_max
                )

                
            except Exception as e:
                print(f"\n=== ERROR OCCURRED ===")
                print(f"Exception type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("\nCurrent variable states:")
                print(f"action_previous length: {len(action_previous) if hasattr(action_previous, '__len__') else 'N/A'}")
                print(f"r_limit_sigmoid_1000_endList length: {len(r_limit_sigmoid_1000_endList) if hasattr(r_limit_sigmoid_1000_endList, '__len__') else 'N/A'}")
                print(f"self.abc_execution_count: {getattr(self, 'abc_execution_count', 'N/A')}")
                import traceback
                traceback.print_exc()
                # import sys
                sys.exit(1)

            # action_previous[self.joint_num * 3:self.joint_num * 3+self.joint_rotate_num * 3] = np.clip(action_previous[self.joint_num * 3:self.joint_num * 3+self.joint_rotate_num * 3], -r_limit_sigmoid_1000_endList[self.abc_execution_count], r_limit_sigmoid_1000_endList[self.abc_execution_count])  # [0]
            
            base_action_prime_previous = np.array([0.0]*3*self.joint_num + [0.0]*3*self.joint_rotate_num + [0.0]*3*self.joint_ooc_l_num + [0.0]*3*self.joint_ooc_r_num + [0.0]*3*self.joint_frontalis_num)       

            # print("size of action", len(action))
            # print("size of base_action_prime", len(base_action_prime))

            # for translation
            for actionI in range(0,3*self.joint_num):           
                base_action_prime_previous[int(actionI)] = action_previous[int(actionI)]                       

            # for rotation
            for actionI in range(3*self.joint_num, 3*self.joint_num + self.joint_rotate_num * 3):    # 0,3*self.joint_num         
                base_action_prime_previous[int(actionI)] = action_previous[int(actionI)]

            return base_action_prime_previous


        # Run ABC function if enabled and within limit
        # if self.run_abc and self.abc_execution_count < self.MAX_ABC_EXECUTIONS:
        #     self._run_abc_function()
        #     self.abc_execution_count += 1

        # if current_phase == 3:
        #     print("self.run_abc is:", self.run_abc)
        #     print("self.abc_execution_count", self.abc_execution_count)

        if self.run_abc and self.abc_execution_count < self.MAX_ABC_EXECUTIONS:  #
            # Run ABC function
            base_action_prime_previous = run_previous_ending_actions()
            base_action_prime = base_action_prime + base_action_prime_previous
            self.abc_execution_count += 1

            # if current_phase == 3:
            #     print("did the previous action carryover, base_action_prime_previous is:", base_action_prime_previous[0:15])


        if "both" == self.action_mode:
            x_joint_ = self.x_joint + base_action_prime[0:3*self.joint_num]
            x_joint_rotation_ = self.x_joint_rotation + base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]
            x_ooc_l_joint_ = self.x_ooc_l_joint + base_action_prime[3*self.joint_num + 3*self.joint_rotate_num:3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num]
            x_ooc_r_joint_ = self.x_ooc_r_joint + base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num]
            x_frontalis_joint_ = self.x_frontalis_joint + base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num]





        # new_coords_temp0 = self.returnCoordinates("middleres_OrbicularisOris")

        # Visualization check at line 1776
        def visualize(vertices, object_name):
            """
            Visualize the vertices, the sampled joint positions, and the surface mesh.

            Args:
            - vertices (list of lists): List of 3D coordinates of vertices.
            - object_name (str): Name for the output file.
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

            # Save the plot to a file
            output_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\d83 initiate rest to AU12 smile"
            output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        # Visualize new_coords_temp0 at line 1776
        # coords_temp0_non_concatenate = [new_coords_temp0[i:i+3] for i in range(0, len(new_coords_temp0), 3)]
        # visualize(coords_temp0_non_concatenate, "line1776_new_coords_temp0")

        # new_coords_jaw = self.returnCoordinates("middleres_Jaw", self.vertex_jaw_index)


        # if current_phase == 3:
        #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #     # s_jaw_next_adhoc = self.x_jaw_bar_j
        #     # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #     # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)
        #     method = 'euclidean'

        #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), new_coords_jaw)
        #     print("m1 line 1653 before action, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)




        relative_rotation_value = base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]


        # rotate. 
        if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
            for i in range(0,self.joint_rotate_num):  #3-3,18-3
                flag_temp1 = np.array_equal(self.x_joint_rotation[3*i],x_joint_rotation_[3*i])
                flag_temp2 = np.array_equal(self.x_joint_rotation[3*i+1],x_joint_rotation_[3*i+1])
                flag_temp3 = np.array_equal(self.x_joint_rotation[3*i+2],x_joint_rotation_[3*i+2])
                # print("rotate flag_temp1, ", flag_temp1)
                # print("flag_temp2, ", flag_temp2)
                # print("flag_temp3, ", flag_temp3)
                if (not flag_temp1) or (not flag_temp2) or (not flag_temp3):  
                    # self.cmds.rotate( relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws = True)
                    # print("really rotate l423")


                    try:
                        
                        self.cmds.rotate( relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws = True)
                        # print("really rotate l428")
                        #bk2
                        # temporary manually lock rotate y and z
                    except Exception as e:
                        print(f"l332 at env, unimportant error, An error occurred: {e}")
                        if relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0:
                            try:
                                print("\n\n line 594 joint_jaw_rotate_num:", self.joint_jaw_rotate_num, " \n\n'jaw0'+str(i+1)", '\n\njaw0', str(i+1))
                                self.cmds.rotate( 0, 0, 0, 'jaw0'+str(i+1), relative=True, ws = True)

                            except Exception as e3:
                                print(f"An error occurred: {e3}")
                                pass 

        # calculate the elastic energy incurred by the rotation of jaw.
        # theta = cumulative_trapezoid(velocity_rad, t, initial=0)
        # base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]
        # # theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]
        
        # k_tissue = 10000  # Stiffness of soft tissues (N/m)
        # r = 0.17  # Radius of rotation (m)        
        # # Calculate displacement (x) as r * theta
        # x = r * theta        

        # # Elastic energy function
        # def elastic_energy():
        #     return 0.5 * k_tissue * x**2
        
        # E_elastic = elastic_energy()


        # # translate
        # if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
        #     for i in range(0,self.joint_num):
        #         flag_temp1 = np.array_equal(self.x_joint[3*i],x_joint_[3*i])
        #         flag_temp2 = np.array_equal(self.x_joint[3*i+1],x_joint_[3*i+1])
        #         flag_temp3 = np.array_equal(self.x_joint[3*i+2],x_joint_[3*i+2])
        #         if (not flag_temp1) or (not flag_temp2) or (not flag_temp3):  
        #             # print("translated l449 i = ", i)
        #             self.cmds.move( x_joint_[3*i], x_joint_[3*i+1], x_joint_[3*i+2], 'lip_lower0'+str(7), absolute=True )


        # translate
        if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
            _moved = np.any(np.asarray(self.x_joint, dtype=float).reshape(-1, 3) != np.asarray(x_joint_, dtype=float).reshape(-1, 3), axis=1)
            for i in range(self.joint_num):
                if _moved[i]:
                    # Determine the joint name based on the index i
                    if i + 1 <= 9:
                        joint_name = f'oor_0{i + 1}'  # Leading zero for single-digit numbers
                    elif i + 1 <= 99:
                        joint_name = f'oor_{i + 1}'
                    elif i + 1 <= 999:
                        joint_name = f'oor_{i + 1}'
                    else:
                        joint_name = f'oor_{i + 1}'

                    # Move the joint
                    if 1==0: # joint_name == "lip_lower191":
                        # print("lip_lower191 x_joint_", x_joint_)
                        
                        # print("lip_lower191 self.x_joint", self.x_joint)
                        # print("lip_lower191 size self.x_joint", len(self.x_joint))
                        # print("lip_lower191 base_action_prime[0:3*self.joint_num]", base_action_prime[0:3*self.joint_num])
                        print("lip_lower191 size base_action_prime[0:3*self.joint_num]", len(base_action_prime[0:3*self.joint_num]))
                        # self.cmds.move(self.x_joint[3 * i], x_joint_[3 * i + 1], self.x_joint[3 * i + 2], joint_name, absolute=True)
                        objectList = "lip_lower191"
                        position_t = self.cmds.xform(objectList, query=True, worldSpace=True, translation=True)
                        print(position_t)

                        x_joint_init_conca_t = position_t
                        i_temp2 = 0
                        self.cmds.move( x_joint_init_conca_t[3*i_temp2]+0.17, x_joint_init_conca_t[3*i_temp2+1], x_joint_init_conca_t[3*i_temp2+2], joint_name, absolute=True)
                    else: 
                        # print("joint_name to deform this time", joint_name)
                        self.cmds.move(x_joint_[3 * i], x_joint_[3 * i + 1], x_joint_[3 * i + 2], joint_name, absolute=True)


        # translate ooc l
        # if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
        #     for i in range(self.joint_ooc_l_num):  # joint_num is replaced by joint_ooc_l_num
        #         flag_temp1 = np.array_equal(self.x_ooc_l_joint[3 * i], x_ooc_l_joint_[3 * i])
        #         flag_temp2 = np.array_equal(self.x_ooc_l_joint[3 * i + 1], x_ooc_l_joint_[3 * i + 1])
        #         flag_temp3 = np.array_equal(self.x_ooc_l_joint[3 * i + 2], x_ooc_l_joint_[3 * i + 2])
        #         if not flag_temp1 or not flag_temp2 or not flag_temp3:
        #             # Determine the joint name based on the index i
        #             if i + 1 <= 9:
        #                 joint_name = f'ooc_l_0{i + 1}'  # Leading zero for single-digit numbers
        #             elif i + 1 <= 99:
        #                 joint_name = f'ooc_l_{i + 1}'
        #             elif i + 1 <= 999:
        #                 joint_name = f'ooc_l_{i + 1}'
        #             else:
        #                 joint_name = f'ooc_l_{i + 1}'

        #             # Move the joint
        #             if 1==0: # joint_name == "lip_lower191":
        #                 pass                    

        #             else: 
        #                 # print("joint_name to deform this time", joint_name)
        #                 self.cmds.move(x_ooc_l_joint_[3 * i], x_ooc_l_joint_[3 * i + 1], x_ooc_l_joint_[3 * i + 2], joint_name, absolute=True)

        # # In your deformation code, add the comparison logic:

        # if not hasattr(self, 'iteration_count'):
        #     self.iteration_count = 0
        #     self.max_iterations = 200
            
        #     # Create debug file header (ONLY ONCE)
        #     debug_file_path = os.path.join(self.save_folder, "difference_debug_comparison.csv")
        #     with open(debug_file_path, 'w', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow([
        #             'iteration', 'average_difference', 'all_differences'
        #         ])

            
        # if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
        #     # Store coordinates before deformation
        #     before_deform_coords_ooc_l = self.returnCoordinates("middleres_OrbicularisOculi_l", self.vertex_ooc_l_index)
            
        #     # Perform the deformation (your existing code)
        #     for i in range(self.joint_ooc_l_num):
        #         flag_temp1 = np.array_equal(self.x_ooc_l_joint[3 * i], x_ooc_l_joint_[3 * i])
        #         flag_temp2 = np.array_equal(self.x_ooc_l_joint[3 * i + 1], x_ooc_l_joint_[3 * i + 1])
        #         flag_temp3 = np.array_equal(self.x_ooc_l_joint[3 * i + 2], x_ooc_l_joint_[3 * i + 2])
        #         if not flag_temp1 or not flag_temp2 or not flag_temp3:
        #             if i + 1 <= 9:
        #                 joint_name = f'ooc_l_0{i + 1}'
        #             elif i + 1 <= 99:
        #                 joint_name = f'ooc_l_{i + 1}'
        #             elif i + 1 <= 999:
        #                 joint_name = f'ooc_l_{i + 1}'
        #             else:
        #                 joint_name = f'ooc_l_{i + 1}'

        #             if 1==0: # joint_name == "lip_lower191":
        #                 pass                    
        #             else: 
        #                 self.cmds.move(x_ooc_l_joint_[3 * i], x_ooc_l_joint_[3 * i + 1], x_ooc_l_joint_[3 * i + 2], joint_name, absolute=True)

        #     # Get coordinates after deformation
        #     new_coords_ooc_l = self.returnCoordinates("middleres_OrbicularisOculi_l", self.vertex_ooc_l_index)
            
        #     # print("before_deform_coords_ooc_l", before_deform_coords_ooc_l)
        #     # print("new_coords_ooc_l", new_coords_ooc_l)
        #     # Calculate the difference
        #     if before_deform_coords_ooc_l is not None and new_coords_ooc_l is not None:
        #         if len(before_deform_coords_ooc_l) == len(new_coords_ooc_l):
        #             # Calculate Euclidean distance for each vertex
        #             differences = []
        #             # for before, after in zip(before_deform_coords_ooc_l, new_coords_ooc_l):
        #             #     if len(before) >= 3 and len(after) >= 3:
        #             diff = np.linalg.norm(np.array(before_deform_coords_ooc_l) - np.array(new_coords_ooc_l)) # [:3]
        #             differences.append(diff)
                    
        #             if differences:
        #                 avg_diff = np.mean(differences)
        #                 max_diff = np.max(differences)
        #                 min_diff = np.min(differences)
                        
        #             # Save to CSV file
        #             debug_file_path = os.path.join(self.save_folder, "difference_debug_comparison.csv")
        #             with open(debug_file_path, 'a', newline='') as csvfile:
        #                 writer = csv.writer(csvfile)
        #                 writer.writerow([
        #                     self.iteration_count,  # iteration
        #                     avg_diff,              # position_difference
        #                     differences, # all_differences
        #                 ])
                    
        #             print(f"Iteration {self.iteration_count}: Avg diff: {avg_diff:.6f}, Max diff: {max_diff:.6f}, Min diff: {min_diff:.6f}")
                        

                        

            
        #     # Increment iteration counter
        #     self.iteration_count += 1
            
        #     # Check if we've reached the maximum iterations
        #     if self.iteration_count >= self.max_iterations:
        #         print("\n=== Deformation Test Completed ===")
        #         print(f"Performed {self.max_iterations} iterations")
                
        #         # Print summary statistics
        #         if differences:
        #             # all_avg_diffs = [diff['average_difference'] for diff in self.coordinate_differences]
        #             # all_max_diffs = [diff['max_difference'] for diff in self.coordinate_differences]
                    
        #             # final_avg = np.mean(all_avg_diffs)
        #             # final_max = np.max(all_max_diffs)
                    
        #             print(f"Final average difference across all iterations: {avg_diff:.6f}")
        #             print(f"Maximum difference observed: {max_diff:.6f}")
                    
        #             # Save results to file
        #             # self.save_deformation_results()
                
        #         print("Test completed, now exiting.")
        #         # import sys
        #         sys.exit(0)

        # In your deformation code, add the comparison logic:

        # if not hasattr(self, 'iteration_count'):
        #     self.iteration_count = 0
        #     self.max_iterations = 200
            
        #     # Create debug file header (ONLY ONCE)
        #     debug_file_path = os.path.join(self.save_folder, "difference_debug_comparison.csv")
        #     with open(debug_file_path, 'w', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow([
        #             'iteration', 'average_difference', 'max_difference', 'min_difference',
        #             'element_wise_differences'
        #         ])

            
        if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
            # Store coordinates before deformation
            # before_deform_coords_ooc_l = self.returnCoordinates("middleres_OrbicularisOculi_l", self.vertex_ooc_l_index)
            # print("before_deform_coords_ooc_l", before_deform_coords_ooc_l)
            
            # Perform the deformation (your existing code)
            _moved = np.any(np.asarray(self.x_ooc_l_joint, dtype=float).reshape(-1, 3) != np.asarray(x_ooc_l_joint_, dtype=float).reshape(-1, 3), axis=1)
            for i in range(self.joint_ooc_l_num):
                if _moved[i]:
                    if i + 1 <= 9:
                        joint_name = f'ooc_l_0{i + 1}'
                    elif i + 1 <= 99:
                        joint_name = f'ooc_l_{i + 1}'
                    elif i + 1 <= 999:
                        joint_name = f'ooc_l_{i + 1}'
                    else:
                        joint_name = f'ooc_l_{i + 1}'

                    if 1==0: # joint_name == "lip_lower191":
                        pass                    
                    else: 
                        # print("ooc_l deformed here l1987 i = ", i)
                        self.cmds.move(x_ooc_l_joint_[3 * i], x_ooc_l_joint_[3 * i + 1], x_ooc_l_joint_[3 * i + 2], joint_name, absolute=True)

            # # Get coordinates after deformation
            # new_coords_ooc_l = self.returnCoordinates("middleres_OrbicularisOculi_l", self.vertex_ooc_l_index)
            # print("new_coords_ooc_l", new_coords_ooc_l)
            
            # # Calculate the difference
            # if before_deform_coords_ooc_l is not None and new_coords_ooc_l is not None:
            #     if len(before_deform_coords_ooc_l) == len(new_coords_ooc_l):
            #         # Convert to numpy arrays for easier computation
            #         before_array = np.array(before_deform_coords_ooc_l)
            #         after_array = np.array(new_coords_ooc_l)

            #         print("before_array", before_array)
            #         print("after_array", after_array)
                    
            #         # Calculate Euclidean distance for each vertex (element-wise)
            #         vertex_differences = before_array - after_array
                    
            #         # Calculate coordinate-wise differences (x, y, z separately)
            #         coordinate_differences = np.abs(before_array - after_array)
                    
            #         # Calculate summary statistics
            #         avg_diff = np.mean(vertex_differences)
            #         max_diff = np.max(vertex_differences)
            #         min_diff = np.min(vertex_differences)

            #         print("vertex_differences", vertex_differences)
                    
            #         # Format differences for CSV output
            #         # For vertex differences: list of Euclidean distances for each vertex
            #         # vertex_diff_str = ';'.join([f'{d:.6f}' for d in vertex_differences])
                    
            #         # For coordinate differences: format as "vertex_idx:x_diff,y_diff,z_diff;..."
            #         # coord_diff_str_parts = []
            #         # for i, (before_coord, after_coord) in enumerate(zip(before_array, after_array)):
            #         #     x_diff = abs(before_coord[0] - after_coord[0])
            #         #     y_diff = abs(before_coord[1] - after_coord[1])
            #         #     z_diff = abs(before_coord[2] - after_coord[2])
            #         #     coord_diff_str_parts.append(f'{i}:{x_diff:.6f},{y_diff:.6f},{z_diff:.6f}')
            #         # coord_diff_str = ';'.join(coord_diff_str_parts)
                    
            #         # Save to CSV file
            #         debug_file_path = os.path.join(self.save_folder, "difference_debug_comparison.csv")
            #         with open(debug_file_path, 'a', newline='') as csvfile:
            #             writer = csv.writer(csvfile)
            #             writer.writerow([
            #                 self.iteration_count,  # iteration
            #                 f'{avg_diff:.6f}',    # average_difference
            #                 f'{max_diff:.6f}',    # max_difference
            #                 f'{min_diff:.6f}',    # min_difference
            #                 # vertex_diff_str,      # all_differences (vertex Euclidean distances)
            #                 vertex_differences        # element_wise_differences (x,y,z per vertex)
            #             ])
                    
            #         # Also print some summary info to console
            #         print(f"Iteration {self.iteration_count}: Avg diff: {avg_diff:.6f}, Max diff: {max_diff:.6f}, Min diff: {min_diff:.6f}")
                    
            #         # Print the vertex with maximum difference for debugging
            #         max_diff_vertex = np.argmax(vertex_differences)
            #         max_diff_coords = coordinate_differences[max_diff_vertex]
            #         print(f"  Max difference at vertex {max_diff_vertex}: "
            #             )
                    
            #         # f"x={max_diff_coords[0]:.6f}, y={max_diff_coords[1]:.6f}, z={max_diff_coords[2]:.6f}"

            
            # # Increment iteration counter
            # self.iteration_count += 1
            
            # # Check if we've reached the maximum iterations
            # if self.iteration_count >= self.max_iterations:
            #     print("\n=== Deformation Test Completed ===")
            #     print(f"Performed {self.max_iterations} iterations")
                
            #     # Print final summary statistics
            #     print(f"Final average difference: {avg_diff:.6f}")
            #     print(f"Maximum difference observed: {max_diff:.6f}")
            #     print(f"Minimum difference observed: {min_diff:.6f}")
                
            #     # Find and print the vertex with the overall maximum difference
            #     if vertex_differences.size > 0:
            #         overall_max_vertex = np.argmax(vertex_differences)
            #         overall_max_coords = coordinate_differences[overall_max_vertex]
            #         # print(f"Overall maximum difference at vertex {overall_max_vertex}: "
            #         #     f"x={overall_max_coords[0]:.6f}, y={overall_max_coords[1]:.6f}, z={overall_max_coords[2]:.6f}")
                
            #     print("Test completed, now exiting.")
            #     # import sys
            #     sys.exit(0)

        # translate ooc l
        if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
            _moved = np.any(np.asarray(self.x_ooc_r_joint, dtype=float).reshape(-1, 3) != np.asarray(x_ooc_r_joint_, dtype=float).reshape(-1, 3), axis=1)
            for i in range(self.joint_ooc_r_num):
                if _moved[i]:
                    # Determine the joint name based on the index i
                    if i + 1 <= 9:
                        joint_name = f'ooc_r_0{i + 1}'  # Leading zero for single-digit numbers
                    elif i + 1 <= 99:
                        joint_name = f'ooc_r_{i + 1}'
                    elif i + 1 <= 999:
                        joint_name = f'ooc_r_{i + 1}'
                    else:
                        joint_name = f'ooc_r_{i + 1}'

                    # Move the joint
                    if 1==0: # joint_name == "lip_lower191":
                        pass                    

                    else: 
                        # print("joint_name to deform this time", joint_name)
                        self.cmds.move(x_ooc_r_joint_[3 * i], x_ooc_r_joint_[3 * i + 1], x_ooc_r_joint_[3 * i + 2], joint_name, absolute=True)


        # translate frontalis
        if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
            _moved = np.any(np.asarray(self.x_frontalis_joint, dtype=float).reshape(-1, 3) != np.asarray(x_frontalis_joint_, dtype=float).reshape(-1, 3), axis=1)
            for i in range(self.joint_frontalis_num):
                if _moved[i]:
                    # Determine the joint name based on the index i
                    if i + 1 <= 9:
                        joint_name = f'frontalis_0{i + 1}'  # Leading zero for single-digit numbers
                    elif i + 1 <= 99:
                        joint_name = f'frontalis_{i + 1}'
                    elif i + 1 <= 999:
                        joint_name = f'frontalis_{i + 1}'
                    else:
                        joint_name = f'frontalis_{i + 1}'

                    # Move the joint
                    if 1==0: # joint_name == "lip_lower191":
                        pass                    

                    else: 
                        # print("joint_name to deform this time", joint_name)
                        self.cmds.move(x_frontalis_joint_[3 * i], x_frontalis_joint_[3 * i + 1], x_frontalis_joint_[3 * i + 2], joint_name, absolute=True)


        # objectList = ['middleres_Jaw']
        # x_conca_temp = []


        # if current_phase == 3:
        #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #     s_jaw_next_adhoc = self.x_jaw_bar_j
        #     # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #     # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)


        #     # distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_adhoc_conca, self.vertex_jaw_index))
        #     method = 'euclidean'
        #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(self.target_jaw_state, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_adhoc, self.vertex_jaw_index))
        #     print("m1 line 1801, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)
       

        new_coords = self.returnCoordinates("middleres_OrbicularisOris", self.vertex_index)
        
        new_coords_jaw = self.returnCoordinates("middleres_Jaw", self.vertex_jaw_index)
        new_coords_ooc_r = self._read_if_moved("middleres_OrbicularisOculi_r", x_ooc_r_joint_, self.vertex_ooc_r_index)
        new_coords_ooc_l = self._read_if_moved("middleres_OrbicularisOculi_l", x_ooc_l_joint_, self.vertex_ooc_l_index)
        new_coords_frontalis = self._read_if_moved("middleres_Frontalis", x_frontalis_joint_, self.vertex_frontalis_index)
        # bypass these three since we don't handle them in this task.
      

        # new_coords_temp1 = self.returnCoordinates("middleres_OrbicularisOris")

        # Visualization check at line 2232 (now line 2266)
        def visualize_temp1(vertices, object_name):
            """
            Visualize the vertices, the sampled joint positions, and the surface mesh.

            Args:
            - vertices (list of lists): List of 3D coordinates of vertices.
            - object_name (str): Name for the output file.
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

            # Save the plot to a file
            output_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\d83 initiate rest to AU12 smile"
            output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        # Visualize new_coords_temp1 at line 2232 (now line 2266)
        # coords_temp1_non_concatenate = [new_coords_temp1[i:i+3] for i in range(0, len(new_coords_temp1), 3)]
        # visualize_temp1(coords_temp1_non_concatenate, "line2232_new_coords_temp1")

        # print("passed the test, now quit.")
        # # import sys
        # sys.exit(0)  # Exit 

        # if current_phase == 3:
        #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #     # s_jaw_next_adhoc = self.x_jaw_bar_j
        #     # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #     # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)
        #     method = 'euclidean'

        #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), new_coords_jaw)
        #     print("m1 line 1822, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)


        # new_coords_ooc_l = self.position_ooc_l
        # new_coords_ooc_r = self.position_ooc_r
        # new_coords_frontalis = self.position_frontalis


        # self.log_plot_every = 5
        # calculate energy every 200 internal steps.
        if self.step_counter_internal % self.log_plot_every == 0:
            new_coords_for_energy = self.returnCoordinates("middleres_OrbicularisOris")
            coords_rest = self.x_bar_j  # self.target_state
            def strain_energy_calculator(new_coords, coords_rest):

                import numpy as np
                # import sys
                sys.path.insert(0, str(a3_stochastic_relative_path / "a1-blendshape-guided-animation-pipeline"/ "database"))
                import importlib

                # import dataV22_after_deform
                # import dataV22
                # importlib.reload(dataV22_after_deform)
                # importlib.reload(dataV22)


                # Material properties (Lamé parameters)
                mu = 1.0      # Shear modulus
                lambda_ = 1.0  # Lamé's first parameter

                # Rest pose: mixed mesh (triangles and quadrangles) in 3D
                vertices_rest = np.array(coords_rest)

                # Deformed pose for hybrid method
                vertices_deformed_hybrid = np.array(new_coords)

                # Deformed pose for pure kinematic method
                # vertices_deformed_kinematic = np.array(dataV22.x_middleres_OrbicularisOris1)

                # Elements: mix of triangles and quadrangles (vertex indices)
                # elements = dataV22.x_middleres_OrbicularisOris_face_0
                elements = self.elements

                # Function to split quadrangles into triangles
                def split_quadrangle(element):
                    # Split quadrangle into two triangles: [v0, v1, v2], [v0, v2, v3]
                    return [[element[0], element[1], element[2]], [element[0], element[2], element[3]]]

                # Function to compute deformation gradient F and area A
                # def compute_F_and_A(vertices_rest, vertices_deformed, element):
                #     # Rest pose edges
                #     X = vertices_rest[element]
                
                # def compute_F_and_A(vertices_rest, vertices_deformed, triangle):
                #     print(f"\n=== DEBUG: compute_F_and_A ===")
                #     print(f"vertices_rest shape: {vertices_rest.shape}")
                #     print(f"vertices_deformed shape: {vertices_deformed.shape}")
                #     print(f"triangle: {triangle}")
                #     print(f"Max element index in triangle: {max(triangle)}")
                    
                #     # Check if any triangle indices are out of bounds
                #     for i, element in enumerate(triangle):
                #         if element >= len(vertices_rest):
                #             print(f"ERROR: Triangle index {i} has invalid element {element} (max allowed: {len(vertices_rest)-1})")
                    
                #     X = vertices_rest[element]  # This is where the error occurs   

                #     E1_rest = X[1] - X[0]
                #     E2_rest = X[2] - X[0]
                    
                #     # Deformed pose edges
                #     x = vertices_deformed[element]
                #     E1_deformed = x[1] - x[0]
                #     E2_deformed = x[2] - x[0]
                    

                #     # debug the "F = np.column_stack((E1_deformed, E2_def" dimonsion error
                #     # log_path = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\log_debug25-2-20.txt"


                #     # new_coords_for_energy = new_coords
                #     # with open(log_path, 'w') as f:
                #     #     # Print new_coords_for_energy
                #     #     print("new_coords_for_energy shape:", np.shape(new_coords_for_energy), file=f)
                #     #     print("new_coords_for_energy first 5 elements:\n", new_coords_for_energy[:5], file=f)
                        
                #     #     # Print coords_rest
                #     #     print("\ncoords_rest shape:", np.shape(coords_rest), file=f)
                #     #     print("coords_rest first 5 elements:\n", coords_rest[:5], file=f)

                #     #     # Print self.elements
                #     #     print("\nself.elements shape:", np.shape(self.elements), file=f)
                #     #     print("self.elements first 5 elements:\n", self.elements[:5], file=f)

                #     #     # Compute deformation vectors
                #     #     # E1_deformed = new_coords_for_energy[self.elements[:, 1]] - new_coords_for_energy[self.elements[:, 0]]
                #     #     # E2_deformed = new_coords_for_energy[self.elements[:, 2]] - new_coords_for_energy[self.elements[:, 0]]
                        
                #     #     # E1_rest = coords_rest[self.elements[:, 1]] - coords_rest[self.elements[:, 0]]
                #     #     # E2_rest = coords_rest[self.elements[:, 2]] - coords_rest[self.elements[:, 0]]

                #     #     # Print E1_deformed and E2_deformed
                #     #     print("\nE1_deformed shape:", np.shape(E1_deformed), file=f)
                #     #     print("E1_deformed first 1 elements:\n", E1_deformed, file=f)
                        
                #     #     print("\nE2_deformed shape:", np.shape(E2_deformed), file=f)
                #     #     print("E2_deformed first 1 elements:\n", E2_deformed, file=f)

                #     #     # Print E1_rest and E2_rest
                #     #     print("\nE1_rest shape:", np.shape(E1_rest), file=f)
                #     #     print("E1_rest first 1 elements:\n", E1_rest, file=f)

                #     #     print("\nE2_rest shape:", np.shape(E2_rest), file=f)
                #     #     print("E2_rest first 1 elements:\n", E2_rest, file=f)

                #     #     # Check if any of these arrays are empty (which might cause the np.cross error)
                #     #     if E1_deformed.size == 0 or E2_deformed.size == 0:
                #     #         print("\nERROR: One of the deformation vectors (E1_deformed, E2_deformed) is empty!", file=f)

                #     #     if E1_rest.size == 0 or E2_rest.size == 0:
                #     #         print("\nERROR: One of the rest vectors (E1_rest, E2_rest) is empty!", file=f)

                #     #     # Try computing the transformation matrix F and catch errors
                #     #     try:
                #     #         F = np.column_stack((E1_deformed, E2_deformed, np.cross(E1_deformed, E2_deformed))) @ \
                #     #             np.linalg.inv(np.column_stack((E1_rest, E2_rest, np.cross(E1_rest, E2_rest))))
                #     #         print("\nF matrix computed successfully!", file=f)
                #     #     except Exception as e:
                #     #         print("\nERROR while computing F:", str(e), file=f)



                #     # Deformation gradient F (3x3 matrix, approximate plane deformation)
                #     F = np.column_stack((E1_deformed, E2_deformed, np.cross(E1_deformed, E2_deformed))) @ \
                #         np.linalg.inv(np.column_stack((E1_rest, E2_rest, np.cross(E1_rest, E2_rest))))
                    
                #     # Area of the triangle (half the magnitude of the cross product of the two edges)
                #     area = 0.5 * np.linalg.norm(np.cross(E1_rest, E2_rest))
                    
                #     return F, area

                def compute_F_and_A(vertices_rest, vertices_deformed, triangle):
                    # print(f"\n=== DEBUG: compute_F_and_A ===")
                    # print(f"Triangle indices: {triangle}")
                    # print(f"vertices_rest shape: {vertices_rest.shape}")
                    # print(f"vertices_deformed shape: {vertices_deformed.shape}")
                    
                    # Get the triangle vertices
                    X = vertices_rest[triangle]
                    x = vertices_deformed[triangle]
                    # print(f"Rest vertices (X): {X}")
                    # print(f"Deformed vertices (x): {x}")
                    
                    # Compute edge vectors
                    E1_rest = X[1] - X[0]
                    E2_rest = X[2] - X[0]
                    E1_deformed = x[1] - x[0]
                    E2_deformed = x[2] - x[0]
                    
                    # print(f"E1_rest: {E1_rest}, shape: {E1_rest.shape}")
                    # print(f"E2_rest: {E2_rest}, shape: {E2_rest.shape}")
                    # print(f"E1_deformed: {E1_deformed}, shape: {E1_deformed.shape}")
                    # print(f"E2_deformed: {E2_deformed}, shape: {E2_deformed.shape}")
                    
                    # Check for zero-length edges
                    if np.linalg.norm(E1_rest) < 1e-10 or np.linalg.norm(E2_rest) < 1e-10:
                        print("WARNING: Degenerate rest triangle detected!")
                    if np.linalg.norm(E1_deformed) < 1e-10 or np.linalg.norm(E2_deformed) < 1e-10:
                        print("WARNING: Degenerate deformed triangle detected!")
                    
                    try:
                        F = np.column_stack((E1_deformed, E2_deformed, np.cross(E1_deformed, E2_deformed))) @ \
                            np.linalg.inv(np.column_stack((E1_rest, E2_rest, np.cross(E1_rest, E2_rest))))
                        A = 0.5 * np.linalg.norm(np.cross(E1_rest, E2_rest))
                    except Exception as e:
                        print(f"ERROR in compute_F_and_A: {str(e)}")
                        print("Returning identity matrix and zero area")
                        return np.eye(3), 0.0
                    
                    return F, A

                # Strain energy density function (St. Venant-Kirchhoff, 3D)
                def strain_energy_density(F):
                    C = F.T @ F  # Right Cauchy-Green tensor
                    E = 0.5 * (C - np.eye(3))  # Green-Lagrange strain tensor
                    trace_E = np.trace(E)
                    return 0.5 * lambda_ * (trace_E ** 2) + mu * np.sum(E ** 2)

                # Total energy calculation
                def compute_total_energy(vertices_rest, vertices_deformed, elements):
                    # print(f"\n=== DEBUG: compute_total_energy ===")
                    # print(f"vertices_rest length: {len(vertices_rest)}")
                    # print(f"vertices_deformed length: {len(vertices_deformed)}")
                    # print(f"elements shape: {len(elements)}")
                    # print(f"Max index in elements: {elements.max()}")
                    
                    # if elements.max() >= len(vertices_rest):
                    #     print("ERROR: Elements contain indices larger than vertex array!")
                    # Convert elements to numpy array if it's a list
                    elements_arr = np.array(elements)
                    # print(f"elements shape: {elements_arr.shape}")
                    # print(f"Max index in elements: {elements_arr.max()}")
                    
                    if elements_arr.max() >= len(vertices_rest):
                        print("ERROR: Elements contain indices larger than vertex array!")                    

                # def compute_total_energy(vertices_rest, vertices_deformed, elements):
                    total_energy = 0
                    for element in elements:
                        if len(element) == 4:  # Quadrangle
                            # Split quadrangle into two triangles
                            triangles = split_quadrangle(element)
                        else:
                            # Single triangle
                            triangles = [element]
                        
                        for triangle in triangles:
                            F, A = compute_F_and_A(vertices_rest, vertices_deformed, triangle)
                            psi = strain_energy_density(F)
                            total_energy += psi * A
                    return total_energy

                # Compute energies
                # energy_rest = compute_total_energy(vertices_rest, vertices_rest, elements)
                energy_hybrid = compute_total_energy(vertices_rest, vertices_deformed_hybrid, elements)
                # energy_kinematic = compute_total_energy(vertices_rest, vertices_deformed_kinematic, elements)

                # Print results
                # print(f"Energy at rest: {energy_rest}")
                # print(f"Energy for hybrid method: {energy_hybrid}")
                # print(f"Energy for half pure kinematic method: {energy_kinematic}")

                return energy_hybrid # energy_rest, 

            # call:
            new_coords_for_energy_non_concatenate = [new_coords_for_energy[i:i+3] for i in range(0, len(new_coords_for_energy), 3)]
            coords_rest_non_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]
            # strain_energy = strain_energy_calculator(new_coords_for_energy_non_concatenate, coords_rest_non_concatenate)


            # calculate the kinetic energy.
            if self.fix_flag_ship[1] == 1: # rotation has stopped, kinetic energy to 0.
                rotation_velocity = 0
                rotational_kinetic_energy = 0     
            else:                   
                # expected_total_steps_for_jaw = 3500
                rotation_velocity = rotational_velocity_average  #  / (  (35/30) /self.expected_total_steps_for_jaw) # unit: degree per second
                rotation_velocity_prime = rotation_velocity * math.pi / 180  # unit: rad/s
                I = 1/3 * 0.1089 * 0.17**2  # jaw weighs (wet, and estimated mass with marrow) 0.1089kg, Radius of Rotation: 17 cm (from the center of rotation to the middle teeth).
                # jaw weight referece: {Zhang, Futang, title={Mass properties of the human mandible}
                E_rotation = 1/2 * I * rotation_velocity_prime**2  # equal to "The work (physical terminology) done by the muscles"
                rotational_kinetic_energy = E_rotation
                convert_factor = 1.0  # 10e-3  if convert J to mJ 
                rotational_kinetic_energy = rotational_kinetic_energy * convert_factor

                # After calculating rotation_velocity, add:
                # print(f"\n=== ROTATION VELOCITY DEBUG ===")
                # print(f"Phase: {'2' if phase_2_flag else '1'}")
                # print(f"Current step: {self.step_counter_internal}")
                # print(f"Action segment (rotation part): {action[self.joint_num*3:self.joint_num*3+self.joint_rotate_num*3]}")
                # print(f"Elements extracted: {elements}")
                # print(f"Raw average: {sum(elements)/len(elements)}")
                # print(f"Time factor: {(35/30)/self.expected_total_steps_for_jaw}")
                # print(f"Final rotation_velocity: {rotation_velocity}")
                # print(f"r_limit_sigmoid: {r_limit_sigmoid}")
                # print(f"velocity_rad: {velocity_rad}")            

        # Extract the x, y, and z coordinates from new_coords
        x_coords = new_coords[0::3]
        y_coords = new_coords[1::3]
        z_coords = new_coords[2::3]

        x_coords_jaw = new_coords_jaw[0::3]
        y_coords_jaw = new_coords_jaw[1::3]
        z_coords_jaw = new_coords_jaw[2::3]
        # Check if all three vertices are within the boundary cube
        within_boundary = all(
            self.MAZE_BOUNDARY[0] <= x <= self.MAZE_BOUNDARY[1] and
            self.MAZE_BOUNDARY[2] <= y <= self.MAZE_BOUNDARY[3] and
            self.MAZE_BOUNDARY[4] <= z <= self.MAZE_BOUNDARY[5]
            for x, y, z in zip(x_coords, y_coords, z_coords)
        )

        within_boundary_jaw = all(
            self.MAZE_BOUNDARY_jaw[0] <= x <= self.MAZE_BOUNDARY_jaw[1] and
            self.MAZE_BOUNDARY_jaw[2] <= y <= self.MAZE_BOUNDARY_jaw[3] and
            self.MAZE_BOUNDARY_jaw[4] <= z <= self.MAZE_BOUNDARY_jaw[5]
            for x, y, z in zip(x_coords_jaw, y_coords_jaw, z_coords_jaw)
        )

        # if within_boundary and within_boundary_jaw:
        if 1==1:

        # Check if new position is within boundaries
            # return next_coords
            s_next = new_coords
            s_jaw_next = new_coords_jaw
            s_ooc_l_next = new_coords_ooc_l
            s_ooc_r_next = new_coords_ooc_r
            s_frontalis_next = new_coords_frontalis

            # if "translation" != self.action_mode:
            #     s_jaw_next = self.returnCoordinates("middleres_Jaw", self.vertex_jaw_index)
            # else:
            #     s_jaw_next = s_jaw_prime
            # pass

        else:
            # Action is invalid, return current state

            s_next = s_prime
            s_jaw_next = s_jaw_prime
            x_joint_ = self.x_joint
            x_joint_rotation_ = self.x_joint_rotation

            # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:                
            #     print('Step ' + repr(self.step_counter) + ',  Action is invalid ', file = f)

        self.position = s_next  
        self.position_jaw = s_jaw_next
        self.position_ooc_r = s_ooc_r_next
        self.position_ooc_l = s_ooc_l_next
        self.position_frontalis = s_frontalis_next

        success_flag = -1       

        distance = 0.0
        
        method = 'euclidean'
        distance = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(self.target_state, self.vertex_index), s_next)           

        # def plot_adhoc():

        #     def visualize(vertices, object_name):
        #         """
        #         Visualize the vertices, the sampled joint positions, and the surface mesh.

        #         Args:
        #         - vertices (list of lists): List of 3D coordinates of vertices.
        #         - faces (list of lists): List of indices defining the faces of the mesh.
        #         - sampled_points (list of lists): List of 3D coordinates of sampled joint positions.
        #         """
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111, projection='3d')

        #         # Plot the vertices
        #         vertices_np = np.array(vertices)
        #         ax.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2], c='b', marker='o', s=1, label='Vertices')

        #         # Set the view angle
        #         ax.view_init(elev=(180+30), azim=0, roll=90)  # Change these values to set the desired view

        #         ax.set_xlabel('X Label')
        #         ax.set_ylabel('Y Label')
        #         ax.set_zlabel('Z Label')
        #         ax.legend()
        #         # plt.show()
                

        #         # Save the plot to a file
        #         output_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\e68 backed to e71 no speed up and visualize debug"
        #         output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
        #         plt.savefig(output_file, dpi=300, bbox_inches='tight')
        #         plt.close()        

        #     s_jaw_next_full = self.returnCoordinates("middleres_Jaw")
        #     coords_rest = s_jaw_next_full # np.array(dataV22.x_middleres_OrbicularisOris0)
            
        #     if current_phase == 3:
        #         x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #         x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #         target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #         s_jaw_next_adhoc = s_jaw_next_full
        #         # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #         # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)

        #         # distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_adhoc_conca, self.vertex_jaw_index))
        #         method = 'euclidean'
        #         distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(self.target_jaw_state, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_adhoc, self.vertex_jaw_index))
        #         print("m1 line 2199, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)


        #     # call:
        #     # new_coords_for_energy_non_concatenate = [new_coords_for_energy[i:i+3] for i in range(0, len(new_coords_for_energy), 3)]
        #     coords_rest_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]
        #     # x_bar_conca_rest_adhoc = [x_bar_conca_rest_adhoc[i:i+3] for i in range(0, len(x_bar_conca_rest_adhoc), 3)]

        #     # rest_vertices = np.array(dataV22.x_middleres_OrbicularisOris0)
        #     visualize(coords_rest_concatenate, "phase3 jaw plot")                    


        # if current_phase == 3:
        #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
        #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
        #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
        #     # s_jaw_next_adhoc = s_jaw_next_full
        #     # s_jaw_next_adhoc_conca = [item for sublist in s_jaw_next_adhoc for item in sublist]
        #     # s_jaw_next_adhoc_conca = np.asarray(s_jaw_next_adhoc_conca)
        #     method = 'euclidean'

        #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), s_jaw_next)
        #     print("m1 line 2220, at the start of phase 3, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)




        # method = 'euclidean'
        distance_jaw = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(self.target_jaw_state, self.vertex_jaw_index), s_jaw_next)

        # if current_phase == 3:
        #     plot_adhoc()
        #     print("at begining of phase 3, distance_jaw between current jaw and phoneme T: ", distance_jaw)
        #     # print("debug end")
        #     # sys.exit(0)  # Exit with status code 0   



        distance_ooc_r = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(self.target_ooc_r_state, self.vertex_ooc_r_index), s_ooc_r_next)
        distance_ooc_l = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(self.target_ooc_l_state, self.vertex_ooc_l_index), s_ooc_l_next)
        distance_frontalis = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(self.target_frontalis_state, self.vertex_frontalis_index), s_frontalis_next)


        distance_temp = distance



        visualize_organ = 0
        if visualize_organ == 1:
            keyframe_num_temp = 2

            x_bar_j = self.x_bar_j
            # x_bar_conca = [item for sublist in getattr(dataV22, "x_middleres_OrbicularisOris" + str(keyframe_num_temp - 2)) for item in sublist]
            
            new_coords_for_energy = self.returnCoordinates("middleres_OrbicularisOris")
            coords_rest = x_bar_j

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

            rest_vertices = coords_rest # np.array(dataV22.x_middleres_OrbicularisOris0)
            


            # call:
            new_coords_for_energy_non_concatenate = [new_coords_for_energy[i:i+3] for i in range(0, len(new_coords_for_energy), 3)]
            coords_rest_non_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]

            # rest_vertices = np.array(dataV22.x_middleres_OrbicularisOris0)
            # visualize(coords_rest_non_concatenate, "rest")
            visualize(new_coords_for_energy_non_concatenate, "current")

        # current_state_temp = s_next
        # target_state_temp = self.control_vertices_slice(self.target_state, self.vertex_index)


        epsilon_min = 0.01
        pause_gradient_ratio = 0.9
        if distance <= pause_gradient_ratio * self.distance_min + epsilon_min:
            self.fix_flag_ship[0] = 1

        if distance_jaw <= pause_gradient_ratio * self.distance_jaw_min:  #  + epsilon_min
            self.fix_flag_ship[1] = 1
        
        if distance_ooc_l <= pause_gradient_ratio * self.distance_ooc_l_min + epsilon_min:
            self.fix_flag_ship[2] = 1
        
        if distance_ooc_r <= pause_gradient_ratio * self.distance_ooc_r_min + epsilon_min:
            self.fix_flag_ship[3] = 1
                                    
        if distance_frontalis <= pause_gradient_ratio * self.distance_frontalis_min + epsilon_min:
            self.fix_flag_ship[4] = 1

    
        if "rotation" == self.action_mode:
            # w_dist = 0
            # w_direction = 0.001*100
            w_dist = 0.1
            w_direction_translation = 0.001
            w_direction_translation_jaw = 0.1

        if "translation" == self.action_mode:
            # do translation only
            # w_dist = 0.999*100 * 0.1
            # w_direction = 0.001*100  * 0.9   
            # w_dist = 0.1
            # w_direction = 1

            w_dist = 0.1
            w_direction_translation = 0.001
            w_direction_translation_jaw = 0.1

        if "both" == self.action_mode:
            w_dist = 0.1
            w_direction_translation = 0.001
            w_direction_translation_jaw = 0.1
            # need tune this. make it balanced.


        distance_ = distance
        distance_jaw_ = distance_jaw
        distance_ooc_l_ = distance_ooc_l
        distance_ooc_r_ = distance_ooc_r
        distance_frontalis_ = distance_frontalis
        # note: this is high level debug print.

        # reward_dist = math.exp(-distance**2)
        # reward_dist_jaw = math.exp(-distance_jaw**2)
        # reward_dist = -distance


        w_differential_translation = 1
        w_differential_rotation = 1

        # Example usage

        reward, w_dist, w_dist_jaw, w_dist_ooc_l, w_dist_ooc_r, w_dist_frontalis, self.fix_flag_ship = self.reward_function.compute_combined_reward(
            distance, 
            distance_jaw, 
            distance_ooc_l,
            distance_ooc_r,
            distance_frontalis,
            # w_direction_translation,
            # reward_direction_translation,
            # w_differential_translation,
            # reward_differential_translation,
            # w_differential_rotation,
            # reward_differential_rotation,
            self.fix_flag_ship,
            self.action_mode
        )
        # , addition_distance, addition_distance_jaw

        # self.point1_list.append(self.target_state)
        # self.point_jaw_list.append(self.target_jaw_state)
        
        if self.step_counter % self.print_every == 0:                 
            
            self.print_allow = 1
            self.i_count = 0
        # print("self.print_allow", self.print_allow)
        if(self.print_allow == 1):

            # def extend_boundaries():
            #     n = len(self.vertex_index)
            #     max_boundaries_single = np.array([self.MAZE_BOUNDARY[1], self.MAZE_BOUNDARY[3], self.MAZE_BOUNDARY[5]])
            #     min_boundaries_single = np.array([self.MAZE_BOUNDARY[0], self.MAZE_BOUNDARY[2], self.MAZE_BOUNDARY[4]])
                
            #     # Extend the arrays by repeating them n times
            #     max_boundaries = np.tile(max_boundaries_single, n)
            #     min_boundaries = np.tile(min_boundaries_single, n)
                
            #     return max_boundaries, min_boundaries

            # max_boundaries, min_boundaries = extend_boundaries()

            # Phase 1
            self.log_step_details(
                current_phase=current_phase,
                step_counter=self.step_counter,
                s_prime=s_prime,
                base_action_prime=base_action_prime,
                s_next=s_next,
                x_joint=self.x_joint,
                x_joint_=x_joint_,
                x_joint_rotation=self.x_joint_rotation,
                x_joint_rotation_=x_joint_rotation_,
                distance=distance,
                distance_jaw=distance_jaw,
                distance_ooc_r=distance_ooc_r,
                distance_ooc_l=distance_ooc_l,
                distance_frontalis=distance_frontalis,
                reward=reward,
                expected_total_steps=self.expected_total_steps,
                step_counter_internal=self.step_counter_internal,
                remaining_steps=self.remaining_steps
            )

            # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
            #     print('Step ' + repr(self.step_counter) + ', \n  current State[0:12]: ' + repr(s_prime[0:12]) + ', \n Action[0:12]: ' + repr(base_action_prime[0:12]) + ', \n next State[0:12]:  ' + repr(s_next[0:12])+ ', \n Target State[0:12]:  ' + repr(self.control_vertices_slice(self.target_state, self.vertex_index)[0:12]) + ', \n current joint position[0:12]: ' + repr(self.x_joint[0:12]) + ', \n next joint position[0:12]: ' + repr(x_joint_[0:12]) + ', \n current joint rotation position: '+ repr(self.x_joint_rotation) + ', \n next joint rotation position: '+ repr(x_joint_rotation_) , file = f)
            #     print(" distance after action ", distance, file = f)
            #     print(" distance jaw after action ", distance_jaw, file = f)
            #     print(" rotation action ", base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num], file = f)
            #     print(" distance ooc_r after action ", distance_ooc_r, file = f)
            #     print(" distance ooc_l after action ", distance_ooc_l, file = f)
            #     print(" distance frontalis after action ", distance_frontalis, file = f)                                
            #     # print(f"Angle between normal vector of s_next and target_state: {angle_degrees_rotation:.4f} degrees", file=f)               
            #     # print(" w*reward_dist ", w_dist * reward_dist, file = f)
            #     # print("w* reward_direction translation", w_direction_translation * reward_direction_translation, file = f)
            #     # print("w_dist_jaw * reward_dist_jaw", w_dist_jaw * reward_dist_jaw, file = f)
            #     # print("w* reward_differential translation ", w_differential_translation * reward_differential_translation, file = f)
            #     # print("w* reward_differential rotation ", w_differential_rotation * reward_differential_rotation, file = f)     
            #     # print("addition_distance ", addition_distance, file = f)     
            #     # print("addition_distance_jaw ", addition_distance_jaw, file = f)                                       
            #     print(" reward ", reward, file = f)
            #     print("self.expected_total_steps, self.step_counter_internal , self.remaining_steps ", self.expected_total_steps, self.step_counter_internal, self.remaining_steps, file = f)
            #     # print("MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max", self.MAZE_BOUNDARY[0:12], file = f)
            #     # print("difference with max control_vertices_slice", s_prime - max_boundaries, file = f)
            #     # print("difference with min control_vertices_slice", s_prime - min_boundaries, file = f)
               
            self.i_count += 1
            # print("self.i_count", self.i_count)
            if self.i_count >=self.print_continue_num:
                self.print_allow = 0
        self.step_counter = self.step_counter + 1       

        # --- Exit Condition ---
        # if self.step_counter == 2250:
        #     print("passed the test, now quit.")
            
        #     sys.exit(0)  # Exit with status code 0                 


        self.action_real_list.append(base_action_prime) # important line.

        delta_b0_conca_not_used = self.calculate_accumulated_rotated_angle()



        epsilon_min = 0.01 # avoid the small change leading to not converge when the distance min is 0.

        if "both" == self.action_mode:
            # Bypass OOC success threshold if theoretical distance is 0 for these specific tasks
            current_phase = self.phase_manager.current_phase if hasattr(self, 'phase_manager') else 1
            current_task = self.phase_params.get('task', {}).get(current_phase, "")
            
            # ooc_bypassed = current_task in ["dataV22_AU6 to dataV22_AU7", "dataV22_AU7 to dataV22_AU6"]
            
            # ooc_l_condition = True if ooc_bypassed else (distance_ooc_l <= self.distance_ooc_l_min + epsilon_min)
            # ooc_r_condition = True if ooc_bypassed else (distance_ooc_r <= self.distance_ooc_r_min + epsilon_min)

            # ooc_l_condition and ooc_r_condition and
            # if distance_jaw <= self.distance_jaw_min and distance <= self.distance_min  + epsilon_min and distance_frontalis <= self.distance_frontalis_min  + epsilon_min:
            if distance_jaw <= self.distance_jaw_min and distance <= self.distance_min  + epsilon_min and distance_ooc_l <= self.distance_ooc_l_min  + epsilon_min and distance_ooc_r <= self.distance_ooc_r_min  + epsilon_min and distance_frontalis <= self.distance_frontalis_min  + epsilon_min:

                # def obtain_r_limit_sigmoid_1000_endList():
                #     adjust_factor_adhoc = 1.0 # 40/55
                #     cap_max = self.UNIT_PRIME_jaw_translate * 3000/max((self.expected_total_steps - 200), 3000) *  adjust_factor_adhoc
                #     omega_max = cap_max
                #     T = self.step_counter_internal + 1000
                #     # t1 = 0.25*T
                #     # t2 = 0.75*T
                #     # k = 0.002    # Sigmoid steepness    the 0.004 kind of corresponding to the omega_max (0.0024296)
                #     # 0.004 is after tuning, for 0.002 (too flat) (0.01, too steep).

                #     # new version for the more "linear velosity" during transition, and involve less overlapping of two controllers.
                #     t1 = 0.07 * T
                #     t2 = 0.93 * T
                #     k = 0.004  # Sigmoid steepness

                #     def omega(t):
                #         def safe_exp(x, limit=700):
                #             """
                #             Safely compute exp(x) by clipping x to avoid overflow.
                #             """
                #             return np.exp(np.clip(x, -limit, limit))


                #         # # Debugging: Check input values
                #         # print(f"k: {k}, t: {t}, t1: {t1}, t2: {t2}")

                #         # # Calculate sigmoid values safely
                #         # arg1 = -k * (t - t1)
                #         # arg2 = -k * (t - t2)
                #         # print(f"arg1: {arg1}, arg2: {arg2}")





                #         # Use safe_exp in your sigmoid calculations
                #         sigmoid1 = 1 / (1 + safe_exp(-k * (t - t1)))  # Sigmoid for acceleration
                #         sigmoid2 = 1 / (1 + safe_exp(-k * (t - t2)))  # Sigmoid for deceleration            
                #         # sigmoid1 = 1 / (1 + np.exp(-k * (t - t1)))  # Sigmoid for acceleration
                #         # sigmoid2 = 1 / (1 + np.exp(-k * (t - t2)))  # Sigmoid for deceleration
                #         # print(f"sigmoid1: {sigmoid1}, sigmoid2: {sigmoid2}")

                #         envelope = (1 - np.exp(-k * t)) * (1 - np.exp(-k * (T - t)))  # Envelope to ensure omega(0) = 0 and omega(T) = 0
                #         return omega_max * (sigmoid1 - sigmoid2) * envelope        
                    
                #     t =  np.arange(self.step_counter_internal, self.step_counter_internal + 1000)  # t from 0 to 4200 # self.step_counter_internal
                    
                #     velocity = omega(t)
                #     r_limit_sigmoid = velocity
                #     r_limit_sigmoid_1000_endList = r_limit_sigmoid

                #     return r_limit_sigmoid_1000_endList

                

                # def obtain_r_limit_sigmoid_1000_endList():
                    # print("\n=== DEBUGGING obtain_r_limit_sigmoid_1000_endList ===")
                    # print(f"self.UNIT_PRIME_jaw_translate: {self.UNIT_PRIME_jaw_translate}")
                    # print(f"self.expected_total_steps: {self.expected_total_steps}")
                    # print(f"self.step_counter_internal: {self.step_counter_internal}")
                    
                    # adjust_factor_adhoc = 1.0
                    # cap_max = self.UNIT_PRIME_jaw_translate * 3000/max((self.expected_total_steps - 200), 3000) * adjust_factor_adhoc
                    # omega_max = cap_max
                    # T = self.step_counter_internal + 1000
                    
                    # # print(f"cap_max: {cap_max}")
                    # # print(f"omega_max: {omega_max}")
                    # # print(f"T: {T}")
                    
                    # t1 = 0.07 * T
                    # t2 = 0.93 * T
                    # k = 0.004
                    
                    # # print(f"t1: {t1}, t2: {t2}, k: {k}")
                    
                    # def omega(t):
                    #     def safe_exp(x, limit=700):
                    #         return np.exp(np.clip(x, -limit, limit))
                        
                    #     sigmoid1 = 1 / (1 + safe_exp(-k * (t - t1)))
                    #     sigmoid2 = 1 / (1 + safe_exp(-k * (t - t2)))
                    #     envelope = (1 - np.exp(-k * t)) * (1 - np.exp(-k * (T - t)))
                    #     return omega_max * (sigmoid1 - sigmoid2) * envelope
                    
                    # t = np.arange(self.step_counter_internal, self.step_counter_internal + 1000)
                    # # print(f"t array shape: {t.shape}, min: {t.min()}, max: {t.max()}")
                    
                    # velocity = omega(t)
                    # # print(f"velocity array shape: {velocity.shape}")
                    # # print(f"velocity sample values: {velocity[:5]}...{velocity[-5:]}")
                    
                    # r_limit_sigmoid = velocity
                    # r_limit_sigmoid_1000_endList = r_limit_sigmoid
                    
                    # return r_limit_sigmoid_1000_endList

                def obtain_r_limit_sigmoid_1000_endList():
                    """Returns vector of 2000 smoothly decreasing values from t_limit_sigmoid to 0"""
                    t = np.linspace(0, 1, self.overlapping_length)  # 2000 points from 0 to 1
                    
                    def get_decreasing_smooth_limit_vector(phase1_ending_value, t_normalized):
                        """Vectorized version of the decreasing smooth limit"""
                        # Cubic easing function (vectorized)
                        eased_t = t_normalized**2 * (3 - 2*t_normalized)
                        
                        # Invert for decreasing curve
                        inverted_eased_t = 1 - eased_t
                        
                        # Scale by starting value and ensure non-negative
                        return np.maximum(phase1_ending_value * inverted_eased_t, 0.0)
                    
                    # Get the starting value (assuming t_limit_sigmoid is defined in outer scope)2375
                    phase1_ending_value = r_limit_sigmoid  
                    
                    # Compute the full curve
                    r_limit_sigmoid_1000_endList = get_decreasing_smooth_limit_vector(phase1_ending_value, t)
                    
                    return r_limit_sigmoid_1000_endList

                # Usage:
                r_limit_sigmoid_1000_endList = obtain_r_limit_sigmoid_1000_endList()                    
                
                # r_limit_sigmoid_1000_endList = obtain_r_limit_sigmoid_1000_endList()


                # def obtain_r_limit_sigmoid_1000_endList():

                #     t = np.arange(0, 1000)

                #     def get_decreasing_smooth_limit(phase1_ending_value, t):
                #         """Smooth decreasing limit from phase1_ending_value to 0 over 1000 steps"""
                #         total_decay_steps = 1000
                        
                #         # Normalize step counter between 0 and 1
                #         t = min(self.step_counter_internal / total_decay_steps, 1.0)
                        
                #         # Cubic easing function (same as before but decreasing)
                #         eased_t = t**2 * (3 - 2*t)  # This still goes from 0 to 1
                        
                #         # Invert the easing to get a decreasing curve
                #         inverted_eased_t = 1 - eased_t  # Now goes from 1 to 0
                        
                #         # Scale by the starting value
                #         t_limit = phase1_ending_value * inverted_eased_t
                        
                #         # Ensure we don't go below 0
                #         final_limit = max(t_limit, 0.0)
                        
                #         return final_limit

                #     # Usage:
                #     phase1_ending_value = t_limit_sigmoid # 0.5  # Set this to your desired starting value
                #     t_limit_sigmoid_1000_endList = get_decreasing_smooth_limit(phase1_ending_value, t)
                #     # action[0:self.joint_num * 3] = np.clip(action[0:self.joint_num * 3], -t_limit_sigmoid, t_limit_sigmoid)   

                #     # r_limit_sigmoid_1000_endList = r_limit_sigmoid
                    
                #     return t_limit_sigmoid_1000_endList
                
                # t_limit_sigmoid_1000_endList = obtain_r_limit_sigmoid_1000_endList()


                def obtain_t_limit_sigmoid_1000_endList():
                    """Returns vector of 1000 smoothly decreasing values from t_limit_sigmoid to 0"""
                    t = np.linspace(0, 1, self.overlapping_length)  # 2000 points from 0 to 1
                    
                    def get_decreasing_smooth_limit_vector(phase1_ending_value, t_normalized):
                        """Vectorized version of the decreasing smooth limit"""
                        # Cubic easing function (vectorized)
                        eased_t = t_normalized**2 * (3 - 2*t_normalized)
                        
                        # Invert for decreasing curve
                        inverted_eased_t = 1 - eased_t
                        
                        # Scale by starting value and ensure non-negative
                        return np.maximum(phase1_ending_value * inverted_eased_t, 0.0)
                    
                    # Get the starting value (assuming t_limit_sigmoid is defined in outer scope)2375
                    phase1_ending_value = t_limit_sigmoid  
                    
                    # Compute the full curve
                    t_limit_sigmoid_1000_endList = get_decreasing_smooth_limit_vector(phase1_ending_value, t)
                    
                    return t_limit_sigmoid_1000_endList

                # Usage:
                t_limit_sigmoid_1000_endList = obtain_t_limit_sigmoid_1000_endList()
                # self.plot_sigmoid_limit_progression(t_limit_sigmoid_1000_endList, phase_2_flag)
                self.plot_sigmoid_limit_progression(t_limit_sigmoid_1000_endList, current_phase)



                reward = reward + self.reward_win
                done = True
                
                # print("env debug l484")
                success_flag = 1
                print("in env, hit one success. should early stop now. success_flag:", success_flag, "distance:", distance, "min dis:", self.distance_min,
                      "distance_ooc_l:", distance_ooc_l, "min distance_ooc_l:", self.distance_ooc_l_min,
                      "distance_ooc_r:", distance_ooc_r, "min distance_ooc_r:", self.distance_ooc_r_min,
                      "distance_frontalis:", distance_frontalis, "min distance_frontalis:", self.distance_frontalis_min)                
                # print("final degree is: ", angle_degrees_rotation)
                print("final distance_jaw is: ", distance_jaw)
                # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:                                                            
                #     print(" final distance_jaw is:  ", distance_jaw, "distance:", distance, "min dis:", self.distance_min,
                #       "distance_ooc_l:", distance_ooc_l, "min distance_ooc_l:", self.distance_ooc_l_min,
                #       "distance_ooc_r:", distance_ooc_r, "min distance_ooc_r:", self.distance_ooc_r_min,
                #       "distance_frontalis:", distance_frontalis, "min distance_frontalis:", self.distance_frontalis_min, file = f)
                self.log_final_distances(
                    current_phase=current_phase,
                    distance_jaw=distance_jaw,
                    distance=distance,
                    distance_ooc_l=distance_ooc_l,
                    distance_ooc_r=distance_ooc_r,
                    distance_frontalis=distance_frontalis
                )                                    
                
                # self.step_counter_internal = 0
                self.info["TimeLimit.truncated"] = False

                # output_folder = self.save_folder
                # 'D:/keyLabLocal_D/a2-DSAC-T-for-pipelineV1/results/DSAC_V2_m1_vertexMove_Va002/plot'
                # output_file = os.path.join(output_folder, 'normal_verctor_compare_plot.png')            
                # angle_degrees_s_next, angle_degrees_x_bar_j = self.compare_angles(s_jaw_next, self.control_vertices_slice(self.target_jaw_state, self.vertex_jaw_index), self.control_vertices_slice(self.x_jaw_bar_j, self.vertex_jaw_index), output_file)
                # compare_angles(self, s_next, target_state, x_bar_j, output_file):
                # with open(self.save_folder + "/log" + str(self.i_func) + ".txt", 'a') as f:
                #     print(f"Angle between normal vector of s_next and target_state: {angle_degrees_s_next:.4f} degrees", file=f)
                #     print(f"Angle between normal vector of x_bar_j and target_state: {angle_degrees_x_bar_j:.4f} degrees", file=f)

                # print(f"Angle between normal vector of s_next and target_state: {angle_degrees_s_next:.4f} degrees")
                # print(f"Angle between normal vector of x_bar_j and target_state: {angle_degrees_x_bar_j:.4f} degrees")            
                
                # if phase_2_flag == 0:  # we are in phase 1
                #     object_name = "OrbicularisOris"
                    
                #     s_next_full = self.returnCoordinates("middleres_OrbicularisOris")  # , self.vertex_index
                #     self.plot_distances_portal(self.target_state, s_next_full, object_name, self.vertex_index)
                #     object_name = "jaw_down"
                #     s_jaw_next_full = self.returnCoordinates("middleres_Jaw")  # , self.vertex_jaw_index
                #     self.plot_distances_portal(self.target_jaw_state, s_jaw_next_full, object_name, self.vertex_jaw_index)
                #     # object_name = "ooc_l"
                #     # s_ooc_l_next_full = self.returnCoordinates("middleres_OrbicularisOculi_l")  # , self.vertex_ooc_l_index
                #     # self.plot_distances_portal(self.target_ooc_l_state, s_ooc_l_next_full, object_name, self.vertex_ooc_l_index)
                #     # object_name = "ooc_r"
                #     # s_ooc_r_next_full = self.returnCoordinates("middleres_OrbicularisOculi_r")  # , self.vertex_ooc_r_index
                #     # self.plot_distances_portal(self.target_ooc_r_state, s_ooc_r_next_full, object_name, self.vertex_ooc_r_index)        

                #     # object_name = "frontalis"
                #     # s_frontalis_next_full = self.returnCoordinates("middleres_Frontalis")  # , self.vertex_frontalis_index
                #     # self.plot_distances_portal(self.target_frontalis_state, s_frontalis_next_full, object_name, self.vertex_frontalis_index)     

                # else: # we are in phase 2
                    
                #     object_name = "OrbicularisOris_phase2"
                #     s_next_full = self.returnCoordinates("middleres_OrbicularisOris")  # , self.vertex_index
                #     self.plot_distances_portal(self.target_state, s_next_full, object_name, self.vertex_index)
                #     object_name = "jaw_down_phase2"
                #     s_jaw_next_full = self.returnCoordinates("middleres_Jaw")  # , self.vertex_jaw_index
                #     self.plot_distances_portal(self.target_jaw_state, s_jaw_next_full, object_name, self.vertex_jaw_index)
                #     # object_name = "ooc_l"
                #     # s_ooc_l_next_full = self.returnCoordinates("middleres_OrbicularisOculi_l")  # , self.vertex_ooc_l_index
                #     # self.plot_distances_portal(self.target_ooc_l_state, s_ooc_l_next_full, object_name, self.vertex_ooc_l_index)
                #     # object_name = "ooc_r"
                #     # s_ooc_r_next_full = self.returnCoordinates("middleres_OrbicularisOculi_r")  # , self.vertex_ooc_r_index
                #     # self.plot_distances_portal(self.target_ooc_r_state, s_ooc_r_next_full, object_name, self.vertex_ooc_r_index)        

                #     # object_name = "frontalis"
                #     # s_frontalis_next_full = self.returnCoordinates("middleres_Frontalis")  # , self.vertex_frontalis_index
                #     # self.plot_distances_portal(self.target_frontalis_state, s_frontalis_next_full, object_name, self.vertex_frontalis_index)     

                # Replace the entire if-else block with:
                object_name = "OrbicularisOris" + (f"_phase{current_phase}" if current_phase > 1 else "")
                s_next_full = self.returnCoordinates("middleres_OrbicularisOris")
                self.plot_distances_portal(self.target_state, s_next_full, object_name, self.vertex_index)

                object_name = "jaw_down" + (f"_phase{current_phase}" if current_phase > 1 else "")
                s_jaw_next_full = self.returnCoordinates("middleres_Jaw")
                self.plot_distances_portal(self.target_jaw_state, s_jaw_next_full, object_name, self.vertex_jaw_index)

                object_name = "ooc_l"
                s_ooc_l_next_full = self.returnCoordinates("middleres_OrbicularisOculi_l")  # , self.vertex_ooc_l_index
                self.plot_distances_portal(self.target_ooc_l_state, s_ooc_l_next_full, object_name, self.vertex_ooc_l_index)
                object_name = "ooc_r"
                s_ooc_r_next_full = self.returnCoordinates("middleres_OrbicularisOculi_r")  # , self.vertex_ooc_r_index
                self.plot_distances_portal(self.target_ooc_r_state, s_ooc_r_next_full, object_name, self.vertex_ooc_r_index)        

                object_name = "frontalis"
                s_frontalis_next_full = self.returnCoordinates("middleres_Frontalis")  # , self.vertex_frontalis_index
                self.plot_distances_portal(self.target_frontalis_state, s_frontalis_next_full, object_name, self.vertex_frontalis_index)        


                # def plot_adhoc():

                #     def visualize(vertices, object_name):
                #         """
                #         Visualize the vertices, the sampled joint positions, and the surface mesh.

                #         Args:
                #         - vertices (list of lists): List of 3D coordinates of vertices.
                #         - faces (list of lists): List of indices defining the faces of the mesh.
                #         - sampled_points (list of lists): List of 3D coordinates of sampled joint positions.
                #         """
                #         fig = plt.figure()
                #         ax = fig.add_subplot(111, projection='3d')

                #         # Plot the vertices
                #         vertices_np = np.array(vertices)
                #         ax.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2], c='b', marker='o', s=1, label='Vertices')

                #         # Set the view angle
                #         ax.view_init(elev=(180+30), azim=0, roll=90)  # Change these values to set the desired view

                #         ax.set_xlabel('X Label')
                #         ax.set_ylabel('Y Label')
                #         ax.set_zlabel('Z Label')
                #         ax.legend()
                #         # plt.show()
                        

                #         # Save the plot to a file
                #         output_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\e68 backed to e71 no speed up and visualize debug"
                #         output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
                #         plt.savefig(output_file, dpi=300, bbox_inches='tight')
                #         plt.close()        

                #     coords_rest = s_jaw_next_full # np.array(dataV22.x_middleres_OrbicularisOris0)
                    


                #     # call:
                #     # new_coords_for_energy_non_concatenate = [new_coords_for_energy[i:i+3] for i in range(0, len(new_coords_for_energy), 3)]
                #     coords_rest_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]
                #     # x_bar_conca_rest_adhoc = [x_bar_conca_rest_adhoc[i:i+3] for i in range(0, len(x_bar_conca_rest_adhoc), 3)]

                #     # rest_vertices = np.array(dataV22.x_middleres_OrbicularisOris0)
                #     visualize(coords_rest_concatenate, "end_of_phase2_plot_jaw")                    


                # if current_phase == 2:
                #     plot_adhoc()
                #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
                #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
                #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
                #     method = 'euclidean'
                #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_full, self.vertex_jaw_index))
                #     print("at the end of phase 2, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)



                # Commented out sections would follow the same pattern:
                # object_name = "ooc_l" + (f"_phase{current_phase}" if current_phase > 1 else "")
                # s_ooc_l_next_full = self.returnCoordinates("middleres_OrbicularisOculi_l")
                # self.plot_distances_portal(self.target_ooc_l_state, s_ooc_l_next_full, object_name, self.vertex_ooc_l_index)

                calculate_energy_flag = 1
                if calculate_energy_flag  == 1:
                    new_coords_for_energy = self.returnCoordinates("middleres_OrbicularisOris")
                    coords_rest = self.x_bar_j
                    def strain_energy_calculator(new_coords, coords_rest):

                        import numpy as np
                        # import sys
                        sys.path.insert(0, str(a3_stochastic_relative_path / "a1-blendshape-guided-animation-pipeline"/ "database"))
                        import importlib

                        # import dataV22_after_deform
                        # import dataV22
                        # importlib.reload(dataV22_after_deform)
                        # importlib.reload(dataV22)


                        # Material properties (Lamé parameters)
                        mu = 1.0      # Shear modulus
                        lambda_ = 1.0  # Lamé's first parameter

                        # Rest pose: mixed mesh (triangles and quadrangles) in 3D
                        vertices_rest = np.array(coords_rest)

                        # Deformed pose for hybrid method
                        vertices_deformed_hybrid = np.array(new_coords)

                        # Deformed pose for pure kinematic method
                        # vertices_deformed_kinematic = np.array(dataV22.x_middleres_OrbicularisOris1)

                        # Elements: mix of triangles and quadrangles (vertex indices)
                        # elements = dataV22.x_middleres_OrbicularisOris_face_0
                        elements = self.elements

                        # Function to split quadrangles into triangles
                        def split_quadrangle(element):
                            # Split quadrangle into two triangles: [v0, v1, v2], [v0, v2, v3]
                            return [[element[0], element[1], element[2]], [element[0], element[2], element[3]]]

                        # Function to compute deformation gradient F and area A
                        def compute_F_and_A(vertices_rest, vertices_deformed, element):
                            # Rest pose edges
                            X = vertices_rest[element]
                            E1_rest = X[1] - X[0]
                            E2_rest = X[2] - X[0]
                            
                            # Deformed pose edges
                            x = vertices_deformed[element]
                            E1_deformed = x[1] - x[0]
                            E2_deformed = x[2] - x[0]
                            
                            # Deformation gradient F (3x3 matrix, approximate plane deformation)
                            F = np.column_stack((E1_deformed, E2_deformed, np.cross(E1_deformed, E2_deformed))) @ \
                                np.linalg.inv(np.column_stack((E1_rest, E2_rest, np.cross(E1_rest, E2_rest))))
                            
                            # Area of the triangle (half the magnitude of the cross product of the two edges)
                            area = 0.5 * np.linalg.norm(np.cross(E1_rest, E2_rest))
                            
                            return F, area

                        # Strain energy density function (St. Venant-Kirchhoff, 3D)
                        def strain_energy_density(F):
                            C = F.T @ F  # Right Cauchy-Green tensor
                            E = 0.5 * (C - np.eye(3))  # Green-Lagrange strain tensor
                            trace_E = np.trace(E)
                            return 0.5 * lambda_ * (trace_E ** 2) + mu * np.sum(E ** 2)

                        # Total energy calculation
                        def compute_total_energy(vertices_rest, vertices_deformed, elements):
                            total_energy = 0
                            for element in elements:
                                if len(element) == 4:  # Quadrangle
                                    # Split quadrangle into two triangles
                                    triangles = split_quadrangle(element)
                                else:
                                    # Single triangle
                                    triangles = [element]
                                
                                for triangle in triangles:
                                    F, A = compute_F_and_A(vertices_rest, vertices_deformed, triangle)
                                    psi = strain_energy_density(F)
                                    total_energy += psi * A
                            return total_energy

                        # Compute energies
                        # energy_rest = compute_total_energy(vertices_rest, vertices_rest, elements)
                        energy_hybrid = compute_total_energy(vertices_rest, vertices_deformed_hybrid, elements)
                        # energy_kinematic = compute_total_energy(vertices_rest, vertices_deformed_kinematic, elements)

                        # Print results
                        # print(f"Energy at rest: {energy_rest}")
                        # print(f"Energy for hybrid method: {energy_hybrid}")
                        # print(f"Energy for half pure kinematic method: {energy_kinematic}")

                        return energy_hybrid # energy_rest, 

                    # call:
                    new_coords_for_energy_non_concatenate = [new_coords_for_energy[i:i+3] for i in range(0, len(new_coords_for_energy), 3)]
                    coords_rest_non_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]
                    # strain_energy = strain_energy_calculator(new_coords_for_energy_non_concatenate, coords_rest_non_concatenate)


                # calculate the kinetic energy.  rotational_velocity_average  r_limit_sigmoid
                if self.fix_flag_ship[1] == 1: # rotation has stopped, kinetic energy to 0.
                    rotation_velocity = 0
                    rotational_kinetic_energy = 0     
                else:                                   
                    rotation_velocity =  rotational_velocity_average # / (  (35/30) /self.expected_total_steps_for_jaw) # unit: degree per second
                    rotation_velocity_prime = rotation_velocity * math.pi / 180  # unit: rad/s
                    I = 1/3 * 1.5 * 0.17**2  # jaw weighs 1.5kg, Radius of Rotation: 17 cm (from the center of rotation to the middle teeth).
                    E_rotation = 1/2 * I * rotation_velocity_prime**2  # equal to "The work (physical terminology) done by the muscles"
                    rotational_kinetic_energy = E_rotation
                    convert_factor = 1.0  # 10e-3  if convert J to mJ 
                    rotational_kinetic_energy = rotational_kinetic_energy * convert_factor

                # self.update_expected_total_steps(distance_temp, threshold=self.distance_min  + epsilon_min)

                    # After calculating rotation_velocity, add:
                    # print(f"\n=== ROTATION VELOCITY DEBUG ===")
                    # print(f"Phase: {'2' if phase_2_flag else '1'}")
                    # print(f"Current step: {self.step_counter_internal}")
                    # print(f"Action segment (rotation part): {action[self.joint_num*3:self.joint_num*3+self.joint_rotate_num*3]}")
                    # print(f"Elements extracted: {elements}")
                    # print(f"Raw average: {sum(elements)/len(elements)}")
                    # print(f"Time factor: {(35/30)/self.expected_total_steps_for_jaw}")
                    # print(f"Final rotation_velocity: {rotation_velocity}")
                    # print(f"r_limit_sigmoid: {r_limit_sigmoid}")
                    # print(f"velocity_rad: {velocity_rad}")


                current_time = time.time()
                elapsed_time = current_time - self.start_time

                # if phase_2_flag == 1:
                #     self.distance_log_phase2.append(distance)
                #     self.step_counter_internal_log_phase2.append(self.step_counter_internal)
                # else:
                #     self.distance_log.append(distance)
                #     self.step_counter_internal_log.append(self.step_counter_internal)

                # self.distance_log.append(distance)
                # self.time_log.append(elapsed_time)

                # self.time_for_energy_log.append(elapsed_time)
                # self.step_counter_internal_log.append(self.step_counter_internal)

                # self.strain_energy.append(strain_energy)

                # self.time_for_rotational_kinetic_energy_log.append(elapsed_time)




                # calculate the accumulated rotated angle
                # def calculate_accumulated_rotated_angle():
                    
                #     final_epoch_action_list = self.action_real_list
                #     delta_b0j = final_epoch_action_list
                #     delta_b0j_conca = delta_b0j

                #     delta_b0_conca = [0] * 3*self.joint_num + [0] * 3 * self.joint_rotate_num  + [0] * 3 *self.joint_ooc_l_num  + [0] * 3 * self.joint_ooc_r_num  + [0] * 3 * self.joint_frontalis_num
                #     # if next_info['success_flag'] == 1:
                #     #     print("in evaluator l133, hit one success. should early stop now. next_info['success_flag']:", next_info['success_flag'])            
                    
                #     for i in range(len(delta_b0j_conca)):
                #         delta_b0_conca = [delta_b0_conca[j] + delta_b0j_conca[i][j] for j in range(3*self.joint_num + 3*self.joint_rotate_num + 3 *self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num)]
                #     return delta_b0_conca

                # delta_b0_conca = calculate_accumulated_rotated_angle()

                delta_b0_conca = self._delta_b0_conca_accumulator

                # if phase_2_flag == 1:
                #     self.accumulated_rotated_angle_phase2.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])
                # else:
                #     self.accumulated_rotated_angle.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])
                
                # Store angles in phase-specific lists
                # angle_storage = {
                #     1: self.accumulated_rotated_angle,
                #     2: self.accumulated_rotated_angle_phase2
                # }.get(current_phase, self.accumulated_rotated_angle)  # Default to phase 1 storage

                # angle_storage.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])

                # angle_storage = self._get_angle_storage(current_phase)
                # angle_storage.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])

                # if self.this_task in self.contract_to_neutral_task:
                #     theta = abs(self.rest_pose_rotational_position_degree) - abs(self._get_angle_storage(current_phase)[-1][0])
                # else:
                #     theta = self._get_angle_storage(current_phase)[-1][0]

                # calculate the elastic energy incurred by the rotation of jaw.
                # theta = cumulative_trapezoid(velocity_rad, t, initial=0)
                # base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]


                # if self.this_task in self.contract_to_neutral_task:
                #     theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]
                # else:
                #     if phase_2_flag == 0:
                #         theta = self.accumulated_rotated_angle[-1][0]
                #     else:                        
                #         theta = self.accumulated_rotated_angle_phase2[-1][0]
                    
                # if self.this_task in self.contract_to_neutral_task:
                #     theta = abs(self.rest_pose_rotational_position_degree) - self._get_angle_storage(current_phase)[-1][0]
                # else:
                #     # Get the appropriate angle storage based on current phase
                #     current_angle_storage = {
                #         1: self.accumulated_rotated_angle,
                #         2: self.accumulated_rotated_angle_phase2
                #     }.get(current_phase, self.accumulated_rotated_angle)  # Default to phase 1
                    
                #     theta = current_angle_storage[-1][0]

                phase_logs = self._get_phase_logs(current_phase)
                phase_logs['angles'].append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])  

                # calculate the elastic energy incurred by the rotation of jaw.
                # theta = cumulative_trapezoid(velocity_rad, t, initial=0)
                # base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]
                # theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]
                # if self.this_task in self.contract_to_neutral_task:
                #     theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]
                # else:
                #     theta = self.accumulated_rotated_angle[-1][0]


                # angle_storage = self._get_angle_storage(current_phase)
                # angle_storage.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])


                if False:
                # if self.this_task in self.contract_to_neutral_task:
                    # theta = abs(self.rest_pose_rotational_position_degree) - abs(phase_logs['angles'][-1][0])
                    pass
                else:
                    theta = phase_logs['angles'][-1][0]


                def degree_conver_to_rad(theta):
                    return theta * np.pi / 180  # Convert °/s to rad/s
                theta_rad = degree_conver_to_rad(theta)
                k_tissue = 10000  # Stiffness of soft tissues (N/m)
                r = 0.17  # Radius of rotation (m)        
                # Calculate displacement (x) as r * theta
                x = r * theta_rad        

                # Elastic energy function
                def elastic_energy():
                    return 0.5 * k_tissue * x**2
                
                E_elastic = elastic_energy()
                



                # self.rotational_kinetic_energy.append(rotational_kinetic_energy)  
                # self.elastic_energy.append(E_elastic)  
                # self.rotation_velocity.append(rotation_velocity)
                # rotation_velocity = rotation_velocity


                # # update the logging:
                # if phase_2_flag == 1:
                #     self.distance_log_phase2.append(distance)
                #     self.step_counter_internal_log_phase2.append(self.step_counter_internal)
                #     # self.strain_energy_phase2.append(strain_energy)
                #     self.time_for_energy_log_phase2.append(elapsed_time)
                #     self.rotational_kinetic_energy_phase2.append(rotational_kinetic_energy)
                #     self.time_for_rotational_kinetic_energy_log_phase2.append(elapsed_time)
                #     self.elastic_energy_phase2.append(E_elastic)
                #     self.rotation_velocity_phase2.append(rotation_velocity)
                #     self.base_action_prime_log_phase2.append(base_action_prime)
                # else:
                #     self.distance_log.append(distance)
                #     self.step_counter_internal_log.append(self.step_counter_internal)
                #     # self.strain_energy.append(strain_energy)
                #     self.time_for_energy_log.append(elapsed_time)
                #     self.rotational_kinetic_energy.append(rotational_kinetic_energy)
                #     self.time_for_rotational_kinetic_energy_log.append(elapsed_time)
                #     self.elastic_energy.append(E_elastic)
                #     self.rotation_velocity.append(rotation_velocity)
                #     self.base_action_prime_log.append(base_action_prime)

                # Replace the entire if-else block with:
                phase_logs = self._get_phase_logs(current_phase)

                phase_logs['distance'].append(distance)
                # phase_logs['distance'].append(distance_ooc_r)
                phase_logs['step_counter_internal'].append(self.step_counter_internal)
                phase_logs['time_for_energy'].append(elapsed_time)
                phase_logs['rotational_kinetic_energy'].append(rotational_kinetic_energy)
                phase_logs['time_for_rotational_kinetic_energy'].append(elapsed_time)
                phase_logs['elastic_energy'].append(E_elastic)
                phase_logs['rotation_velocity'].append(rotation_velocity)
                phase_logs['base_action_prime'].append(base_action_prime)

                

                # self.data_save_distance_vs_time()
                # self.data_save_energy_vs_time()
                # self.data_save_rotational_kinetic_energy_vs_time()
                # self.data_save_elastic_energy_vs_time_plot()
                # self.data_save_accumulated_work_done()						
                
                # Then call the plotting/saving functions with phase flag
                # self.data_save_distance_vs_time(phase_2_flag)
                # self.data_save_energy_vs_time(phase_2_flag)
                # self.data_save_rotational_kinetic_energy_vs_time(phase_2_flag)
                # self.data_save_elastic_energy_vs_time_plot(phase_2_flag)
                # self.data_save_accumulated_work_done(phase_2_flag)

                self.save_all_phase_data(current_phase)

                # self.plot_distance_vs_time_plot()
                # self.plot_distance_vs_time_plot(phase_2_flag)

                self.plot_distance_vs_time_plot(current_phase)



                # self.plot_energy_vs_time_plot()
                
                # self.plot_rotational_kinetic_energy_vs_time_plot()

                # self.plot_elastic_energy_vs_time_plot()

                # total accumulated work done
                # self.plot_accumulated_work_done()
                # self.plot_energy_vs_time_plot(phase_2_flag)
                # self.plot_rotational_kinetic_energy_vs_time_plot(phase_2_flag)
                # self.plot_angular_velocity_vs_time_plot(phase_2_flag)

                # self.plot_elastic_energy_vs_time_plot(phase_2_flag)
                # self.plot_accumulated_work_done(phase_2_flag)                

                # # self.plot_sigmoid_limit_progression(phase_2_flag)
                # self.plot_true_translation_elements(phase_2_flag)       


                self.plot_rotational_kinetic_energy_vs_time_plot(current_phase)
                self.plot_angular_velocity_vs_time_plot(current_phase)
                self.plot_elastic_energy_vs_time_plot(current_phase)
                self.plot_accumulated_work_done(current_phase)
                self.plot_true_translation_elements(current_phase)

                self.fix_flag_ship = [0] * (self.joint_num + self.joint_rotate_num  + self.joint_ooc_l_num + self.joint_ooc_r_num + self.joint_frontalis_num)

                self.fix_r_limit_flag = 0
                self.max_velocity_achieved = 0
                self.step_counter = 0

                self.cmds.select(cl=True)
                self.cmds.currentCtx()
                # Close the reference                
                # print("l858 self.opened_file", self.opened_file)
                # self.cmds.file(self.opened_file, removeReference=True)  
                self.opened_file_status = False
                                  
                
            else:
                # reward = reward + self.idling_punish
                # reward = self.idling_punish
                # object_name = "OrbicularisOris"
                object_name = "OrbicularisOris" + (f"_phase{current_phase}" if current_phase > 1 else "")

                

                # Increment the counter
                self.plot_counter += 1                
                
                # Plot only once every 500 iterations
                if self.plot_counter % self.plot_debug_every == 0 or self.plot_counter == 1:
                    s_next_full = self.returnCoordinates("middleres_OrbicularisOris")  # , self.vertex_index
                    self.plot_distances_portal(self.target_state, s_next_full, object_name, self.vertex_index)
                
                # self.plot_distances_portal(self.target_state, s_next_full, object_name, self.vertex_index)

                pass
                done = False 





        if self.step_counter_internal >= self.step_counter_internal_max:
            # reward = -1.0
            # print("env debug l499")
            if success_flag == 1:
                print("in env l526, hit one success. should early stop now. success_flag:", success_flag)            
            else:
                print("in env l526, failed. reached step_counter_internal_max, should early stop now. flag:", success_flag)            
            done = True   
            self.fix_flag_ship = [0] * (self.joint_num + self.joint_rotate_num  + self.joint_ooc_l_num + self.joint_ooc_r_num + self.joint_frontalis_num)
            
            success_flag = 0  
            # self.step_counter_internal = 0
            self.info["TimeLimit.truncated"] = False
            # self.plot_distances_portal(self.target_state, s_next)
            
            # self.render_distances_portal()
            self.cmds.select(cl=True)
            self.cmds.currentCtx()
            # Close the reference        
            # print("l927 self.opened_file", self.opened_file)        
            # self.cmds.file(self.opened_file, removeReference=True)                    
            self.opened_file_status = False


        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # print("current_time", current_time)
        # print("self.start_time", self.start_time)
        # print("elapsed_time", self.start_time)        
        # print("distance", distance)

        # Log the distance and the corresponding timestamp every 1000 iterations

        # self.log_plot_every = 5
        if self.step_counter_internal % self.log_plot_every == 0:
            # current_time = time.time() - start_time  # Calculate elapsed time
            # if phase_2_flag == 0: # type: ignore
            #     self.distance_log.append(distance)
            # else:
            #     self.distance_log_phase2.append(distance)
            # self.time_log.append(elapsed_time)

            phase_logs = self._get_phase_logs(current_phase)
            # phase_logs['distance'].append(distance_ooc_r)
            phase_logs['distance'].append(distance)




        if self.step_counter_internal % self.log_plot_every == 0:
            # current_time = time.time() - start_time  # Calculate elapsed time

            
            # if phase_2_flag == 0:
            #     self.time_for_energy_log.append(elapsed_time)
            # else:
            #     self.time_for_energy_log_phase2.append(elapsed_time)
            # self.strain_energy.append(strain_energy)
            phase_logs = self._get_phase_logs(current_phase)
            phase_logs['time_for_energy'].append(elapsed_time)            


        if self.step_counter_internal % self.log_plot_every == 0:
            # current_time = time.time() - start_time  # Calculate elapsed time

            
            # if phase_2_flag == 0:
            #     self.time_for_rotational_kinetic_energy_log.append(elapsed_time)
            #     self.base_action_prime_log.append(base_action_prime)
            # else:
            #     self.time_for_rotational_kinetic_energy_log_phase2.append(elapsed_time)
            #     self.base_action_prime_log_phase2.append(base_action_prime)
            phase_logs = self._get_phase_logs(current_phase)
            phase_logs['time_for_rotational_kinetic_energy'].append(elapsed_time)  
            phase_logs['base_action_prime'].append(base_action_prime)   # note that we should not use the action list from this phase_logs: it only loged every "self.log_plot_every", not all of them.
            

            # def calculate_accumulated_rotated_angle():
            #     # calculate the accumulated rotated angle
            #     final_epoch_action_list = self.action_real_list
            #     delta_b0j = final_epoch_action_list
            #     delta_b0j_conca = delta_b0j

            #     delta_b0_conca = [0] * 3*self.joint_num + [0] * 3 * self.joint_rotate_num  + [0] * 3 *self.joint_ooc_l_num  + [0] * 3 * self.joint_ooc_r_num  + [0] * 3 * self.joint_frontalis_num
            #     # if next_info['success_flag'] == 1:
            #     #     print("in evaluator l133, hit one success. should early stop now. next_info['success_flag']:", next_info['success_flag'])            
                
            #     for i in range(len(delta_b0j_conca)):
            #         delta_b0_conca = [delta_b0_conca[j] + delta_b0j_conca[i][j] for j in range(3*self.joint_num + 3*self.joint_rotate_num + 3 *self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num)]
            #     return delta_b0_conca

            # delta_b0_conca = calculate_accumulated_rotated_angle()
            delta_b0_conca = self._delta_b0_conca_accumulator

            # self.accumulated_rotated_angle.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])
            # if phase_2_flag == 1:
            #     self.accumulated_rotated_angle_phase2.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])
            # else:
            #     self.accumulated_rotated_angle.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])
            phase_logs = self._get_phase_logs(current_phase)
            phase_logs['angles'].append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])  

            

            # def debug_rotation_angle_speedup():
            #     delta_b0_conca = self.calculate_accumulated_rotated_angle()
            #     rotated_angles = delta_b0_conca[self.n : self.n + 3 * self.joint_rotate_num]
            #     self.accumulated_rotated_angle.append(rotated_angles)            


            # debug_rotation_angle_speedup()

            # calculate the elastic energy incurred by the rotation of jaw.
            # theta = cumulative_trapezoid(velocity_rad, t, initial=0)
            # base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]
            # theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]
            # if self.this_task in self.contract_to_neutral_task:
            #     theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]
            # else:
            #     theta = self.accumulated_rotated_angle[-1][0]


            # angle_storage = self._get_angle_storage(current_phase)
            # angle_storage.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])

            # if self.this_task in self.contract_to_neutral_task:
            if False:
                # theta = abs(self.rest_pose_rotational_position_degree) - abs(phase_logs['angles'][-1][0])
                pass
            else:
                theta = phase_logs['angles'][-1][0]


            # if self.this_task in self.contract_to_neutral_task:
            #     theta = abs(self.rest_pose_rotational_position_degree) - phase_logs['angles'][-1][0]
            # else:
            #     if phase_2_flag == 0:
            #         theta = self.accumulated_rotated_angle[-1][0]
            #     else:                        
            #         theta = self.accumulated_rotated_angle_phase2[-1][0]

            def degree_conver_to_rad(theta):
                return theta * np.pi / 180  # Convert ° to rad
            theta_rad = degree_conver_to_rad(theta)
            k_tissue = 10000  # Stiffness of soft tissues (N/m)
            r = 0.17  # Radius of rotation (m)        
            # Calculate displacement (x) as r * theta
            x = r * theta_rad  

            # Elastic energy function
            def elastic_energy():
                return 0.5 * k_tissue * x**2
            
            E_elastic = elastic_energy()  #


            # self.rotational_kinetic_energy.append(rotational_kinetic_energy)
            # self.elastic_energy.append(E_elastic)
            # self.rotation_velocity.append(rotation_velocity)
            # if phase_2_flag == 0:
            #     # Phase 1 logging
            #     self.rotational_kinetic_energy.append(rotational_kinetic_energy)
            #     self.elastic_energy.append(E_elastic)
            #     self.rotation_velocity.append(rotation_velocity)
            #     self.step_counter_internal_log.append(self.step_counter_internal)
            # else:
            #     # Phase 2 logging - initialize lists if they don't exist
            #     # if not hasattr(self, 'rotational_kinetic_energy_phase2'):
            #     #     self.rotational_kinetic_energy_phase2 = []
            #     #     self.elastic_energy_phase2 = []
            #     #     self.rotation_velocity_phase2 = []
            #     #     self.step_counter_internal_log_phase2 = []
                
            #     # Phase 2 data collection
            #     self.rotational_kinetic_energy_phase2.append(rotational_kinetic_energy)
            #     self.elastic_energy_phase2.append(E_elastic)
            #     self.rotation_velocity_phase2.append(rotation_velocity)
            #     self.step_counter_internal_log_phase2.append(self.step_counter_internal)

            phase_logs = self._get_phase_logs(current_phase)

            # phase_logs['distance'].append(distance)
            phase_logs['step_counter_internal'].append(self.step_counter_internal)
            # phase_logs['time_for_energy'].append(elapsed_time)
            phase_logs['rotational_kinetic_energy'].append(rotational_kinetic_energy)
            # phase_logs['time_for_rotational_kinetic_energy'].append(elapsed_time)
            phase_logs['elastic_energy'].append(E_elastic)
            phase_logs['rotation_velocity'].append(rotation_velocity)
            # phase_logs['base_action_prime'].append(base_action_prime)
            # self.step_counter_internal_log.append(self.step_counter_internal)

        # else:
        #     delta_b0_conca = None

        # if self.step_counter_internal % self.log_plot_every == 0 and len(self.distance_log)!= 0:
        #     self.data_save_distance_vs_time()

        # if delta_b0_conca is None:
        #     self.debug_rotation_angle_speedup(current_phase)
        # else:
        #     self.debug_rotation_angle_speedup(current_phase, delta_b0_conca)

        
        # self.debug_rotation_angle_speedup(current_phase, delta_b0_conca = None)        

        # Then call the plotting/saving functions with phase flag
        if True:  # phase_2_flag == 0
            # Phase 1 condition - check regular energy lists
            if (self.step_counter_internal % self.log_plot_every == 0 and 
                len(phase_logs['rotational_kinetic_energy']) != 0):

        
                # self.data_save_distance_vs_time(phase_2_flag)
                # self.data_save_energy_vs_time(phase_2_flag)
                # self.data_save_rotational_kinetic_energy_vs_time(phase_2_flag)
                # self.data_save_elastic_energy_vs_time_plot(phase_2_flag)
                # self.data_save_accumulated_work_done(phase_2_flag)
                
                self.data_save_distance_vs_time(current_phase)
                # self.data_save_energy_vs_time(current_phase)
                self.data_save_angular_velocity_vs_time(current_phase)
                self.data_save_rotational_kinetic_energy_vs_time(current_phase)
                self.data_save_elastic_energy_vs_time_plot(current_phase)
                self.data_save_accumulated_work_done(current_phase)    

                if self.plot_distance_figures_flag == 1:
                    # self.plot_distance_vs_time_plot(phase_2_flag)
                    self.plot_distance_vs_time_plot(current_phase)
                if self.plot_4_figures_flag == 1:
                    # self.plot_energy_vs_time_plot(phase_2_flag)
                    # self.plot_rotational_kinetic_energy_vs_time_plot(phase_2_flag)
                    # self.plot_angular_velocity_vs_time_plot(phase_2_flag)

                    # self.plot_elastic_energy_vs_time_plot(phase_2_flag)
                    # self.plot_accumulated_work_done(phase_2_flag)
             
                    self.plot_rotational_kinetic_energy_vs_time_plot(current_phase)
                    self.plot_angular_velocity_vs_time_plot(current_phase)
                    self.plot_elastic_energy_vs_time_plot(current_phase)
                    self.plot_accumulated_work_done(current_phase)
                    


        # else:
        #     # Phase 2 condition - check phase2 energy lists
        #     if (self.step_counter_internal % self.log_plot_every == 0 and 
        #         len(self.rotational_kinetic_energy_phase2) != 0):

        #         self.data_save_distance_vs_time(phase_2_flag)
        #         # self.data_save_energy_vs_time(phase_2_flag)
        #         self.data_save_rotational_kinetic_energy_vs_time(phase_2_flag)
        #         self.data_save_elastic_energy_vs_time_plot(phase_2_flag)
        #         self.data_save_accumulated_work_done(phase_2_flag)
                
        #         if self.plot_distance_figures_flag == 1:
        #             self.plot_distance_vs_time_plot(phase_2_flag)
        #         if self.plot_4_figures_flag == 1:
        #             # self.plot_energy_vs_time_plot(phase_2_flag)
        #             self.plot_rotational_kinetic_energy_vs_time_plot(phase_2_flag)
        #             self.plot_angular_velocity_vs_time_plot(phase_2_flag)
        #             self.plot_elastic_energy_vs_time_plot(phase_2_flag)
        #             self.plot_accumulated_work_done(phase_2_flag)

        # else:
        #     # Phase 2 condition - check phase2 energy lists
        #     # print(f"\n=== PHASE 2 DEBUG === Step: {self.step_counter_internal} ===")
        #     # print(f"log_plot_every: {self.log_plot_every}")
        #     # print(f"rotational_kinetic_energy_phase2 length: {len(self.rotational_kinetic_energy_phase2)}")
        #     # print(f"step_counter_internal_log_phase2: {self.step_counter_internal_log_phase2[-5:] if len(self.step_counter_internal_log_phase2) > 0 else 'EMPTY'}")
            
        #     if (self.step_counter_internal % self.log_plot_every == 0 and 
        #         len(self.rotational_kinetic_energy_phase2) != 0):
                
        #         # print("\n=== CALLING SAVE FUNCTIONS ===")
        #         # print(f"Before data_save_rotational_kinetic_energy_vs_time - data length: {len(self.rotational_kinetic_energy_phase2)}")
                
        #         self.data_save_rotational_kinetic_energy_vs_time(phase_2_flag)
                
        #         # Check file contents after saving
        #         try:
        #             with open(os.path.join(self.save_folder, "Rotational_kineticenergy_data_phase2.csv"), 'r') as f:
        #                 lines = f.readlines()
        #                 print(f"File has {len(lines)} lines (header + {len(lines)-1} data points)")
        #         except FileNotFoundError:
        #             print("Rotational kinetic energy file not found after saving!")                

        # # Then call the plotting/saving functions with phase flag
        # if self.step_counter_internal % self.log_plot_every == 0 and len(self.rotational_kinetic_energy)!= 0:
        #     self.data_save_distance_vs_time(phase_2_flag)
        #     # self.data_save_energy_vs_time(phase_2_flag)
        #     self.data_save_rotational_kinetic_energy_vs_time(phase_2_flag)
        #     self.data_save_elastic_energy_vs_time_plot(phase_2_flag)
        #     self.data_save_accumulated_work_done(phase_2_flag)
            
        #     if self.plot_distance_figures_flag == 1:
        #         self.plot_distance_vs_time_plot(phase_2_flag)
        #     if self.plot_4_figures_flag == 1:
        #         # self.plot_energy_vs_time_plot(phase_2_flag)
        #         self.plot_rotational_kinetic_energy_vs_time_plot(phase_2_flag)
        #         self.plot_elastic_energy_vs_time_plot(phase_2_flag)
        #         self.plot_accumulated_work_done(phase_2_flag)
            

        # if self.step_counter_internal % self.log_plot_every == 0 and len(self.strain_energy)!= 0:
        #     self.data_save_energy_vs_time()
        #     if self.plot_4_figures_flag == 1:
        #         self.plot_energy_vs_time_plot()

        # if self.step_counter_internal % self.log_plot_every == 0  and len(self.rotational_kinetic_energy)!= 0:
        #     self.data_save_rotational_kinetic_energy_vs_time()
        #     self.data_save_elastic_energy_vs_time_plot()
        #     self.data_save_accumulated_work_done()

        #     if self.plot_4_figures_flag == 1:
        #         self.plot_rotational_kinetic_energy_vs_time_plot()
        #         self.plot_elastic_energy_vs_time_plot()
        #         self.plot_accumulated_work_done()


        if self.step_counter_internal % 50  == 0  and self.step_counter_internal!= 0:
            epsilon_min = 0.01
            self.update_expected_total_steps(distance_temp, threshold=self.distance_min  + epsilon_min)            
            



        # print("self.distance_log", self.distance_log)
        # print("self.time_log", self.time_log)

        # if elapsed_time > self.time_limit:
        #     done = True
        #     success_flag = 0 
        #     with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
        #         print("elapsed_time > self.time_limit, terminate now. reward is:", reward, file = f)            
            
        #     self.info["TimeLimit.truncated"] = False
        #     self.plot_distances_portal(self.target_state, s_next)

        # if distance >= 20 * (self.distance_min/self.distance_threshold_percentage) or distance_jaw >= 5 * (self.distance_jaw_min/self.distance_jaw_threshold_percentage):        
        # 5  to 20


        epsilon_min = 0.01
        if "both" == self.action_mode:
            if distance > 20 * (self.distance_min/self.distance_threshold_percentage) + epsilon_min or distance_ooc_l > 20 * (self.distance_ooc_l_min/self.distance_threshold_percentage) + epsilon_min or distance_ooc_r > 20 * (self.distance_ooc_r_min/self.distance_threshold_percentage) + epsilon_min or distance_jaw >= 20 * (self.distance_jaw_min/self.distance_jaw_threshold_percentage)  + epsilon_min: #  or distance_jaw >= 20 * (self.distance_jaw_min/self.distance_jaw_threshold_percentage), distance >= 20 * (self.distance_min/self.distance_threshold_percentage) or 
            
                # reward = reward + self.outsider_punish
            
                # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                #     print("trigerred outsider_punish, reward is: ", reward, file = f)
                #     print("outsider distance is:", distance, file = f)
                #     print("outsider distance_jaw is:", distance_jaw, file = f)
                if success_flag == 1:
                    print("in env l472, hit one success. should early stop now. success_flag:", success_flag)            
                done = True   
                self.fix_flag_ship = [0] * (self.joint_num + self.joint_rotate_num  + self.joint_ooc_l_num + self.joint_ooc_r_num + self.joint_frontalis_num)
                success_flag = 0  
                # self.step_counter_internal = 0
                self.info["TimeLimit.truncated"] = False
                # self.plot_distances_portal(self.target_state, s_next)
                self.cmds.select(cl=True)
                self.cmds.currentCtx()
                # Close the reference 
                # print("l960 self.opened_file", self.opened_file)               
                # self.cmds.file(self.opened_file, removeReference=True)                    
                self.opened_file_status = False



        # if "rotation" == self.action_mode:
        #     if distance_jaw >= 20 * (self.distance_jaw_min/self.distance_jaw_threshold_percentage):
            
        #         # reward = reward + self.outsider_punish
            
        #         with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
        #             print("trigerred outsider_punish, reward is: ", reward, file = f)
        #             print("outsider distance is:", distance, file = f)
        #             print("outsider distance_jaw is:", distance_jaw, file = f)
        #         if success_flag == 1:
        #             print("in env l472, hit one success. should early stop now. success_flag:", success_flag)            
        #         done = True   
        #         self.fix_flag_ship = [0] * (self.joint_num + self.joint_rotate_num  + self.joint_ooc_l_num + self.joint_ooc_r_num)
        #         success_flag = 0  
        #         # self.step_counter_internal = 0
        #         self.info["TimeLimit.truncated"] = False
        #         # self.plot_distances_portal(self.target_state, s_next)
        #         self.cmds.select(cl=True)
        #         self.cmds.currentCtx()
        #         # Close the reference 
        #         # print("l960 self.opened_file", self.opened_file)               
        #         # self.cmds.file(self.opened_file, removeReference=True)                    
        #         self.opened_file_status = False


        # if "translation" == self.action_mode:
            # if distance >= 20 * (self.distance_min/self.distance_threshold_percentage):
            
            #     # reward = reward + self.outsider_punish
            
            #     with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
            #         print("trigerred outsider_punish, reward is: ", reward, file = f)
            #         print("outsider distance is:", distance, file = f)
            #         print("outsider distance_jaw is:", distance_jaw, file = f)
            #     if success_flag == 1:
            #         print("in env l472, hit one success. should early stop now. success_flag:", success_flag)            
            #     done = True   
            #     self.fix_flag_ship = [0] * (self.joint_num + self.joint_rotate_num  + self.joint_ooc_l_num + self.joint_ooc_r_num)
            #     success_flag = 0  
            #     # self.step_counter_internal = 0
            #     self.info["TimeLimit.truncated"] = False
            #     # self.plot_distances_portal(self.target_state, s_next)
            #     self.cmds.select(cl=True)
            #     self.cmds.currentCtx()
            #     # Close the reference 
            #     # print("l960 self.opened_file", self.opened_file)               
            #     # self.cmds.file(self.opened_file, removeReference=True)                    
            #     self.opened_file_status = False




        self.x_joint = x_joint_
        self.x_joint_rotation = x_joint_rotation_ 
        self.x_ooc_l_joint = x_ooc_l_joint_
        self.x_ooc_r_joint = x_ooc_r_joint_
        self.x_frontalis_joint = x_frontalis_joint_ 


        self.step_counter_internal += 1


        if success_flag == 1:
            print("in env l646, hit one success. should early stop now. ") # self.info['success_flag']:", self.info['success_flag']     

            # if current_phase == 2:
            #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
            #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
            #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
            #     method = 'euclidean'
            #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_full, self.vertex_jaw_index))
            #     print("m1 line 3535, at the end of phase 2, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)

            s_next_full, s_jaw_next_full, s_ooc_l_next_full, s_ooc_r_next_full, s_frontalis_next_full = self.suc_position_save()

            # if current_phase == 2:
            #     x_jaw = getattr(dataV22_phoneme_T, "x_middleres_Jaw" + str(1))
            #     x_jaw_conca = [item for sublist in x_jaw for item in sublist]
            #     target_jaw_state_adhoc = np.asarray(x_jaw_conca)
            #     method = 'euclidean'
            #     distance_jaw_adhoc = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_jaw_state_adhoc, self.vertex_jaw_index), self.control_vertices_slice(s_jaw_next_full, self.vertex_jaw_index))
            #     print("m1 line 3544, at the end of phase 2, the jaw pose and the phoneme T distance is", distance_jaw_adhoc)



            self.info.update({
                's_next_full': s_next_full,
                's_jaw_next_full': s_jaw_next_full,            
                's_ooc_l_next_full': s_ooc_l_next_full,
                's_ooc_r_next_full': s_ooc_r_next_full,
                's_frontalis_next_full': s_frontalis_next_full,       
                                
            })


        self.info.update({
            'success_flag': success_flag,
            # 'fix_flag_ship': self.fix_flag_ship,
            'step_counter': self.step_counter,
            'base_action_prime': base_action_prime,
            'x_joint': self.x_joint,
            'x_ooc_l_joint': self.x_ooc_l_joint,
            'x_ooc_r_joint': self.x_ooc_r_joint,
            'x_frontalis_joint': self.x_frontalis_joint,
            'print_allow': self.print_allow,
            'i_count': self.i_count,
            'step_counter_internal': self.step_counter_internal,
            'distance': distance_,
            'distance_jaw': distance_jaw_,
            'distance_ooc_l': distance_ooc_l_,
            'distance_ooc_r': distance_ooc_r_,
            'distance_frontalis': distance_frontalis_,
            # 'angle_degrees': angle_degrees_rotation,
            's_next': self.position,
            's_jaw_next': self.position_jaw,
            's_ooc_r_next': self.position_ooc_r,
            's_frontalis_next': self.position_frontalis,
            's_ooc_l_next': self.position_ooc_l,                        
            # Add more variables as needed
                             
        })




        


        self.obs = self._get_obs()
        
        self.cmds.select(cl=True)
        self.cmds.currentCtx()
        # Close the reference
        
        # if phase_2_flag == 0:
        #     r_limit_sigmoid_1000_endList = []                

        return np.array(self.obs, dtype=np.float32), reward, done, self.info, r_limit_sigmoid_1000_endList, t_limit_sigmoid_1000_endList


    def mayaSave2file(self):
        
        file_to_save = self.save_folder + "/cp26-a001 after-deform-task b2.mb"
        # import os
        # if os.path.isfile(file_to_save):
        #     # if it exists, delete it
        #     os.remove(file_to_save)
        #     print(f"The file '{file_to_save}' has been deleted.")        
        self.cmds.file(rename = file_to_save)
        self.cmds.file( save=True,f=True, type='mayaBinary' )

    def euclidean_distance_ship(self, point1, point2):
        # Convert points to numpy arrays
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Reshape points if necessary
        vertices_num = len(point1) // 3 # take what remains after it  /3 .
        point1_reshaped = point1.reshape((vertices_num, 3))
        point2_reshaped = point2.reshape((vertices_num, 3))

        # # Calculate the element-wise squared differences
        # squared_diff = (point1 - point2)**2

        # # Calculate distance_ship using vectorized operations
        # distance_ship = np.sqrt(squared_diff)

        # # Calculate the Euclidean distance between entire point sets
        # distance = np.linalg.norm(point1 - point2)

        # Calculate distance_ship_vertices using vectorized operations
        squared_diff_vertices = np.sum((point1_reshaped - point2_reshaped)**2, axis=1)
        distance_ship_vertices = np.sqrt(squared_diff_vertices)

        return distance_ship_vertices.tolist()
  

    def _read_if_moved(self, object_name, joints_now, vertex_indices):
        # Reuse the last readback of this mesh if its joints are exactly where they were then.
        cache = self.__dict__.setdefault("_readback_cache", {})
        joints_now = np.array(joints_now, dtype=float)
        last = cache.get(object_name)
        if last is not None and np.array_equal(last[0], joints_now):
            return last[1].copy()
        coords = self.returnCoordinates(object_name, vertex_indices)
        cache[object_name] = (joints_now, np.array(coords))
        return coords

    def returnCoordinates(self, object_name, vertex_indices=None):
        # Select the object
        self.cmds.select(object_name, visible=True)
        
        # Get the active selection list
        selection = self.OpenMaya.MSelectionList()
        self.OpenMaya.MGlobal.getActiveSelectionList(selection)
        
        # Create an iterator for the selection list
        iterSel = self.OpenMaya.MItSelectionList(selection, self.OpenMaya.MFn.kMesh)
        
        # Get the DAG path and MFnMesh object
        dagPath = self.OpenMaya.MDagPath()
        iterSel.getDagPath(dagPath)
        currentInMeshMFnMesh = self.OpenMaya.MFnMesh(dagPath)
        
        # Get the mesh points in world space
        inMeshMPointArray = self.OpenMaya.MPointArray()
        currentInMeshMFnMesh.getPoints(inMeshMPointArray, self.OpenMaya.MSpace.kWorld)
        
        # Determine if we need to get all vertices or specific ones
        if vertex_indices is None:
            # Pre-allocate NumPy array for all points
            num_points = inMeshMPointArray.length()
            pointArray = np.zeros((num_points, 3))
            
            # Copy all points from MPointArray to NumPy array
            for i in range(num_points):
                point = inMeshMPointArray[i]
                pointArray[i, 0] = point.x
                pointArray[i, 1] = point.y
                pointArray[i, 2] = point.z
        else:
            # Extract the coordinates of the specified vertices
            vertex_coordinates = []
            for index in vertex_indices:
                if index < inMeshMPointArray.length():
                    point = inMeshMPointArray[index]
                    vertex_coordinates.append([point.x, point.y, point.z])
                else:
                    raise ValueError(f"Vertex index {index} is out of range for the mesh.")
            pointArray = np.array(vertex_coordinates)
        
        # Flatten the NumPy array to 1D
        one_d_list = pointArray.flatten()
        
        return one_d_list


    def plot_distances_portal(self, x_bar_conca, s_next_conca, object_name, vertex_index):

        distance = 0.0    
        # distance, distance_ship, distance_ship_vertices = self.euclidean_distance_ship(x_bar_conca, s_next_conca)
        distance_ship_vertices = self.euclidean_distance_ship(x_bar_conca, s_next_conca)
        # method = 'euclidean'
        # distance = self.distance_calculator.calculate_distance(method, self.target_state, s_next_conca)             
        # rms_distance_value = self.distance_calculator.rms_distance(self.target_state, s_next_conca)
        # del distance
        # distance = rms_distance_value

        norm_distances = self.normalize_distances(distance_ship_vertices)

        # Define the output path
        
        output_folder = self.save_folder
        # 'D:/keyLabLocal_D/a2-DSAC-T-for-pipelineV1/results/DSAC_V2_m1_vertexMove_Va002/plot'
        # output_file = os.path.join(output_folder, 'distance_plot.png')
        output_file = os.path.join(output_folder, f'distance_plot_{object_name}.png')


        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        self.plot_distances(x_bar_conca, norm_distances, distance_ship_vertices, output_file, vertex_index)
        
        # print("original distance is: ", distance)
        return   

    def render_distances_portal(self):

        # self.point1_list, self.norm_distances_list, self.distance_ship_list

        # Define the output path
        
        output_folder = self.save_folder + "/animationVideo"
        # 'D:/keyLabLocal_D/a2-DSAC-T-for-pipelineV1/results/DSAC_V2_m1_vertexMove_Va002/plot'
        # time
    
        
        output_file = os.path.join(output_folder, datetime.datetime.now().strftime("%y%m%d-%H%M%S") + "heat_map_animation.mp4")

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        
        # self.save_animation(self.point1_list, self.norm_distances_list, self.distance_ship_list, output_file)
        
        return 

    def normalize_distances(self, distances):
        max_distance = max(distances)
        min_distance = min(distances)
        if max_distance - min_distance == 0:
            norm_distances = [1e-7 for d in distances]
        else:
            norm_distances = [(d - min_distance) / (max_distance - min_distance) for d in distances]
        return norm_distances


    # def plot_distances(self, point1, norm_distances, distance_ship, output_file, vertex_index):
    #     vertices_num = int(len(point1) / 3)
        
    #     # Extracting X, Y, Z coordinates
    #     x_coords = [point1[3*i] for i in range(vertices_num)]
    #     y_coords = [point1[3*i+1] for i in range(vertices_num)]
    #     z_coords = [point1[3*i+2] for i in range(vertices_num)]
        
    #     # Coordinates of the points to highlight
    #     highlight_points = self.control_vertices_slice(point1, vertex_index)
        
    #     # Create a color map based on normalized distances
    #     norm = Normalize(vmin=0, vmax=1)  # Normalize to the range [0, 1]
    #     colors = plt.cm.jet(norm(norm_distances))  # Jet colormap transitions from blue to red

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')

    #     # Highlight the specific points with larger markers
    #     for i in range(len(vertex_index)):
    #         highlight_x = highlight_points[3*i]
    #         highlight_y = highlight_points[3*i+1]
    #         highlight_z = highlight_points[3*i+2]
    #         ax.scatter([highlight_x], [highlight_y], [highlight_z], color='black', marker='*', s=200, label='CTRL', zorder=10)
            
    #         # Add a text annotation to each highlighted point, with an offset to avoid the color bar
    #         text_offset_x = 0.1  # Adjust this value to move the label in the X direction
    #         text_offset_y = 0.1  # Adjust this value to move the label in the Y direction
    #         text_offset_z = 0.1  # Adjust this value to move the label in the Z direction
    #         ax.text(highlight_x + text_offset_x, highlight_y + text_offset_y, highlight_z + text_offset_z,
    #                 'CTRL', color='black', fontsize=12, weight='bold', zorder=10)

    #     # Set the view angle
    #     ax.view_init(elev=(180+30), azim=0, roll=90)  # Change these values to set the desired view

    #     # Adding color bar with normalization
    #     mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
    #     mappable.set_array(norm_distances)
    #     cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', pad=0.1)
    #     cbar.set_label('Distance (mm)')  # Set the label to 'Distance (unit)' TD: to change unit to mm in the code level.
    #     cbar.set_ticks([0, 1])  # Set colorbar ticks to show min and max
    #     cbar.set_ticklabels([f'{min(distance_ship):.2f}', f'{max(distance_ship):.2f}'])  # Set tick labels to min and max distances

    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('Distance Visualization')

    #     # Save the plot to a file
    #     plt.savefig(output_file, dpi=300, bbox_inches='tight')
    #     plt.close()

    def plot_distances(self, point1, norm_distances, distance_ship, output_file, vertex_index):
        vertices_num = int(len(point1) / 3)
        
        # Extracting X, Y, Z coordinates
        x_coords = [point1[3 * i] for i in range(vertices_num)]
        y_coords = [point1[3 * i + 1] for i in range(vertices_num)]
        z_coords = [point1[3 * i + 2] for i in range(vertices_num)]
        
        # Coordinates of the points to highlight
        highlight_points = self.control_vertices_slice(point1, vertex_index)
        
        # Create a color map based on normalized distances
        norm = Normalize(vmin=0, vmax=1)  # Normalize to the range [0, 1]
        colors = plt.cm.jet(norm(norm_distances))  # Jet colormap transitions from blue to red

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')

        # Highlight the specific points with larger markers
        for i in range(len(vertex_index)):
            highlight_x = highlight_points[3 * i]
            highlight_y = highlight_points[3 * i + 1]
            highlight_z = highlight_points[3 * i + 2]
            ax.scatter([highlight_x], [highlight_y], [highlight_z], color='pink', marker='*', s=100, alpha=0.7, zorder=10)

        # Set the view angle
        ax.view_init(elev=(180 + 30), azim=0, roll=90)  # Change these values to set the desired view

        # Adding color bar with normalization
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
        mappable.set_array(norm_distances)
        cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', pad=0.1)
        cbar.set_label('Distance (mm)')  # Set the label to 'Distance (unit)' TD: to change unit to mm in the code level.
        cbar.set_ticks([0, 1])  # Set colorbar ticks to show min and max
        cbar.set_ticklabels([f'{min(distance_ship):.2f}', f'{max(distance_ship):.2f}'])  # Set tick labels to min and max distances

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Distance Visualization')

        # Save the plot to a file
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


    # def plot_distance_vs_time_plot(self):
    #     # Scaling factor for distance
    #     scaling_factor = 0.0962  # 1 Maya unit = 0.0962 cm
    #     scaled_self_distance_log = [t * scaling_factor for t in self.distance_log]

    #     # Set up the plot with a professional style
    #     plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     plt.rcParams['font.size'] = 12  # Increase font size

    #     # Plot the data with a thicker line and professional color
    #     plt.plot(self.step_counter_internal_log, scaled_self_distance_log, 
    #             label='Distance over optimization steps', 
    #             color='blue', linewidth=2, linestyle='-')

    #     # Add labels with increased font size
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Distance (cm)', fontsize=14)

    #     # Remove the upper and right bounds of the box
    #     plt.gca().spines['top'].set_visible(False)
    #     plt.gca().spines['right'].set_visible(False)

    #     # Add gridlines for better readability
    #     plt.grid(True, linestyle='--', alpha=0.6)

    #     # Add a legend with a shadow and place it in the best location
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # Save the plot with high resolution
    #     plt.savefig(os.path.join(self.save_folder, 'distance_vs_time_plot.png'), bbox_inches='tight', dpi=300)
    #     plt.close()

    #     # Optionally, log the final distances and times to a file
    #     # with open(os.path.join(self.save_folder, "distance_log.txt"), 'a') as f:
    #     #     for t, d in zip(self.step_counter_internal_log, scaled_self_distance_log):
    #     #         f.write(f"Steps: {t}, Distance: {d:.4f}\n")

    # # def data_save_distance_vs_time(self):
    # #     # Scaling factor for distance
    # #     scaling_factor = 0.0962  # 1 Maya unit = 0.0962 cm
    # #     scaled_self_distance_log = [t * scaling_factor for t in self.distance_log]


    # #     # Optionally, log the final distances and times to a file
    # #     with open(os.path.join(self.save_folder, "distance_log.txt"), 'a') as f:
    # #         for t, d in zip(self.step_counter_internal_log, scaled_self_distance_log):
    # #             f.write(f"Steps: {t}, Distance: {d:.4f}\n")

    # def plot_distance_vs_time_plot(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     scaling_factor = 0.0962
    #     scaled_distance_log = [d * scaling_factor for d in data['distance_log']]

    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
    #     plt.plot(data['step_counter_internal_log'], scaled_distance_log, 
    #             color='blue', linewidth=2, linestyle='-')
        
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Distance (cm)', fontsize=14)
    #     plt.gca().spines[['top', 'right']].set_visible(False)
    #     plt.grid(True, linestyle='--', alpha=0.6)
        
    #     plt.savefig(os.path.join(self.save_folder, f'distance_vs_time_plot{phase_suffix}.png'), 
    #             bbox_inches='tight', dpi=300)
    #     plt.close()

    def plot_distance_vs_time_plot(self, current_phase=1):
        """Plot distance vs time for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        scaling_factor = 0.0962
        scaled_distance_log = [d * scaling_factor for d in data['distance_log']]

        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
        plt.plot(data['step_counter_internal_log'], scaled_distance_log, 
                color='blue', linewidth=2, linestyle='-')
        
        plt.xlabel('Optimization steps', fontsize=14)
        plt.ylabel('Distance (cm)', fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(os.path.join(self.save_folder, f'distance_vs_time_plot{phase_suffix}.png'), 
                bbox_inches='tight', dpi=300)
        plt.close()

    # def data_save_distance_vs_time(self, phase_2_flag):
    #     # Determine suffix based on phase
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     # Scaling factor for distance
    #     scaling_factor = 0.0962  # 1 Maya unit = 0.0962 cm
        
    #     # Get the appropriate distance log
    #     distance_log = self.distance_log_phase2 if phase_2_flag == 1 else self.distance_log
    #     step_log = self.step_counter_internal_log_phase2 if phase_2_flag == 1 else self.step_counter_internal_log
        
    #     # Scale the distances
    #     scaled_distance_log = [d * scaling_factor for d in distance_log]
        
    #     # Save to file with phase-specific name
    #     filename = f"distance_log{phase_suffix}.txt"
    #     with open(os.path.join(self.save_folder, filename), 'a') as f:
    #         for step, distance in zip(step_log, scaled_distance_log):
    #             f.write(f"Steps: {step}, Distance: {distance:.4f}\n")

    # def data_save_distance_vs_time(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     scaling_factor = 0.0962
    #     scaled_distance_log = [d * scaling_factor for d in data['distance_log']]
        
    #     with open(os.path.join(self.save_folder, f"distance_log{phase_suffix}.txt"), 'a') as f:
    #         for step, distance in zip(data['step_counter_internal_log'], scaled_distance_log):
    #             f.write(f"Steps: {step}, Distance: {distance:.4f}\n")

    def data_save_distance_vs_time(self, current_phase=1):
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        scaling_factor = 0.0962
        scaled_distance_log = [d * scaling_factor for d in data['distance_log']]
        
        with open(os.path.join(self.save_folder, f"distance_log{phase_suffix}.txt"), 'a') as f:
            for step, distance in zip(data['step_counter_internal_log'], scaled_distance_log):
                f.write(f"Steps: {step}, Distance: {distance:.4f}\n")

    # def data_save_energy_vs_time(self, current_phase=1):
    #     data = self._get_phase_specific_data(current_phase)
    #     phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
    #     scaling_factor = 9.26e-7 * 1e3
    #     scaled_strain_energy = [e * scaling_factor for e in data['strain_energy']]
        
    #     with open(os.path.join(self.save_folder, f"energy_log{phase_suffix}.txt"), 'a') as f:
    #         for t, d in zip(data['step_counter_internal_log'], scaled_strain_energy):
    #             f.write(f"Time: {t} seconds, energy: {d:.4f}\n")

    # def data_save_angular_velocity_vs_time(self, current_phase=1):
    #     data = self._get_phase_specific_data(current_phase)
    #     phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
    #     with open(os.path.join(self.save_folder, f"Angular_velocity_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Time (seconds)", "angular velocity (degree/second)", 
    #                         "internal steps"])
    #         writer.writerows(zip(data['time_for_angular_velocity_log'],
    #                         data['rotation_velocity'],
    #                         data['step_counter_internal_log']))

    # def data_save_rotational_kinetic_energy_vs_time(self, current_phase=1):
    #     data = self._get_phase_specific_data(current_phase)
    #     phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
    #     try:
    #         with open(os.path.join(self.save_folder, f"Rotational_kineticenergy_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #             writer = csv.writer(csvfile)
    #             writer.writerow(["Time (seconds)", "rotational kinetic energy (J)", 
    #                             "angular velocity (degree/second)", "internal steps"])
    #             writer.writerows(zip(
    #                 data['time_for_rotational_kinetic_energy_log'],
    #                 data['rotational_kinetic_energy'],
    #                 data['rotation_velocity'],
    #                 data['step_counter_internal_log']
    #             ))
    #     except Exception as e:
    #         print(f"Error saving rotational kinetic energy data: {str(e)}")

    # def data_save_elastic_energy_vs_time_plot(self, current_phase=1):
    #     data = self._get_phase_specific_data(current_phase)
    #     phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
    #     with open(os.path.join(self.save_folder, f"Elastic_energy_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["internal steps", "accumulated angles (degree)", 'Elastic energy (J)'])
    #         writer.writerows(zip(data['step_counter_internal_log'],
    #                         data['accumulated_rotated_angle'],
    #                         data['elastic_energy']))

    # def data_save_accumulated_work_done(self, current_phase=1):
    #     data = self._get_phase_specific_data(current_phase)
    #     phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
    #     T_prime = 35 / 30
        
    #     if 'rotational_kinetic_energy' not in data or 'elastic_energy' not in data:
    #         print("ERROR: Missing energy data in dictionary")
    #         return 0.0
        
    #     work_rate = np.array(data['rotational_kinetic_energy']) + np.array(data['elastic_energy'])
        
    #     if len(work_rate) == 0:
    #         print("ERROR: Empty work rate array")
    #         return 0.0
        
    #     t = np.linspace(0, T_prime, len(work_rate))
        
    #     try:
    #         W_total = cumulative_trapezoid(work_rate, t, initial=0)
    #     except Exception as e:
    #         print(f"ERROR in integration: {str(e)}")
    #         return

    #     with open(os.path.join(self.save_folder, f"plot_accumulated_work_done{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Time (seconds)", "Total Accumulated Work (J)"])
    #         writer.writerows(zip(t, W_total))

    # def plot_energy_vs_time_plot(self):
    #     # Scaling factor for strain energy
    #     scaling_factor = 9.26 * 10**-7 * 10**3
    #     scaled_self_strain_energy = [t * scaling_factor for t in self.strain_energy]

    #     # Set up the plot with a professional style
    #     plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     plt.rcParams['font.size'] = 12  # Increase font size

    #     # Plot the data with a thicker line and professional color
    #     plt.plot(self.step_counter_internal_log, scaled_self_strain_energy, 
    #             label='Strain energy over optimization steps', 
    #             color='blue', linewidth=2, linestyle='-')

    #     # Add labels with increased font size
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Strain energy (J)', fontsize=14)

    #     # Remove the upper and right bounds of the box
    #     plt.gca().spines['top'].set_visible(False)
    #     plt.gca().spines['right'].set_visible(False)

    #     # Add gridlines for better readability
    #     plt.grid(True, linestyle='--', alpha=0.6)

    #     # Add a legend with a shadow and place it in the best location
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # Save the plot with high resolution
    #     plt.savefig(os.path.join(self.save_folder, 'energy_vs_time_plot.png'), bbox_inches='tight', dpi=300)
    #     plt.close()

    #     # Optionally, log the final distances and times to a file
    #     # with open(os.path.join(self.save_folder, "energy_log.txt"), 'a') as f:
    #     #     for t, d in zip(self.step_counter_internal_log, scaled_self_strain_energy):
    #     #         f.write(f"Time: {t} seconds, energy: {d:.4f}\n")

    # def plot_energy_vs_time_plot(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     scaling_factor = 9.26e-7 * 1e3
    #     scaled_strain_energy = [e * scaling_factor for e in data['strain_energy']]

    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
    #     plt.plot(data['step_counter_internal_log'], scaled_strain_energy, 
    #             color='blue', linewidth=2, linestyle='-')
        
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Strain energy (J)', fontsize=14)
    #     plt.gca().spines[['top', 'right']].set_visible(False)
    #     plt.grid(True, linestyle='--', alpha=0.6)
        
    #     plt.savefig(os.path.join(self.save_folder, f'energy_vs_time_plot{phase_suffix}.png'), 
    #             bbox_inches='tight', dpi=300)
    #     plt.close()

    def plot_energy_vs_time_plot(self, current_phase=1):
        """Plot strain energy vs time for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        scaling_factor = 9.26e-7 * 1e3
        scaled_strain_energy = [e * scaling_factor for e in data['strain_energy']]

        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
        plt.plot(data['step_counter_internal_log'], scaled_strain_energy, 
                color='blue', linewidth=2, linestyle='-')
        
        plt.xlabel('Optimization steps', fontsize=14)
        plt.ylabel('Strain energy (J)', fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(os.path.join(self.save_folder, f'energy_vs_time_plot{phase_suffix}.png'), 
                bbox_inches='tight', dpi=300)
        plt.close()

    # def data_save_energy_vs_time(self):
    #     # Scaling factor for strain energy
    #     scaling_factor = 9.26 * 10**-7 * 10**3
    #     scaled_self_strain_energy = [t * scaling_factor for t in self.strain_energy]

    #     # Set up the plot with a professional style
    #     # plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     # plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     # plt.rcParams['font.size'] = 12  # Increase font size

    #     # # Plot the data with a thicker line and professional color
    #     # plt.plot(self.step_counter_internal_log, scaled_self_strain_energy, 
    #     #         label='Strain energy over optimization steps', 
    #     #         color='blue', linewidth=2, linestyle='-')

    #     # # Add labels with increased font size
    #     # plt.xlabel('Optimization steps', fontsize=14)
    #     # plt.ylabel('Strain energy (J)', fontsize=14)

    #     # # Remove the upper and right bounds of the box
    #     # plt.gca().spines['top'].set_visible(False)
    #     # plt.gca().spines['right'].set_visible(False)

    #     # # Add gridlines for better readability
    #     # plt.grid(True, linestyle='--', alpha=0.6)

    #     # # Add a legend with a shadow and place it in the best location
    #     # plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # # Save the plot with high resolution
    #     # plt.savefig(os.path.join(self.save_folder, 'energy_vs_time_plot.png'), bbox_inches='tight', dpi=300)
    #     # plt.close()

    #     # Optionally, log the final distances and times to a file
    #     with open(os.path.join(self.save_folder, "energy_log.txt"), 'a') as f:
    #         for t, d in zip(self.step_counter_internal_log, scaled_self_strain_energy):
    #             f.write(f"Time: {t} seconds, energy: {d:.4f}\n")

    # def data_save_energy_vs_time(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     scaling_factor = 9.26e-7 * 1e3
    #     scaled_strain_energy = [e * scaling_factor for e in data['strain_energy']]
        
    #     with open(os.path.join(self.save_folder, f"energy_log{phase_suffix}.txt"), 'a') as f:
    #         for t, d in zip(data['step_counter_internal_log'], scaled_strain_energy):
    #             f.write(f"Time: {t} seconds, energy: {d:.4f}\n")

    def data_save_energy_vs_time(self, current_phase=1):
        """Save energy vs time data for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        scaling_factor = 9.26e-7 * 1e3
        scaled_strain_energy = [e * scaling_factor for e in data['strain_energy']]
        
        with open(os.path.join(self.save_folder, f"energy_log{phase_suffix}.txt"), 'a') as f:
            for t, d in zip(data['step_counter_internal_log'], scaled_strain_energy):
                f.write(f"Time: {t} seconds, energy: {d:.4f}\n")

    # def plot_rotational_kinetic_energy_vs_time_plot(self):
    #     # Set up the plot with a professional style
    #     plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     plt.rcParams['font.size'] = 12  # Increase font size

    #     # Plot the data with a thicker line and professional color
    #     plt.plot(self.step_counter_internal_log, self.rotational_kinetic_energy, 
    #             label='Rotational kinetic energy\nover optimization steps',  # Split into two lines
    #             color='blue', linewidth=2, linestyle='-')

    #     # Add labels with increased font size
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Rotational kinetic energy (J)', fontsize=14)

    #     # Remove the upper and right bounds of the box
    #     plt.gca().spines['top'].set_visible(False)
    #     plt.gca().spines['right'].set_visible(False)

    #     # Add gridlines for better readability
    #     plt.grid(True, linestyle='--', alpha=0.6)

    #     # Add a legend with a shadow and place it in the best location
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # Save the plot with high resolution
    #     plt.savefig(os.path.join(self.save_folder, 'Rotational_kinetic_energy_vs_steps_plot.png'), bbox_inches='tight', dpi=300)
    #     plt.close()

    #     # Save the data to a CSV file for use in Origin
    #     # with open(os.path.join(self.save_folder, "Rotational_kineticenergy_data.csv"), 'w', newline='') as csvfile:
    #     #     writer = csv.writer(csvfile)
    #     #     writer.writerow(["Time (seconds)", "rotational kinetic energy (J)", "angular velocity (degree/second)", "internal steps"])  # Write header
    #     #     writer.writerows(zip(self.time_for_rotational_kinetic_energy_log, self.rotational_kinetic_energy, self.rotation_velocity, self.step_counter_internal_log))  # Write data rows

    # def plot_rotational_kinetic_energy_vs_time_plot(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
    #     plt.plot(data['step_counter_internal_log'][0:-1], data['rotational_kinetic_energy'][0:-1],
    #             label='Rotational kinetic energy\nover optimization steps',
    #             color='blue', linewidth=2, linestyle='-')
        
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Rotational kinetic energy (J)', fontsize=14)
    #     plt.gca().spines[['top', 'right']].set_visible(False)
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
    #     plt.savefig(os.path.join(self.save_folder, f'Rotational_kinetic_energy_vs_steps_plot{phase_suffix}.png'),
    #             bbox_inches='tight', dpi=300)
    #     plt.close()

    def plot_rotational_kinetic_energy_vs_time_plot(self, current_phase=1):
        """Plot rotational kinetic energy for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
        plt.plot(data['step_counter_internal_log'][0:-1], data['rotational_kinetic_energy'][0:-1],
                label='Rotational kinetic energy\nover optimization steps',
                color='blue', linewidth=2, linestyle='-')
        
        plt.xlabel('Optimization steps', fontsize=14)
        plt.ylabel('Rotational kinetic energy (J)', fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
        plt.savefig(os.path.join(self.save_folder, f'Rotational_kinetic_energy_vs_steps_plot{phase_suffix}.png'),
                bbox_inches='tight', dpi=300)
        plt.close()

    # def plot_angular_velocity_vs_time_plot(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     # Convert to numpy array for vectorized operations
    #     rotation_velocity = np.array(data['rotation_velocity'])
    #     steps = np.array(data['step_counter_internal_log'])
                
    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
    #     plt.plot(steps, -rotation_velocity,
    #             label='Angular velocity\nover optimization steps',
    #             color='green', linewidth=2, linestyle='-')
        
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Angular velocity (degree/second) absolute value', fontsize=14)
    #     plt.gca().spines[['top', 'right']].set_visible(False)
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
    #     plt.savefig(os.path.join(self.save_folder, f'Angular_velocity_vs_steps_plot{phase_suffix}.png'),
    #             bbox_inches='tight', dpi=300)
    #     plt.close()

    def plot_angular_velocity_vs_time_plot(self, current_phase=1):
        """Plot angular velocity for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        # Convert to numpy array for vectorized operations
        rotation_velocity = np.array(data['rotation_velocity'])
        steps = np.array(data['step_counter_internal_log'])
                
        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
        if self.this_task in self.contract_to_neutral_task:  # current_phase == 1 or current_phase == 5 or current_phase == 7:
            rotation_velocity = -rotation_velocity

        plt.plot(steps, -rotation_velocity,
                label='Angular velocity\nover optimization steps',
                color='green', linewidth=2, linestyle='-')
        
        plt.xlabel('Optimization steps', fontsize=14)
        plt.ylabel('Angular velocity (degree/second) absolute value', fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
        plt.savefig(os.path.join(self.save_folder, f'Angular_velocity_vs_steps_plot{phase_suffix}.png'),
                bbox_inches='tight', dpi=300)
        plt.close()


    # def data_save_angular_velocity_vs_time(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     with open(os.path.join(self.save_folder, f"Angular_velocity_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Time (seconds)", "angular velocity (degree/second)", 
    #                         "internal steps"])
    #         writer.writerows(zip(data['time_for_angular_velocity_log'],
    #                         data['rotation_velocity'],
    #                         data['step_counter_internal_log']))

    def data_save_angular_velocity_vs_time(self, current_phase=1):
        """Save angular velocity data for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        with open(os.path.join(self.save_folder, f"Angular_velocity_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Time (seconds)", "angular velocity (degree/second)", 
                            "internal steps"])
            writer.writerows(zip(data['time_for_rotational_kinetic_energy'],
                            data['rotation_velocity'],
                            data['step_counter_internal_log']))

    # def data_save_rotational_kinetic_energy_vs_time(self):
    #     # Set up the plot with a professional style
    #     # plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     # plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     # plt.rcParams['font.size'] = 12  # Increase font size

    #     # # Plot the data with a thicker line and professional color
    #     # plt.plot(self.step_counter_internal_log, self.rotational_kinetic_energy, 
    #     #         label='Rotational kinetic energy\nover optimization steps',  # Split into two lines
    #     #         color='blue', linewidth=2, linestyle='-')

    #     # # Add labels with increased font size
    #     # plt.xlabel('Optimization steps', fontsize=14)
    #     # plt.ylabel('Rotational kinetic energy (J)', fontsize=14)

    #     # # Remove the upper and right bounds of the box
    #     # plt.gca().spines['top'].set_visible(False)
    #     # plt.gca().spines['right'].set_visible(False)

    #     # # Add gridlines for better readability
    #     # plt.grid(True, linestyle='--', alpha=0.6)

    #     # # Add a legend with a shadow and place it in the best location
    #     # plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # # Save the plot with high resolution
    #     # plt.savefig(os.path.join(self.save_folder, 'Rotational_kinetic_energy_vs_steps_plot.png'), bbox_inches='tight', dpi=300)
    #     # plt.close()

    #     # Save the data to a CSV file for use in Origin
    #     with open(os.path.join(self.save_folder, "Rotational_kineticenergy_data.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Time (seconds)", "rotational kinetic energy (J)", "angular velocity (degree/second)", "internal steps"])  # Write header
    #         writer.writerows(zip(self.time_for_rotational_kinetic_energy_log, self.rotational_kinetic_energy, self.rotation_velocity, self.step_counter_internal_log))  # Write data rows

    # def data_save_rotational_kinetic_energy_vs_time(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     with open(os.path.join(self.save_folder, f"Rotational_kineticenergy_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Time (seconds)", "rotational kinetic energy (J)", 
    #                         "angular velocity (degree/second)", "internal steps"])
    #         writer.writerows(zip(data['time_for_rotational_kinetic_energy_log'],
    #                         data['rotational_kinetic_energy'],
    #                         data['rotation_velocity'],
    #                         data['step_counter_internal_log']))


    # def data_save_rotational_kinetic_energy_vs_time(self, phase_2_flag=0):
    #     # print(f"\n=== DEBUG: data_save_rotational_kinetic_energy_vs_time (phase {phase_2_flag}) ===")
    #     data = self._get_phase_specific_data(phase_2_flag)
        
    #     # Debug print data contents
    #     # print(f"Data keys: {list(data.keys())}")
    #     # print(f"time_for_rotational_kinetic_energy_log length: {len(data.get('time_for_rotational_kinetic_energy_log', []))}")
    #     # print(f"rotational_kinetic_energy length: {len(data.get('rotational_kinetic_energy', []))}")
    #     # print(f"rotation_velocity length: {len(data.get('rotation_velocity', []))}")
    #     # print(f"step_counter_internal_log length: {len(data.get('step_counter_internal_log', []))}")
        
    #     # Verify all arrays have same length
    #     lengths = {
    #         'time': len(data.get('time_for_rotational_kinetic_energy_log', [])),
    #         'energy': len(data.get('rotational_kinetic_energy', [])),
    #         'velocity': len(data.get('rotation_velocity', [])),
    #         'steps': len(data.get('step_counter_internal_log', []))
    #     }
    #     # print("Array lengths:", lengths)
        
    #     if not all(l == lengths['time'] for l in lengths.values()):
    #         print("WARNING: Array length mismatch!")
        
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     try:
    #         with open(os.path.join(self.save_folder, f"Rotational_kineticenergy_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #             writer = csv.writer(csvfile)
    #             writer.writerow(["Time (seconds)", "rotational kinetic energy (J)", 
    #                             "angular velocity (degree/second)", "internal steps"])
                
    #             # Debug first 5 rows being written
    #             rows = list(zip(
    #                 data['time_for_rotational_kinetic_energy_log'],
    #                 data['rotational_kinetic_energy'],
    #                 data['rotation_velocity'],
    #                 data['step_counter_internal_log']
    #             ))
    #             # print("First 5 rows:", rows[:5])
                
    #             writer.writerows(rows)
    #         # print(f"Successfully saved {len(rows)} data points")
    #     except Exception as e:
    #         print(f"Error saving rotational kinetic energy data: {str(e)}")

    def data_save_rotational_kinetic_energy_vs_time(self, current_phase=1):
        """Save rotational kinetic energy data for any phase (1, 2, 3, ...) with debugging"""
        data = self._get_phase_specific_data(current_phase)
        
        # Debug print data contents
        # print(f"\n=== DEBUG: data_save_rotational_kinetic_energy_vs_time (phase {current_phase}) ===")
        # print(f"Data keys: {list(data.keys())}")
        # print(f"time_for_rotational_kinetic_energy length: {len(data.get('time_for_rotational_kinetic_energy', []))}")
        # print(f"rotational_kinetic_energy length: {len(data.get('rotational_kinetic_energy', []))}")
        # print(f"rotation_velocity length: {len(data.get('rotation_velocity', []))}")
        # print(f"step_counter_internal_log length: {len(data.get('step_counter_internal_log', []))}")
        
        # Verify all arrays have same length
        lengths = {
            'time': len(data.get('time_for_rotational_kinetic_energy', [])),
            'energy': len(data.get('rotational_kinetic_energy', [])),
            'velocity': len(data.get('rotation_velocity', [])),
            'steps': len(data.get('step_counter_internal_log', []))
        }
        # print("Array lengths:", lengths)
        
        if not all(l == lengths['time'] for l in lengths.values()):
            print(f"WARNING: Array length mismatch in phase {current_phase}!")
        
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        try:
            with open(os.path.join(self.save_folder, f"Rotational_kineticenergy_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time (seconds)", "rotational kinetic energy (J)", 
                                "angular velocity (degree/second)", "internal steps"])
                
                # Debug first 5 rows being written
                rows = list(zip(
                    data['time_for_rotational_kinetic_energy'],
                    data['rotational_kinetic_energy'],
                    data['rotation_velocity'],
                    data['step_counter_internal_log']
                ))
                # print(f"First 5 rows (phase {current_phase}):", rows[:5])
                
                writer.writerows(rows)
            # print(f"Successfully saved {len(rows)} data points for phase {current_phase}")
        except Exception as e:
            print(f"Error saving rotational kinetic energy data for phase {current_phase}: {str(e)}")


    # def plot_elastic_energy_vs_time_plot(self):
    #     # Set up the plot with a professional style
    #     plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     plt.rcParams['font.size'] = 12  # Increase font size

    #     # Plot the data with a thicker line and professional color
    #     plt.plot(self.step_counter_internal_log, self.elastic_energy, 
    #             label='Elastic energy\nover optimization steps', 
    #             color='blue', linewidth=2, linestyle='-')

    #     # Add labels with increased font size
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Elastic energy (J)', fontsize=14)

    #     # Remove the upper and right bounds of the box
    #     plt.gca().spines['top'].set_visible(False)
    #     plt.gca().spines['right'].set_visible(False)

    #     # Add gridlines for better readability
    #     plt.grid(True, linestyle='--', alpha=0.6)

    #     # Add a legend with a shadow and place it in the best location
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # Save the plot with high resolution
    #     plt.savefig(os.path.join(self.save_folder, 'Elastic_energy_vs_steps_plot.png'), bbox_inches='tight', dpi=300)
    #     plt.close()

    #     # Save the data to a CSV file for use in Origin
    #     # with open(os.path.join(self.save_folder, "Elastic_energy_data.csv"), 'w', newline='') as csvfile:
    #     #     writer = csv.writer(csvfile)
    #     #     writer.writerow(["internal steps", "accumulated angles (degree)", 'Elastic energy (J)'])  # Write header
    #     #     writer.writerows(zip(self.step_counter_internal_log, self.accumulated_rotated_angle, self.elastic_energy))  # Write data rows

    # def plot_elastic_energy_vs_time_plot(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
    #     plt.plot(data['step_counter_internal_log'], data['elastic_energy'],
    #             label='Elastic energy\nover optimization steps',
    #             color='blue', linewidth=2, linestyle='-')
        
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Elastic energy (J)', fontsize=14)
    #     plt.gca().spines[['top', 'right']].set_visible(False)
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
    #     plt.savefig(os.path.join(self.save_folder, f'Elastic_energy_vs_steps_plot{phase_suffix}.png'),
    #             bbox_inches='tight', dpi=300)
    #     plt.close()

    def plot_elastic_energy_vs_time_plot(self, current_phase=1):
        """Plot elastic energy for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
        plt.plot(data['step_counter_internal_log'], data['elastic_energy'],
                label='Elastic energy\nover optimization steps',
                color='blue', linewidth=2, linestyle='-')
        
        plt.xlabel('Optimization steps', fontsize=14)
        plt.ylabel('Elastic energy (J)', fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
        plt.savefig(os.path.join(self.save_folder, f'Elastic_energy_vs_steps_plot{phase_suffix}.png'),
                bbox_inches='tight', dpi=300)
        plt.close()

    # def data_save_elastic_energy_vs_time_plot(self):
    #     # Set up the plot with a professional style
    #     # plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     # plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     # plt.rcParams['font.size'] = 12  # Increase font size

    #     # # Plot the data with a thicker line and professional color
    #     # plt.plot(self.step_counter_internal_log, self.elastic_energy, 
    #     #         label='Elastic energy\nover optimization steps', 
    #     #         color='blue', linewidth=2, linestyle='-')

    #     # # Add labels with increased font size
    #     # plt.xlabel('Optimization steps', fontsize=14)
    #     # plt.ylabel('Elastic energy (J)', fontsize=14)

    #     # # Remove the upper and right bounds of the box
    #     # plt.gca().spines['top'].set_visible(False)
    #     # plt.gca().spines['right'].set_visible(False)

    #     # # Add gridlines for better readability
    #     # plt.grid(True, linestyle='--', alpha=0.6)

    #     # # Add a legend with a shadow and place it in the best location
    #     # plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # # Save the plot with high resolution
    #     # plt.savefig(os.path.join(self.save_folder, 'Elastic_energy_vs_steps_plot.png'), bbox_inches='tight', dpi=300)
    #     # plt.close()

    #     # Save the data to a CSV file for use in Origin
    #     with open(os.path.join(self.save_folder, "Elastic_energy_data.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["internal steps", "accumulated angles (degree)", 'Elastic energy (J)'])  # Write header
    #         writer.writerows(zip(self.step_counter_internal_log, self.accumulated_rotated_angle, self.elastic_energy))  # Write data rows

    # def data_save_elastic_energy_vs_time_plot(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     with open(os.path.join(self.save_folder, f"Elastic_energy_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["internal steps", "accumulated angles (degree)", 'Elastic energy (J)'])
    #         writer.writerows(zip(data['step_counter_internal_log'],
    #                         data['accumulated_rotated_angle'],
    #                         data['elastic_energy']))

    def data_save_elastic_energy_vs_time_plot(self, current_phase=1):
        """Save elastic energy data for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        with open(os.path.join(self.save_folder, f"Elastic_energy_data{phase_suffix}.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["internal steps", "accumulated angles (degree)", 'Elastic energy (J)'])
            writer.writerows(zip(data['step_counter_internal_log'],
                            data['accumulated_rotated_angle'],
                            data['elastic_energy']))

    # def plot_accumulated_work_done(self):
    #     # Time array
    #     T_prime = 35 / 30
    #     # Ensure work_rate is computed as element-wise addition
    #     work_rate = np.array(self.rotational_kinetic_energy) + np.array(self.elastic_energy)        
    #     # work_rate = self.rotational_kinetic_energy + self.elastic_energy
    #     t = np.linspace(0, T_prime, len(work_rate))  # Total time for jaw movement

    #     # Calculate total accumulated work
    #     W_total = cumulative_trapezoid(work_rate, t, initial=0)

    #     # Set up the plot with a professional style
    #     plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     plt.rcParams['font.size'] = 12  # Increase font size

    #     # Plot the data with a thicker line and professional color
    #     plt.plot(t, W_total, 
    #             label="Total Accumulated Work (J)", 
    #             color='green', linewidth=2, linestyle='-')

    #     # Add labels with increased font size
    #     plt.xlabel("Animation time (s)", fontsize=14)
    #     plt.ylabel("Work Done (J)", fontsize=14)

    #     # Remove the upper and right bounds of the box
    #     plt.gca().spines['top'].set_visible(False)
    #     plt.gca().spines['right'].set_visible(False)

    #     # Add gridlines for better readability
    #     plt.grid(True, linestyle='--', alpha=0.6)

    #     # Add a legend with a shadow and place it in the best location
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # Save the plot with high resolution
    #     plt.savefig(os.path.join(self.save_folder, 'plot_accumulated_work_done.png'), bbox_inches='tight', dpi=300)
    #     plt.close()

    #     # Save the data to a CSV file for use in Origin
    #     # with open(os.path.join(self.save_folder, "plot_accumulated_work_done.csv"), 'w', newline='') as csvfile:
    #     #     writer = csv.writer(csvfile)
    #     #     writer.writerow(["Time (seconds)", "Total Accumulated Work (J)"])  # Write header
    #     #     writer.writerows(zip(t, W_total))  # Write data rows
    # def plot_accumulated_work_done(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     T_prime = 35 / 30
    #     work_rate = np.array(data['rotational_kinetic_energy']) + np.array(data['elastic_energy'])
    #     t = np.linspace(0, T_prime, len(work_rate))
    #     W_total = cumulative_trapezoid(work_rate, t, initial=0)

    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
    #     plt.plot(t, W_total,
    #             label="Total Accumulated Work (J)",
    #             color='green', linewidth=2, linestyle='-')
        
    #     plt.xlabel("Animation time (s)", fontsize=14)
    #     plt.ylabel("Work Done (J)", fontsize=14)
    #     plt.gca().spines[['top', 'right']].set_visible(False)
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
    #     plt.savefig(os.path.join(self.save_folder, f'plot_accumulated_work_done{phase_suffix}.png'),
    #             bbox_inches='tight', dpi=300)
    #     plt.close()      

    def plot_accumulated_work_done(self, current_phase=1):
        """Plot accumulated work for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        T_prime = 35 / 30
        work_rate = np.array(data['rotational_kinetic_energy']) + np.array(data['elastic_energy'])
        t = np.linspace(0, T_prime, len(work_rate))
        W_total = cumulative_trapezoid(work_rate, t, initial=0)

        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
        plt.plot(t, W_total,
                label="Total Accumulated Work (J)",
                color='green', linewidth=2, linestyle='-')
        
        plt.xlabel("Animation time (s)", fontsize=14)
        plt.ylabel("Work Done (J)", fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
        plt.savefig(os.path.join(self.save_folder, f'plot_accumulated_work_done{phase_suffix}.png'),
                bbox_inches='tight', dpi=300)
        plt.close()    

    # def data_save_accumulated_work_done(self):
    #     # Time array
    #     T_prime = 35 / 30
    #     # Ensure work_rate is computed as element-wise addition
    #     work_rate = np.array(self.rotational_kinetic_energy) + np.array(self.elastic_energy)        
    #     # work_rate = self.rotational_kinetic_energy + self.elastic_energy
    #     t = np.linspace(0, T_prime, len(work_rate))  # Total time for jaw movement

    #     # Calculate total accumulated work
    #     W_total = cumulative_trapezoid(work_rate, t, initial=0)

    #     # Set up the plot with a professional style
    #     # plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    #     # plt.rcParams['font.family'] = 'serif'  # Use serif font
    #     # plt.rcParams['font.size'] = 12  # Increase font size

    #     # # Plot the data with a thicker line and professional color
    #     # plt.plot(t, W_total, 
    #     #         label="Total Accumulated Work (J)", 
    #     #         color='green', linewidth=2, linestyle='-')

    #     # # Add labels with increased font size
    #     # plt.xlabel("Animation time (s)", fontsize=14)
    #     # plt.ylabel("Work Done (J)", fontsize=14)

    #     # # Remove the upper and right bounds of the box
    #     # plt.gca().spines['top'].set_visible(False)
    #     # plt.gca().spines['right'].set_visible(False)

    #     # # Add gridlines for better readability
    #     # plt.grid(True, linestyle='--', alpha=0.6)

    #     # # Add a legend with a shadow and place it in the best location
    #     # plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    #     # # Save the plot with high resolution
    #     # plt.savefig(os.path.join(self.save_folder, 'plot_accumulated_work_done.png'), bbox_inches='tight', dpi=300)
    #     # plt.close()

    #     # Save the data to a CSV file for use in Origin
    #     with open(os.path.join(self.save_folder, "plot_accumulated_work_done.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Time (seconds)", "Total Accumulated Work (J)"])  # Write header
    #         writer.writerows(zip(t, W_total))  # Write data rows
      
    # def data_save_accumulated_work_done(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     T_prime = 35 / 30
    #     work_rate = np.array(data['rotational_kinetic_energy']) + np.array(data['elastic_energy'])
    #     t = np.linspace(0, T_prime, len(work_rate))
    #     W_total = cumulative_trapezoid(work_rate, t, initial=0)

    # def data_save_accumulated_work_done(self, phase_2_flag=0):
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     T_prime = 35 / 30
        
    #     # Debug prints
    #     # print(f"\n=== DEBUG: data_save_accumulated_work_done ===")
    #     # print(f"Phase: {'2' if phase_2_flag else '1'}")
    #     # print(f"Data keys: {list(data.keys())}")
        
    #     # Check if required data exists
    #     if 'rotational_kinetic_energy' not in data or 'elastic_energy' not in data:
    #         print("ERROR: Missing energy data in dictionary")
    #         return 0.0
        
    #     # Convert to numpy arrays
    #     work_rate = np.array(data['rotational_kinetic_energy']) + np.array(data['elastic_energy'])
    #     # print(f"Work rate array shape: {work_rate.shape}")
    #     # print(f"First 5 work rate values: {work_rate[:5] if len(work_rate) > 0 else 'EMPTY'}")
        
    #     # Validate array length
    #     if len(work_rate) == 0:
    #         print("ERROR: Empty work rate array")
    #         return 0.0
        
    #     # Create time array
    #     t = np.linspace(0, T_prime, len(work_rate))
    #     # print(f"Time array shape: {t.shape}")
        
    #     try:
    #         W_total = cumulative_trapezoid(work_rate, t, initial=0)
    #         # print(f"Calculated total work: {W_total[-1] if len(W_total) > 0 else 0}")
    #         # return W_total
    #     except Exception as e:
    #         print(f"ERROR in integration: {str(e)}")
    #         print(f"Work rate length: {len(work_rate)}")
    #         print(f"Time array length: {len(t)}")
    #         # return np.zeros_like(work_rate)

    #     with open(os.path.join(self.save_folder, f"plot_accumulated_work_done{phase_suffix}.csv"), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Time (seconds)", "Total Accumulated Work (J)"])
    #         writer.writerows(zip(t, W_total))      

    def data_save_accumulated_work_done(self, current_phase=1):
        """Save accumulated work data for any phase (1, 2, 3, ...) with debugging"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        T_prime = 35 / 30
        
        # Debug prints - now phase-aware
        # print(f"\n=== DEBUG: data_save_accumulated_work_done (phase {current_phase}) ===")
        # print(f"Data keys: {list(data.keys())}")
        
        # Check if required data exists
        if 'rotational_kinetic_energy' not in data or 'elastic_energy' not in data:
            print(f"ERROR: Missing energy data in phase {current_phase} dictionary")
            return 0.0
        
        # Convert to numpy arrays
        work_rate = np.array(data['rotational_kinetic_energy']) + np.array(data['elastic_energy'])
        # print(f"Phase {current_phase} work rate array shape: {work_rate.shape}")
        # print(f"First 5 work rate values: {work_rate[:5] if len(work_rate) > 0 else 'EMPTY'}")
        
        # Validate array length
        if len(work_rate) == 0:
            print(f"ERROR: Empty work rate array in phase {current_phase}")
            return 0.0
        
        # Create time array
        t = np.linspace(0, T_prime, len(work_rate))
        # print(f"Phase {current_phase} time array shape: {t.shape}")
        
        try:
            W_total = cumulative_trapezoid(work_rate, t, initial=0)
            # print(f"Phase {current_phase} calculated total work: {W_total[-1] if len(W_total) > 0 else 0}")
        except Exception as e:
            print(f"ERROR in integration (phase {current_phase}): {str(e)}")
            print(f"Work rate length: {len(work_rate)}")
            print(f"Time array length: {len(t)}")
            return np.zeros_like(work_rate)

        with open(os.path.join(self.save_folder, f"plot_accumulated_work_done{phase_suffix}.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Time (seconds)", "Total Accumulated Work (J)"])
            writer.writerows(zip(t, W_total))


    def update_plot(self, frame, point1_list, norm_distances_list, distance_ship_list, scatter, ax, x_limits, y_limits, z_limits):
        # Update point1, norm_distances, distance_ship based on the frame
        point1 = point1_list[frame]
        norm_distances = norm_distances_list[frame]
        distance_ship = distance_ship_list[frame]
        
        # Extracting X, Y, Z coordinates
        vertices_num = int(len(point1) / 3)
        x_coords = [point1[3*i] for i in range(vertices_num)]
        y_coords = [point1[3*i+1] for i in range(vertices_num)]
        z_coords = [point1[3*i+2] for i in range(vertices_num)]
        
        # Coordinates of the point to highlight
        # highlight_point = self.control_vertices_slice(point1, self.vertex_index)
        # highlight_x = highlight_point[0]
        # highlight_y = highlight_point[1]
        # highlight_z = highlight_point[2]
        
        # Update scatter data
        scatter._offsets3d = (x_coords, y_coords, z_coords)
        
        # Normalize the colors based on norm_distances
        norm = Normalize(vmin=0, vmax=1)
        colors = plt.cm.jet(norm(norm_distances))
        
        scatter.set_facecolor(colors)  # Correct method to set colors in 3D scatter

        # Remove the previous highlight if necessary
        # for child in scatter.axes.get_children():
        #     if isinstance(child, plt.Annotation):
        #         child.remove()
        #     elif isinstance(child, PathCollection) and child.get_label() == 'CTRL':
        #         child.remove()

        # # Update the highlighted point
        # scatter_highlight = scatter.axes.scatter([highlight_x], [highlight_y], [highlight_z], color='black', marker='*', s=200, label='CTRL', zorder=10)

        # # Add a text annotation to the highlighted point, with an offset to avoid the color bar
        # text_offset_x = 0.1  # Adjust this value to move the label in the X direction
        # text_offset_y = 0.1  # Adjust this value to move the label in the Y direction
        # text_offset_z = 0.1  # Adjust this value to move the label in the Z direction
        # scatter.axes.text(highlight_x + text_offset_x, highlight_y + text_offset_y, highlight_z + text_offset_z,
        #                 'CTRL', color='black', fontsize=12, weight='bold', zorder=10)

        # # Adding color bar with normalization
        # mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
        # mappable.set_array(norm_distances)

        # # Remove the previous color bar if it exists
        # if self.previous_cbar:
        #     self.previous_cbar.remove()
        
        # # Add a new color bar
        # self.previous_cbar = plt.colorbar(mappable, ax=scatter.axes, orientation='vertical', pad=0.1)
        # self.previous_cbar.set_label('Distance (mm)')
        # self.previous_cbar.set_ticks([0, 1])  # Set colorbar ticks to show min and max
        # self.previous_cbar.set_ticklabels([f'{min(distance_ship):.2f}', f'{max(distance_ship):.2f}'])  # Set tick labels to min and max distances

        scatter.axes.set_xlim3d(x_limits)
        scatter.axes.set_ylim3d(y_limits)
        scatter.axes.set_zlim3d(z_limits)

        scatter.axes.set_xlabel('X')
        scatter.axes.set_ylabel('Y')
        scatter.axes.set_zlabel('Z')
        scatter.axes.set_title('Distance Visualization')

        return scatter,

    def save_animation(self, point1_list, norm_distances_list, distance_ship_list, output_file):
        
        point1 = point1_list[0]
        norm_distances = norm_distances_list[0]
        distance_ship = distance_ship_list[0]
        
        # Extracting X, Y, Z coordinates
        vertices_num = int(len(point1) / 3)
        x_coords = [point1[3*i] for i in range(vertices_num)]
        y_coords = [point1[3*i+1] for i in range(vertices_num)]
        z_coords = [point1[3*i+2] for i in range(vertices_num)]
        
        # Calculate the global limits for the axes
        x_min = min([min([point1[3*i] for i in range(vertices_num)]) for point1 in point1_list])
        x_max = max([max([point1[3*i] for i in range(vertices_num)]) for point1 in point1_list])
        y_min = min([min([point1[3*i+1] for i in range(vertices_num)]) for point1 in point1_list])
        y_max = max([max([point1[3*i+1] for i in range(vertices_num)]) for point1 in point1_list])
        z_min = min([min([point1[3*i+2] for i in range(vertices_num)]) for point1 in point1_list])
        z_max = max([max([point1[3*i+2] for i in range(vertices_num)]) for point1 in point1_list])
        
        x_limits = (x_min, x_max)
        y_limits = (y_min, y_max)
        z_limits = (z_min, z_max)
        
        # Coordinates of the point to highlight
        highlight_points = self.control_vertices_slice(point1, self.vertex_index)
        # highlight_x = highlight_point[0]
        # highlight_y = highlight_point[1]
        # highlight_z = highlight_point[2]        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        norm = Normalize(vmin=0, vmax=1)
        colors = plt.cm.jet(norm(norm_distances))
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')
        
        # Initial plot

        # Highlight the specific points with larger markers
        for i in range(len(self.vertex_index)):
            highlight_x = highlight_points[3*i]
            highlight_y = highlight_points[3*i+1]
            highlight_z = highlight_points[3*i+2]
            ax.scatter([highlight_x], [highlight_y], [highlight_z], color='black', marker='*', s=200, label='CTRL', zorder=10)
            
            # Add a text annotation to each highlighted point, with an offset to avoid the color bar
            text_offset_x = 0.1  # Adjust this value to move the label in the X direction
            text_offset_y = 0.1  # Adjust this value to move the label in the Y direction
            text_offset_z = 0.1  # Adjust this value to move the label in the Z direction
            ax.text(highlight_x + text_offset_x, highlight_y + text_offset_y, highlight_z + text_offset_z,
                    'CTRL', color='black', fontsize=12, weight='bold', zorder=10)
        
        # Highlight the specific point with a larger marker
        # ax.scatter([highlight_x], [highlight_y], [highlight_z], color='black', marker='*', s=200, label='CTRL', zorder=10)
        
        # Add a text annotation to the highlighted point, with an offset to avoid the color bar
        # text_offset_x = 0.1  # Adjust this value to move the label in the X direction
        # text_offset_y = 0.1  # Adjust this value to move the label in the Y direction
        # text_offset_z = 0.1  # Adjust this value to move the label in the Z direction
        # ax.text(highlight_x + text_offset_x, highlight_y + text_offset_y, highlight_z + text_offset_z,
        #         'CTRL', color='black', fontsize=12, weight='bold', zorder=10)

        # Set the view angle
        ax.view_init(elev=(180+30), azim=0, roll=90)  # Change these values to set the desired view

        # Adding color bar with normalization
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
        mappable.set_array(norm_distances)

        cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', pad=0.1)
        # cbar.set_label('Distance (mm)')  # Set the label to 'Distance (unit)' TD: to change unit to mm in the code level.
        cbar.set_label('Distance')  # Set the label to 'Distance (unit)' TD: to change unit to mm in the code level.
        cbar.set_ticks([0, 1])  # Set colorbar ticks to show min and max
        # cbar.set_ticklabels([f'{min(distance_ship):.2f}', f'{max(distance_ship):.2f}'])  # Set tick labels to min and max distances
        cbar.set_ticklabels(["Min", "Max"])  # Set tick labels to "Min Distance" and "Max Distance"
        
        ax.set_xlim3d(x_limits)
        ax.set_ylim3d(y_limits)
        ax.set_zlim3d(z_limits)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Distance Visualization')
        
        print("size of point1_list", len(point1_list))
        print("size of norm_distances_list", len(norm_distances_list))
        print("size of distance_ship_list", len(distance_ship_list))

        ani = FuncAnimation(fig, self.update_plot, frames=len(point1_list), 
                            fargs=(point1_list, norm_distances_list, distance_ship_list, scatter, ax, x_limits, y_limits, z_limits),
                            interval=200, blit=False)
        
        ani.save(output_file, writer='ffmpeg')
        plt.close(fig)

    
    def compute_direction_reward(self, s_prime, s_next, target_state):
        s_prime = np.array(s_prime)
        s_next = np.array(s_next)
        target_state = np.array(target_state)

        # Desired direction vector: from s_prime to target_state
        desired_direction = target_state - s_prime

        # Check if s_prime and s_next are the same
        if np.array_equal(s_prime, s_next):
            reward = 0
            cos_theta = 1  # cos(0 degrees) is 1
        else:
            # Real direction vector: from s_prime to s_next
            real_direction = s_next - s_prime

            # Normalize the vectors
            desired_direction = desired_direction / np.linalg.norm(desired_direction)
            real_direction = real_direction / np.linalg.norm(real_direction)

            # Compute the cosine of the angle between the two vectors
            cos_theta = np.dot(desired_direction, real_direction)

            # Compute the reward
            reward = 0.5 * cos_theta + 0.5
        
        return reward, cos_theta
        
    def control_vertices_slice(self, target_state, vertex_index):
        combined_slices = []
        
        for index in vertex_index:
            slice_part = target_state[index*3:index*3+3]
            combined_slices.extend(slice_part)
        
        return combined_slices    
    


    def calculate_normal(self, vertex1, vertex2, vertex3):
        # Create vectors from the vertices
        v1 = np.array(vertex2) - np.array(vertex1)
        v2 = np.array(vertex3) - np.array(vertex1)
        
        # Calculate the normal using the cross product
        normal = np.cross(v1, v2)
        # normal = - normal
        # print("normal", normal)
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        return normal

    def compute_direction_reward_normal(self, s_next, target_state):
        # Extract vertices from the state lists
        A1, B1, C1 = s_next[0:3], s_next[3:6], s_next[6:9]
        A2, B2, C2 = target_state[0:3], target_state[3:6], target_state[6:9]
        
        # Calculate the normal vectors
        real_direction = self.calculate_normal(A1, B1, C1)
        desired_direction = self.calculate_normal(A2, B2, C2)
        # print("real_direction:", real_direction)
        # print("desired_direction:", desired_direction)
        # Normalize the vectors (already normalized in calculate_normal)
        real_direction = real_direction / np.linalg.norm(real_direction)
        desired_direction = desired_direction / np.linalg.norm(desired_direction)
        
        # Compute the cosine of the angle between the two vectors
        cos_theta = np.dot(desired_direction, real_direction)
                
        return cos_theta    
    
    def cos_theta_to_degrees(self, cos_theta):
        # Compute the arccosine of cos_theta to get the angle in radians
        angle_radians = np.arccos(cos_theta)
        
        # Convert the angle from radians to degrees
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees




    def plot_normal_vector(self, ax, vertices, color, label, offset, length=25):
        A, B, C = vertices[0:3], vertices[3:6], vertices[6:9]
        normal = self.calculate_normal(A, B, C)
        
        # Apply offset for better visualization
        A, B, C = np.array(A) + offset, np.array(B) + offset, np.array(C) + offset

        # Plot the vertices of the plane
        ax.scatter(*A, color=color, s=100, alpha=0.6)
        ax.scatter(*B, color=color, s=100, alpha=0.6)
        ax.scatter(*C, color=color, s=100, alpha=0.6)
        
        # Plot the normal vector
        centroid = np.mean([A, B, C], axis=0)
        normal_scaled = normal * length
        ax.quiver(centroid[0], centroid[1], centroid[2], normal_scaled[0], normal_scaled[1], normal_scaled[2], 
                  color=color, label=label, arrow_length_ratio=0.1)
        
        # Calculate endpoint of the normal vector
        endpoint = centroid + normal_scaled
        
        # Plot the endpoint of the normal vector
        ax.scatter(endpoint[0], endpoint[1], endpoint[2], color=color, marker='*', s=100)

    def compare_angles(self, s_next, target_state, x_bar_j, output_file):
        # Extract vertices from the state lists
        A1, B1, C1 = s_next[0:3], s_next[3:6], s_next[6:9]
        A2, B2, C2 = target_state[0:3], target_state[3:6], target_state[6:9]
        
        # Calculate the normal vectors
        real_direction = self.calculate_normal(A1, B1, C1)
        desired_direction = self.calculate_normal(A2, B2, C2)
        
        # Normalize the vectors (already normalized in calculate_normal)
        real_direction = real_direction / np.linalg.norm(real_direction)
        desired_direction = desired_direction / np.linalg.norm(desired_direction)
        
        # Compute the cosine of the angle between the two vectors
        cos_theta_s_next = np.dot(desired_direction, real_direction)
        angle_degrees_s_next = self.cos_theta_to_degrees(cos_theta_s_next)
        
        # Calculate the normal vector for x_bar_j
        A3, B3, C3 = x_bar_j[0:3], x_bar_j[3:6], x_bar_j[6:9]
        x_bar_j_direction = self.calculate_normal(A3, B3, C3)
        x_bar_j_direction = x_bar_j_direction / np.linalg.norm(x_bar_j_direction)
        
        # Compute the cosine of the angle between the two vectors
        cos_theta_x_bar_j = np.dot(desired_direction, x_bar_j_direction)
        angle_degrees_x_bar_j = self.cos_theta_to_degrees(cos_theta_x_bar_j)
        
        # Create a single figure and axis for plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define offsets for better visualization
        offset_x_bar_j = np.array([0, 0, 0])
        # offset_s_next = np.array([1, 0, 0])  # Slight offset
        # offset_target_state = np.array([-1, 0, 0])  # Slight offset
        offset_s_next = np.array([0, 0, 0])  # Slight offset
        offset_target_state = np.array([0, 0, 0])  # Slight offset

        # Plot the normal vectors
        self.plot_normal_vector(ax, x_bar_j, 'r', 'x_bar_j', offset_x_bar_j, length=25)
        self.plot_normal_vector(ax, s_next, 'b', 's_next', offset_s_next, length=25)
        self.plot_normal_vector(ax, target_state, 'g', 'target_state', offset_target_state, length=25)
        
        # Set labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Set the view angle
        ax.view_init(elev=(180+30), azim=0, roll=90)  # Change these values to set the desired view        
        
        # Save the plot to a file
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()        
        
        # Return the angles
        return angle_degrees_s_next, angle_degrees_x_bar_j



    # def return_coordinates_adhoc(self, rot_forward, phase_2_flag):
    #     # Redirect stdout and stderr to null
    #     sys.stdout = open(os.devnull, 'w')
    #     sys.stderr = open(os.devnull, 'w')

    #     try:
    #         # print("debug l1767")
    #         # Check if the file is not open
    #         if not self.opened_file_status:
    #             # self.opened_file = self.cmds.file(self.maya_file_to_open, r=True, ns=self.namespace)


    #             # Check for unsaved changes and handle them
    #             if self.cmds.file(query=True, modified=True):
    #                 # Save or discard changes as needed
    #                 print("Unsaved changes detected.")
    #                 # save_changes = input("You have unsaved changes. Do you want to save them before opening the new file? (yes/no): ").strip().lower()
    #                 if 1==0:  # save_changes in ['yes', 'y']:
    #                     self.cmds.file(save=True)
    #                     print("Changes saved.")
    #                 else:
    #                     self.cmds.file(newFile=True, force=True)  # Discard changes

    #             # Open the Maya file
    #             try:
    #                 self.opened_file = self.cmds.file(self.maya_file_to_open, open=True, force=True)
    #                 print("Opened file:", self.opened_file)

    #                 self.cmds.playbackOptions(min=1, max=50, animationStartTime=1, animationEndTime=50)

    #                 # Function to delete existing cache nodes
    #                 def delete_cache_nodes():
    #                     cache_nodes = self.cmds.ls(type='cacheFile')
    #                     for node in cache_nodes:
    #                         self.cmds.delete(node)

    #                 # Delete existing cache nodes
    #                 delete_cache_nodes()

    #                 self.cmds.cacheFile(createCacheNode=True, singleCache=True, startTime=1, endTime=50, fileName='myCache', dir='cacheDir')

    #                 self.cmds.currentTime(15)

    #             except Exception as e:
    #                 print(f"Failed to open Maya file: {e}")


    #             self.opened_file_status = True  # Update the state to indicate the file is open
    #         else:
    #             pass
    #             # print("File is already open. Skipping the open operation.")


    #         # Perform the action
    #         # ...
        
    #     finally:
    #         # print("debug l1782")
    #         # Restore stdout and stderr
    #         sys.stdout = sys.__stdout__
    #         sys.stderr = sys.__stderr__        


    #     def print_all_objects():
    #         all_objects = self.cmds.ls()
    #         print("All objects in the scene:")
    #         for obj in all_objects:
    #             print(obj)

    #     # print_all_objects()
    #     i = 0



    #     relative_rotation_value = rot_forward

    #     # print("relative_rotation_value, ", relative_rotation_value)

    #     # rotate. 
        
    #     for i in range(0,self.joint_rotate_num):  #3-3,18-3
    #         flag_temp1 = relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0
    #         # flag_temp2 = np.array_equal(self.x_joint_rotation[3*i+1],x_joint_rotation_[3*i+1])
    #         # flag_temp3 = np.array_equal(self.x_joint_rotation[3*i+2],x_joint_rotation_[3*i+2])
    #         # print("rotate flag_temp1, ", flag_temp1)
    #         # print("flag_temp2, ", flag_temp2)
    #         # print("flag_temp3, ", flag_temp3)
    #         if not flag_temp1:  
    #             # self.cmds.rotate( relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws = True)
    #             # print("really rotate l423")


    #             try:
                    
    #                 self.cmds.rotate( relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws = True)
    #                 # print("really rotate l428")
    #                 #bk2
    #                 # temporary manually lock rotate y and z
    #             except Exception as e:
    #                 print(f"l332 at env, unimportant error, An error occurred: {e}")
    #                 if relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0:
    #                     try:
    #                         print("\n\n line 594 joint_jaw_rotate_num:", self.joint_jaw_rotate_num, " \n\n'jaw0'+str(i+1)", '\n\njaw0', str(i+1))
    #                         self.cmds.rotate( 0, 0, 0, 'jaw0'+str(i+1), relative=True, ws = True)

    #                     except Exception as e3:
    #                         print(f"An error occurred: {e3}")
    #                         pass 


    #     #bk1  
    #     new_coords_jaw_forward = self.returnCoordinates("middleres_Jaw", self.vertex_jaw_index)



    #     def calculate_accumulated_work():

    #         def kinetic():
    #         # calculate the kinetic energy.
    #             elements = relative_rotation_value
    #             rotational_velocity_average = sum(elements) # / len(elements)
    #             if self.fix_flag_ship[1] == 1: # rotation has stopped, kinetic energy to 0.
    #                 rotation_velocity = 0
    #                 rotational_kinetic_energy = 0     
    #             else:                   
                    
    #                 rotation_velocity = rotational_velocity_average # / (  (35/30) /self.expected_total_steps_for_jaw) # unit: degree per second
    #                 rotation_velocity_prime = rotation_velocity * math.pi / 180  # unit: rad/s
    #                 I = 1/3 * 0.1089 * 0.17**2  # jaw weighs (wet, and estimated mass with marrow) 0.1089kg, Radius of Rotation: 17 cm (from the center of rotation to the middle teeth).
    #                 # jaw weight referece: {Zhang, Futang, title={Mass properties of the human mandible}
    #                 E_rotation = 1/2 * I * rotation_velocity_prime**2  # equal to "The work (physical terminology) done by the muscles"
    #                 rotational_kinetic_energy = E_rotation
    #                 convert_factor = 1.0  # 10e-3  if convert J to mJ 
    #                 rotational_kinetic_energy = rotational_kinetic_energy * convert_factor

    #             self.rotational_kinetic_energy.append(rotational_kinetic_energy)

    #         kinetic()


    #         def elastic():
    #             # calculate the accumulated rotated angle
    #             def calculate_accumulated_rotated_angle():
                    
    #                 # action = np.array([0.0]*3*self.joint_num + [0.0]*3*self.joint_rotate_num + [0.0]*3*self.joint_ooc_l_num + [0.0]*3*self.joint_ooc_r_num + [0.0]*3*self.joint_frontalis_num)       

    #                 # action[self.joint_num * 3:self.joint_num * 3+self.joint_rotate_num * 3] = relative_rotation_value
    #                 # base_action_prime = np.array([0.0]*3*self.joint_num + [0.0]*3*self.joint_rotate_num + [0.0]*3*self.joint_ooc_l_num + [0.0]*3*self.joint_ooc_r_num + [0.0]*3*self.joint_frontalis_num)       


    #                 # format of relative_rotation_value
    #                 # rotation_values = 3 * self.joint_rotate_num * [0.0]
    #                 # relative_rotation_value = rotation_values

    #                 # Initialize action array
    #                 action = np.array([0.0] * 3 * self.joint_num + 
    #                                 [0.0] * 3 * self.joint_rotate_num + 
    #                                 [0.0] * 3 * self.joint_ooc_l_num + 
    #                                 [0.0] * 3 * self.joint_ooc_r_num + 
    #                                 [0.0] * 3 * self.joint_frontalis_num)

    #                 # Debugging: Print lengths and values before assignment
    #                 # print("Length of relative_rotation_value:", len(relative_rotation_value))
    #                 # print("relative_rotation_value:", relative_rotation_value)
    #                 # print("Length of action:", len(action))
    #                 # print("Slice indices:", self.joint_num * 3, "to", self.joint_num * 3 + self.joint_rotate_num * 3)
    #                 # print("Length of slice:", len(action[self.joint_num * 3:self.joint_num * 3 + self.joint_rotate_num * 3]))

    #                 # Perform the assignment
    #                 action[self.joint_num * 3:self.joint_num * 3 + self.joint_rotate_num * 3] = relative_rotation_value

    #                 # Debugging: Print action after assignment
    #                 # print("Action after assignment:", action)
    #                 base_action_prime = np.array([0.0]*3*self.joint_num + [0.0]*3*self.joint_rotate_num + [0.0]*3*self.joint_ooc_l_num + [0.0]*3*self.joint_ooc_r_num + [0.0]*3*self.joint_frontalis_num)       



    #                 # for rotation
    #                 for actionI in range(3*self.joint_num, 3*self.joint_num + self.joint_rotate_num * 3):    # 0,3*self.joint_num         
    #                     base_action_prime[int(actionI)] = action[int(actionI)]                    
    #                 if self.fix_flag_ship[1] == 1:
    #                     base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num] = 0
                                        
    #                 self.action_real_list.append(base_action_prime)
    #                 final_epoch_action_list = self.action_real_list
    #                 delta_b0j = final_epoch_action_list
    #                 delta_b0j_conca = delta_b0j

    #                 delta_b0_conca = [0] * 3*self.joint_num + [0] * 3 * self.joint_rotate_num  + [0] * 3 *self.joint_ooc_l_num  + [0] * 3 * self.joint_ooc_r_num  + [0] * 3 * self.joint_frontalis_num
    #                 # if next_info['success_flag'] == 1:
    #                 #     print("in evaluator l133, hit one success. should early stop now. next_info['success_flag']:", next_info['success_flag'])            
                    
    #                 for i in range(len(delta_b0j_conca)):
    #                     delta_b0_conca = [delta_b0_conca[j] + delta_b0j_conca[i][j] for j in range(3*self.joint_num + 3*self.joint_rotate_num + 3 *self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num)]
    #                 return delta_b0_conca

    #             delta_b0_conca = calculate_accumulated_rotated_angle()
    #             # self.accumulated_rotated_angle.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])
    #             if phase_2_flag == 1:
    #                 self.accumulated_rotated_angle_phase2.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])
    #             else:
    #                 self.accumulated_rotated_angle.append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])




    #             # calculate the elastic energy incurred by the rotation of jaw.
    #             # theta = cumulative_trapezoid(velocity_rad, t, initial=0)
    #             # base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]

    #             # assume only z axis has non-trival significant rotational angles.
    #             # theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]

    #             # if self.this_task in self.contract_to_neutral_task:
    #             #     theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]
    #             # else:
    #             #     theta = self.accumulated_rotated_angle[-1][0]
    #             if self.this_task in self.contract_to_neutral_task:
    #                 theta = abs(self.rest_pose_rotational_position_degree) - self.accumulated_rotated_angle[-1][0]
    #             else:
    #                 if phase_2_flag == 0:
    #                     theta = self.accumulated_rotated_angle[-1][0]
    #                 else:                        
    #                     theta = self.accumulated_rotated_angle_phase2[-1][0]                                    

    #             def degree_conver_to_rad(theta):
    #                 return theta * np.pi / 180  # Convert °/s to rad/s
    #             theta_rad = degree_conver_to_rad(theta)
    #             k_tissue = 10000  # Stiffness of soft tissues (N/m)
    #             r = 0.17  # Radius of rotation (m)        
    #             # Calculate displacement (x) as r * theta
    #             x = r * theta_rad        

    #             # Elastic energy function
    #             def elastic_energy():
    #                 return 0.5 * k_tissue * x**2
                
    #             E_elastic = elastic_energy()
    #             self.elastic_energy.append(E_elastic)  

    #         elastic()

    #         T_prime_real = 35 / 30
    #         T_prime = T_prime_real * self.step_counter_internal/self.expected_total_steps
    #         # Ensure work_rate is computed as element-wise addition
    #         work_rate = np.array(self.rotational_kinetic_energy) + np.array(self.elastic_energy)        
    #         # work_rate = self.rotational_kinetic_energy + self.elastic_energy
    #         t = np.linspace(0, T_prime, len(work_rate))  # Total time for jaw movement

    #         # Calculate total accumulated work
    #         W_total = cumulative_trapezoid(work_rate, t, initial=0)

    #         W_total_at_last = W_total[-1]

    #         return W_total_at_last


    #     if self.add_energy_to_jaw_objective_flag == 1:
    #         accumulated_work_at_last = calculate_accumulated_work()
    #     else:
    #         accumulated_work_at_last = 0


    #     relative_rotation_value = -rot_forward

    #     # rotate. 
        
    #     for i in range(0,self.joint_rotate_num):  #3-3,18-3
    #         flag_temp1 = relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0
    #         # flag_temp2 = np.array_equal(self.x_joint_rotation[3*i+1],x_joint_rotation_[3*i+1])
    #         # flag_temp3 = np.array_equal(self.x_joint_rotation[3*i+2],x_joint_rotation_[3*i+2])
    #         # print("rotate flag_temp1, ", flag_temp1)
    #         # print("flag_temp2, ", flag_temp2)
    #         # print("flag_temp3, ", flag_temp3)
    #         if not flag_temp1:  
    #             # self.cmds.rotate( relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws = True)
    #             # print("really rotate l4230")


    #             try:
                    
    #                 self.cmds.rotate( relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws = True)
    #                 # print("really rotate l4280")
    #                 #bk2
    #                 # temporary manually lock rotate y and z
    #             except Exception as e:
    #                 # print(f"l332 at env, unimportant error, An error occurred: {e}")
    #                 if relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0:
    #                     try:
    #                         # print("\n\n line 594 joint_jaw_rotate_num:", self.joint_jaw_rotate_num, " \n\n'jaw0'+str(i+1)", '\n\njaw0', str(i+1))
    #                         self.cmds.rotate( 0, 0, 0, 'jaw0'+str(i+1), relative=True, ws = True)

    #                     except Exception as e3:
    #                         print(f"An error occurred: {e3}")
    #                         pass         


    #     def undo_this_append():
    #         # Pop elements without storing the returned values
    #         if self.rotational_kinetic_energy:
    #             self.rotational_kinetic_energy.pop()
            
    #         if self.action_real_list:
    #             self.action_real_list.pop()
            
    #         if self.accumulated_rotated_angle:
    #             self.accumulated_rotated_angle.pop()
            
    #         if self.elastic_energy:
    #             self.elastic_energy.pop()                         

    #         if phase_2_flag == 1:
    #             if self.elastic_energy:
    #                 self.accumulated_rotated_angle_phase2.pop()
    #         else:
    #             if self.accumulated_rotated_angle:
    #                 self.accumulated_rotated_angle.pop()


    #     if self.add_energy_to_jaw_objective_flag == 1:
    #         undo_this_append()
    #     else:
    #         pass
        

    #     self.cmds.select(cl=True)
    #     self.cmds.currentCtx()
    #     # Close the reference            

    #     return new_coords_jaw_forward, accumulated_work_at_last     
    
    def return_coordinates_adhoc(self, rot_forward, current_phase=1):

        relative_rotation_value = rot_forward

        for i in range(0,self.joint_rotate_num):
            flag_temp1 = relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0
            if not flag_temp1:  
                try:
                    self.cmds.rotate(relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws=True)
                except Exception as e:
                    print(f"l332 at env, unimportant error, An error occurred: {e}")
                    if relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0:
                        try:
                            self.cmds.rotate(0, 0, 0, 'jaw0'+str(i+1), relative=True, ws=True)
                        except Exception as e3:
                            print(f"An error occurred: {e3}")

        new_coords_jaw_forward = self.returnCoordinates("middleres_Jaw", self.vertex_jaw_index)


        def calculate_accumulated_work():
            def kinetic():
                elements = relative_rotation_value
                rotational_velocity_average = sum(elements)
                if self.fix_flag_ship[1] == 1:
                    rotation_velocity = 0
                    rotational_kinetic_energy = 0     
                else:                   
                    rotation_velocity = rotational_velocity_average
                    rotation_velocity_prime = rotation_velocity * math.pi / 180
                    I = 1/3 * 0.1089 * 0.17**2
                    E_rotation = 1/2 * I * rotation_velocity_prime**2
                    rotational_kinetic_energy = E_rotation * 1.0  # convert_factor

                # Get phase-specific data storage
                phase_data = self._get_phase_specific_data(current_phase)
                phase_data['rotational_kinetic_energy'].append(rotational_kinetic_energy)

            kinetic()

            def elastic():


                action = np.array([0.0] * 3 * self.joint_num + 
                                [0.0] * 3 * self.joint_rotate_num + 
                                [0.0] * 3 * self.joint_ooc_l_num + 
                                [0.0] * 3 * self.joint_ooc_r_num + 
                                [0.0] * 3 * self.joint_frontalis_num)
                action[self.joint_num * 3:self.joint_num * 3 + self.joint_rotate_num * 3] = relative_rotation_value
                base_action_prime = np.array([0.0]*3*self.joint_num + [0.0]*3*self.joint_rotate_num + [0.0]*3*self.joint_ooc_l_num + [0.0]*3*self.joint_ooc_r_num + [0.0]*3*self.joint_frontalis_num)
                
                for actionI in range(3*self.joint_num, 3*self.joint_num + self.joint_rotate_num * 3):         
                    base_action_prime[int(actionI)] = action[int(actionI)]                    
                if self.fix_flag_ship[1] == 1:
                    base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num] = 0
                                    
                self.action_real_list.append(base_action_prime)

                delta_b0_conca = self.calculate_accumulated_rotated_angle()

                # Store in phase-specific data
                phase_data = self._get_phase_specific_data(current_phase)
                phase_data['angles'].append(delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num])

                # if self.this_task in self.contract_to_neutral_task:
                if False:
                    # theta = abs(self.rest_pose_rotational_position_degree) - phase_data['angles'][-1][0]
                    pass
                else:
                    theta = phase_data['angles'][-1][0]

                def degree_conver_to_rad(theta):
                    return theta * np.pi / 180
                theta_rad = degree_conver_to_rad(theta)
                k_tissue = 10000
                r = 0.17        
                x = r * theta_rad        

                def elastic_energy():
                    return 0.5 * k_tissue * x**2
                
                E_elastic = elastic_energy()
                phase_data['elastic_energy'].append(E_elastic)

            elastic()

            phase_data = self._get_phase_specific_data(current_phase)
            T_prime_real = 35 / 30
            T_prime = T_prime_real * self.step_counter_internal/self.expected_total_steps
            work_rate = np.array(phase_data['rotational_kinetic_energy']) + np.array(phase_data['elastic_energy'])
            t = np.linspace(0, T_prime, len(work_rate))
            W_total = cumulative_trapezoid(work_rate, t, initial=0)
            return W_total[-1]

        if self.add_energy_to_jaw_objective_flag == 1:
            accumulated_work_at_last = calculate_accumulated_work()
        else:
            accumulated_work_at_last = 0

        relative_rotation_value = -rot_forward

 

        for i in range(0,self.joint_rotate_num):
            flag_temp1 = relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0
            if not flag_temp1:  
                try:
                    self.cmds.rotate(relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws=True)
                except Exception as e:
                    if relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0:
                        try:
                            self.cmds.rotate(0, 0, 0, 'jaw0'+str(i+1), relative=True, ws=True)
                        except Exception as e3:
                            print(f"An error occurred: {e3}")



        def undo_this_append():
            phase_data = self._get_phase_specific_data(current_phase)
            if phase_data['rotational_kinetic_energy']:
                phase_data['rotational_kinetic_energy'].pop()
            
            if self.action_real_list:
                new_action = self.action_real_list[-1]
                self._delta_b0_conca_accumulator = self._delta_b0_conca_accumulator - np.array(new_action, dtype=np.float32)
                self.action_real_list.pop()
            
            if phase_data['angles']:
                phase_data['angles'].pop()
            
            if phase_data['elastic_energy']:
                phase_data['elastic_energy'].pop()

        if self.add_energy_to_jaw_objective_flag == 1:
            undo_this_append()


        self.cmds.select(cl=True)
        self.cmds.currentCtx()



        return new_coords_jaw_forward, accumulated_work_at_last

    def update_expected_total_steps(self, error, threshold):
        """
        Dynamically adjust self.expected_total_steps based on convergence.
        """
        # Calculate the difference between current and target states
        # error = np.linalg.norm(current_state - target_state)
        
        # If the error is below the threshold, stop early
        # if error <= threshold:
        #     self.expected_total_steps = self.step_counter_internal
        #     return
        # Estimate remaining steps based on the current convergence rate
        convergence_rate = (self.initial_distance - error) / self.step_counter_internal

        # Avoid division by zero or very small convergence_rate
        if abs(convergence_rate) > 1e-10:  # Check if convergence_rate is not too small
            self.remaining_steps = int(abs(error - threshold) / convergence_rate)
        else:
            # Bypass the calculation if convergence_rate is too small
            # self.remaining_steps = None  # Or set it to a default value, e.g., 1000
            pass

        # Estimate remaining steps based on the current convergence rate
        # convergence_rate = (self.initial_distance - error) / self.step_counter_internal
        # convergence_rate = error / self.step_counter_internal
        # self.remaining_steps = int( abs(error - threshold) / convergence_rate)
        
        # Update self.expected_total_steps

        adjust_factor = 1000
        self.expected_total_steps = self.step_counter_internal + self.remaining_steps
        
        if self.step_counter_internal < 1000:
        
            self.expected_total_steps = self.expected_total_steps + adjust_factor * 4

        if self.step_counter_internal  >= 1000 and self.step_counter_internal < 2000:

            self.expected_total_steps = self.expected_total_steps + adjust_factor * 3

        if self.step_counter_internal  >= 2000 and self.step_counter_internal < 3000:

            self.expected_total_steps = self.expected_total_steps + adjust_factor * 2

        if self.step_counter_internal  >= 3000 and self.step_counter_internal < 4000:
            self.expected_total_steps = self.expected_total_steps + adjust_factor * 1

        if self.step_counter_internal  >= 4000:
            if self.step_counter_internal  > self.expected_total_steps *3/4:
                pass
            else:
                self.expected_total_steps = self.expected_total_steps + adjust_factor * 1



        self.expected_total_steps = np.clip(self.expected_total_steps, self.expected_total_steps_initial - self.expected_total_steps_initial * 0.4, self.expected_total_steps_initial + self.expected_total_steps_initial * 5)

    def suc_position_save(self):

        # # Redirect stdout and stderr to null
        # sys.stdout = open(os.devnull, 'w')
        # sys.stderr = open(os.devnull, 'w')

        # try:
        #     if not self.opened_file_status:
        #         if self.cmds.file(query=True, modified=True):
        #             print("Unsaved changes detected.")
        #             if 1==0:  # save_changes in ['yes', 'y']:
        #                 self.cmds.file(save=True)
        #                 print("Changes saved.")
        #             else:
        #                 self.cmds.file(newFile=True, force=True)

        #         try:
        #             self.opened_file = self.cmds.file(self.maya_file_to_open, open=True, force=True)
        #             print("Opened file:", self.opened_file)
        #             self.cmds.playbackOptions(min=1, max=50, animationStartTime=1, animationEndTime=50)

        #             def delete_cache_nodes():
        #                 cache_nodes = self.cmds.ls(type='cacheFile')
        #                 for node in cache_nodes:
        #                     self.cmds.delete(node)

        #             delete_cache_nodes()
        #             self.cmds.cacheFile(createCacheNode=True, singleCache=True, startTime=1, endTime=50, fileName='myCache', dir='cacheDir')
        #             self.cmds.currentTime(15)

        #         except Exception as e:
        #             print(f"Failed to open Maya file: {e}")

        #         self.opened_file_status = True

        # finally:
        #     sys.stdout = sys.__stdout__
        #     sys.stderr = sys.__stderr__
			

        s_next_full = self.returnCoordinates("middleres_OrbicularisOris")
        s_jaw_next_full = self.returnCoordinates("middleres_Jaw")
        # s_ooc_l_next_full = self.returnCoordinates("middleres_OrbicularisOculi_r", self.vertex_ooc_r_index)
        # s_ooc_r_next_full = self.returnCoordinates("middleres_OrbicularisOculi_l", self.vertex_ooc_l_index)
        # s_frontalis_next_full = self.returnCoordinates("middleres_Frontalis", self.vertex_frontalis_index)      
        # s_ooc_l_next_full = getattr(dataV22, "x_middleres_OrbicularisOculi_l" + str(0)) # did not change, so we use the original one.
        # s_ooc_l_next_full=s_ooc_l_next_full.flatten()
        # s_ooc_r_next_full = getattr(dataV22, "x_middleres_OrbicularisOculi_r" + str(0))
        # s_ooc_r_next_full = s_ooc_r_next_full.flatten()
        # s_frontalis_next_full = getattr(dataV22, "x_middleres_Frontalis" + str(0))         
        # s_frontalis_next_full = s_frontalis_next_full.flatten()
        # s_ooc_l_next_full = np.array(getattr(dataV22, "x_middleres_OrbicularisOculi_l" + str(0))).flatten()
        # s_ooc_r_next_full = np.array(getattr(dataV22, "x_middleres_OrbicularisOculi_r" + str(0))).flatten()
        # s_frontalis_next_full = np.array(getattr(dataV22, "x_middleres_Frontalis" + str(0))).flatten()        
        s_ooc_l_next_full = self.returnCoordinates("middleres_OrbicularisOculi_l")
        s_ooc_r_next_full = self.returnCoordinates("middleres_OrbicularisOculi_r")
        s_frontalis_next_full = self.returnCoordinates("middleres_Frontalis")       
        
        self.cmds.select(cl=True)
        self.cmds.currentCtx()        
                 
        return s_next_full, s_jaw_next_full, s_ooc_l_next_full, s_ooc_r_next_full, s_frontalis_next_full        
    
    def deform_rest_to_previous_pose(self):


        

        with open(self.previous_pose_folder + "/deltaBList_temp.pkl", 'rb') as f:
            data_t = pickle.load(f)
            deltaBList = data_t['deltaBList']
        
        delta_b0_conca = deltaBList[-1]        



        base_action_prime = np.array([0.0]*3*self.joint_num + [0.0]*3*self.joint_rotate_num + [0.0]*3*self.joint_ooc_l_num + [0.0]*3*self.joint_ooc_r_num + [0.0]*3*self.joint_frontalis_num)       

        # print("size of action", len(action))
        # print("size of base_action_prime", len(base_action_prime))

        # s_prime_return_coordinates = self.returnCoordinates()

        # to bypass rotation for now
        # base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num] = 0
        # # to bypass oor for now
        # base_action_prime[0:3*self.joint_num] = 0
        # to bypass ooc for now
        base_action_prime[3*self.joint_num + 3*self.joint_rotate_num:3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num] = 0
        base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num] = 0

        # to bypass frontalis for now
        base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num] = 0

        x_joint_first_wave = getattr(joint_position_first_wave, "x_joint_position_1")
        x_joint_second_wave = getattr(joint_position, "x_joint_position_1")
        x_joint_third_wave = getattr(joint_position_third_wave, "x_joint_position_1")
        x_joint_real_rest  = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave
        x_joint_conca = [item for sublist in x_joint_real_rest for item in sublist]
        x_joint_init_conca = x_joint_conca
        x_joint_init = np.array(x_joint_init_conca)
        # x_joint_ = x_joint_init + delta_b0_conca[0:n]

        x_joint_rotation_real_rest = getattr(dataV22, "x_joint_rotation_cancatenate" + str(0))
        x_joint_rotation_conca = [item for sublist in x_joint_rotation_real_rest for item in sublist]
        x_joint_rotation_init_conca = x_joint_rotation_conca
        x_joint_rotation_init = np.array(x_joint_rotation_init_conca)     
        # x_joint_rotation_ = x_joint_rotation_init + delta_b0_conca[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]



        if "both" == self.action_mode:
            x_joint_ = x_joint_init + delta_b0_conca[0:3*self.joint_num]
            x_joint_rotation_ = x_joint_rotation_init + delta_b0_conca[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]
            x_ooc_l_joint_ = self.x_ooc_l_joint_init + base_action_prime[3*self.joint_num + 3*self.joint_rotate_num:3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num]
            x_ooc_r_joint_ = self.x_ooc_r_joint_init + base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num]
            x_frontalis_joint_ = self.x_frontalis_joint_init + base_action_prime[3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num : 3*self.joint_num + 3*self.joint_rotate_num + 3 * self.joint_ooc_l_num + 3 * self.joint_ooc_r_num + 3 * self.joint_frontalis_num]



        # relative_rotation_value = base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]
        relative_rotation_value = delta_b0_conca[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]
        # rotate. 
        if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
            for i in range(0,self.joint_rotate_num):  #3-3,18-3
                flag_temp1 = np.array_equal(x_joint_rotation_init[3*i],x_joint_rotation_[3*i])
                flag_temp2 = np.array_equal(x_joint_rotation_init[3*i+1],x_joint_rotation_[3*i+1])
                flag_temp3 = np.array_equal(x_joint_rotation_init[3*i+2],x_joint_rotation_[3*i+2])
                # print("rotate flag_temp1, ", flag_temp1)
                # print("flag_temp2, ", flag_temp2)
                # print("flag_temp3, ", flag_temp3)
                if (not flag_temp1) or (not flag_temp2) or (not flag_temp3):  
                    # self.cmds.rotate( relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws = True)
                    # print("really rotate l423")


                    try:
                        
                        self.cmds.rotate( relative_rotation_value[3*i], relative_rotation_value[3*i+1], relative_rotation_value[3*i+2], 'jaw0'+str(i+1), relative=True, ws = True)
                        # print("really rotate l428")
                        #bk2
                        # temporary manually lock rotate y and z
                    except Exception as e:
                        # print(f"l332 at env, unimportant error, An error occurred: {e}")
                        if relative_rotation_value[3*i] == 0 and relative_rotation_value[3*i+1] == 0 and relative_rotation_value[3*i+2] == 0:
                            try:
                                # print("\n\n line 594 joint_jaw_rotate_num:", self.joint_jaw_rotate_num, " \n\n'jaw0'+str(i+1)", '\n\njaw0', str(i+1))
                                self.cmds.rotate( 0, 0, 0, 'jaw0'+str(i+1), relative=True, ws = True)

                            except Exception as e3:
                                print(f"An error occurred: {e3}")
                                pass 


        # translate
        if self.debug_mode_vsCode == 0 or self.debug_mode_vsCode == 1:
            for i in range(self.joint_num):
                flag_temp1 = np.array_equal(x_joint_init[3 * i], x_joint_[3 * i])
                flag_temp2 = np.array_equal(x_joint_init[3 * i + 1], x_joint_[3 * i + 1])
                flag_temp3 = np.array_equal(x_joint_init[3 * i + 2], x_joint_[3 * i + 2])
                if not flag_temp1 or not flag_temp2 or not flag_temp3:
                    # Determine the joint name based on the index i
                    if i + 1 <= 9:
                        joint_name = f'oor_0{i + 1}'  # Leading zero for single-digit numbers
                    elif i + 1 <= 99:
                        joint_name = f'oor_{i + 1}'
                    elif i + 1 <= 999:
                        joint_name = f'oor_{i + 1}'
                    else:
                        joint_name = f'oor_{i + 1}'

                    # Move the joint
                    if 1==0: # joint_name == "lip_lower191":
                        # print("lip_lower191 x_joint_", x_joint_)
                        
                        # print("lip_lower191 self.x_joint", self.x_joint)
                        # print("lip_lower191 size self.x_joint", len(self.x_joint))
                        # print("lip_lower191 base_action_prime[0:3*self.joint_num]", base_action_prime[0:3*self.joint_num])
                        print("lip_lower191 size base_action_prime[0:3*self.joint_num]", len(base_action_prime[0:3*self.joint_num]))
                        # self.cmds.move(self.x_joint[3 * i], x_joint_[3 * i + 1], self.x_joint[3 * i + 2], joint_name, absolute=True)
                        objectList = "lip_lower191"
                        position_t = self.cmds.xform(objectList, query=True, worldSpace=True, translation=True)
                        print(position_t)

                        x_joint_init_conca_t = position_t
                        i_temp2 = 0
                        self.cmds.move( x_joint_init_conca_t[3*i_temp2]+0.17, x_joint_init_conca_t[3*i_temp2+1], x_joint_init_conca_t[3*i_temp2+2], joint_name, absolute=True)
                    else: 
                        # print("joint_name to deform this time", joint_name)
                        self.cmds.move(x_joint_[3 * i], x_joint_[3 * i + 1], x_joint_[3 * i + 2], joint_name, absolute=True)



    # # Modified plotting and data saving functions
    # def _get_phase_specific_data(self, phase_2_flag):
    #     """Helper function to get phase-specific data containers"""
    #     if phase_2_flag == 1:
    #         return {
    #             'distance_log': self.distance_log_phase2,
    #             'step_counter_internal_log': self.step_counter_internal_log_phase2,
    #             'strain_energy': self.strain_energy_phase2,
    #             'time_for_energy_log': self.time_for_energy_log_phase2,
    #             'rotational_kinetic_energy': self.rotational_kinetic_energy_phase2,
    #             'time_for_rotational_kinetic_energy': self.time_for_rotational_kinetic_energy_phase2,
    #             'elastic_energy': self.elastic_energy_phase2,
    #             'rotation_velocity': self.rotation_velocity_phase2,
    #             'accumulated_rotated_angle': self.accumulated_rotated_angle_phase2
    #         }
    #     else:
    #         return {
    #             'distance_log': self.distance_log,
    #             'step_counter_internal_log': self.step_counter_internal_log,
    #             'strain_energy': self.strain_energy,
    #             'time_for_energy_log': self.time_for_energy_log,
    #             'rotational_kinetic_energy': self.rotational_kinetic_energy,
    #             'time_for_rotational_kinetic_energy': self.time_for_rotational_kinetic_energy,
    #             'elastic_energy': self.elastic_energy,
    #             'rotation_velocity': self.rotation_velocity,
    #             'accumulated_rotated_angle': self.accumulated_rotated_angle
    #         }
        
    # def _get_phase_specific_data(self, phase_2_flag):
    #     """Helper function to get phase-specific data containers"""
    #     if phase_2_flag == 1:
    #         return {
    #             'distance_log': self.distance_log_phase2,
    #             'step_counter_internal_log': self.step_counter_internal_log_phase2,
    #             'strain_energy': self.strain_energy_phase2,
    #             'time_for_energy_log': self.time_for_energy_log_phase2,
    #             'rotational_kinetic_energy': self.rotational_kinetic_energy_phase2,
    #             'time_for_rotational_kinetic_energy_log': self.time_for_rotational_kinetic_energy_log_phase2,
    #             'elastic_energy': self.elastic_energy_phase2,
    #             'rotation_velocity': self.rotation_velocity_phase2,
    #             'accumulated_rotated_angle': self.accumulated_rotated_angle_phase2,
    #             'base_action_prime_log': self.base_action_prime_log_phase2  # New addition
    #         }
    #     else:
    #         return {
    #             'distance_log': self.distance_log,
    #             'step_counter_internal_log': self.step_counter_internal_log,
    #             'strain_energy': self.strain_energy,
    #             'time_for_energy_log': self.time_for_energy_log,
    #             'rotational_kinetic_energy': self.rotational_kinetic_energy,
    #             'time_for_rotational_kinetic_energy_log': self.time_for_rotational_kinetic_energy_log,
    #             'elastic_energy': self.elastic_energy,
    #             'rotation_velocity': self.rotation_velocity,
    #             'accumulated_rotated_angle': self.accumulated_rotated_angle,
    #             'base_action_prime_log': self.base_action_prime_log  # New addition
    #         }

    def _get_phase_specific_data(self, current_phase=1):
        """Get all log data for a specific phase"""
        phase_logs = self._get_phase_logs(current_phase)
        return {
            'distance_log': phase_logs['distance'],
            'step_counter_internal_log': phase_logs['step_counter_internal'],
            'strain_energy': phase_logs.get('strain_energy', []),
            'time_for_energy': phase_logs['time_for_energy'],
            'rotational_kinetic_energy': phase_logs['rotational_kinetic_energy'],
            'time_for_rotational_kinetic_energy': phase_logs['time_for_rotational_kinetic_energy'],
            'elastic_energy': phase_logs['elastic_energy'],
            'rotation_velocity': phase_logs['rotation_velocity'],
            'accumulated_rotated_angle': phase_logs['angles'],
            'base_action_prime': phase_logs['base_action_prime']
        }               

    # def _get_log_filename(self, phase_2_flag):
    #     """Helper function to get phase-specific log filename"""
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
    #     return f"{self.save_folder}/log{str(self.i_func)}{phase_suffix}.txt"        
    
    def _get_log_filename(self, current_phase=1):
        if not isinstance(current_phase, int) or current_phase < 1:
            raise ValueError("current_phase must be a positive integer")
        
        phase_suffix = f"_phase{current_phase}" if current_phase > 1 else ""
        return f"{self.save_folder}/log{str(self.i_func)}{phase_suffix}.txt"
    
    # def log_step_details(self, phase_2_flag=0, step_counter=None, s_prime=None, 
    #                     base_action_prime=None, s_next=None, x_joint=None, 
    #                     x_joint_=None, x_joint_rotation=None, x_joint_rotation_=None,
    #                     distance=None, distance_jaw=None, distance_ooc_r=None,
    #                     distance_ooc_l=None, distance_frontalis=None, reward=None,
    #                     expected_total_steps=None, step_counter_internal=None,
    #                     remaining_steps=None):
    #     """Log detailed step information"""
    #     with open(self._get_log_filename(phase_2_flag), 'a') as f:
    #         print(f"Step {step_counter},", file=f)
    #         print(f"current State[0:12]: {s_prime[0:12]},", file=f)
    #         print(f"Action[0:12]: {base_action_prime[0:12]},", file=f)
    #         print(f"next State[0:12]: {s_next[0:12]},", file=f)
    #         print(f"Target State[0:12]: {self.control_vertices_slice(self.target_state, self.vertex_index)[0:12]},", file=f)
    #         print(f"current joint position[0:12]: {x_joint[0:12]},", file=f)
    #         print(f"next joint position[0:12]: {x_joint_[0:12]},", file=f)
    #         print(f"current joint rotation position: {x_joint_rotation},", file=f)
    #         print(f"next joint rotation position: {x_joint_rotation_}", file=f)
    #         print(f"distance after action {distance}", file=f)
    #         print(f"distance jaw after action {distance_jaw}", file=f)
    #         print(f"rotation action {base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]}", file=f)
    #         print(f"distance ooc_r after action {distance_ooc_r}", file=f)
    #         print(f"distance ooc_l after action {distance_ooc_l}", file=f)
    #         print(f"distance frontalis after action {distance_frontalis}", file=f)
    #         print(f"reward {reward}", file=f)
    #         print(f"expected_total_steps, step_counter_internal, remaining_steps: {expected_total_steps}, {step_counter_internal}, {remaining_steps}", file=f)    

    # def log_step_details(self, current_phase=1, step_counter=None, s_prime=None, 
    #                     base_action_prime=None, s_next=None, x_joint=None, 
    #                     x_joint_=None, x_joint_rotation=None, x_joint_rotation_=None,
    #                     distance=None, distance_jaw=None, distance_ooc_r=None,
    #                     distance_ooc_l=None, distance_frontalis=None, reward=None,
    #                     expected_total_steps=None, step_counter_internal=None,
    #                     remaining_steps=None):
    #     """Log detailed step information for any phase (1, 2, 3, ...)"""
    #     with open(self._get_log_filename(current_phase), 'a') as f:
    #         print(f"Step {step_counter},", file=f)
    #         print(f"current State[0:12]: {s_prime[0:12]},", file=f)
    #         print(f"Action[0:12]: {base_action_prime[0:12]},", file=f)
    #         print(f"next State[0:12]: {s_next[0:12]},", file=f)
    #         print(f"Target State[0:12]: {self.control_vertices_slice(self.target_state, self.vertex_index)[0:12]},", file=f)
    #         print(f"current joint position[0:12]: {x_joint[0:12]},", file=f)
    #         print(f"next joint position[0:12]: {x_joint_[0:12]},", file=f)
    #         print(f"current joint rotation position: {x_joint_rotation},", file=f)
    #         print(f"next joint rotation position: {x_joint_rotation_}", file=f)
    #         print(f"distance after action {distance}", file=f)
    #         print(f"distance jaw after action {distance_jaw}", file=f)
    #         print(f"rotation action {base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]}", file=f)
    #         print(f"distance ooc_r after action {distance_ooc_r}", file=f)
    #         print(f"distance ooc_l after action {distance_ooc_l}", file=f)
    #         print(f"distance frontalis after action {distance_frontalis}", file=f)
    #         print(f"reward {reward}", file=f)
    #         print(f"expected_total_steps, step_counter_internal, remaining_steps: {expected_total_steps}, {step_counter_internal}, {remaining_steps}", file=f)
            
    def log_step_details(self, current_phase=1, step_counter=None, s_prime=None, 
                            base_action_prime=None, s_next=None, x_joint=None, 
                            x_joint_=None, x_joint_rotation=None, x_joint_rotation_=None,
                            distance=None, distance_jaw=None, distance_ooc_r=None,
                            distance_ooc_l=None, distance_frontalis=None, reward=None,
                            expected_total_steps=None, step_counter_internal=None,
                            remaining_steps=None):
            """Log detailed step information for any phase (1, 2, 3, ...)"""
            with open(self._get_log_filename(current_phase), 'a') as f:
                print(f"Step {step_counter},", file=f)
                print(f"current State[0:12]: {s_prime[0:12]},", file=f)
                print(f"Action[0:12]: {base_action_prime[0:12]},", file=f)
                print(f"next State[0:12]: {s_next[0:12]},", file=f)
                print(f"Target State[0:12]: {self.control_vertices_slice(self.target_state, self.vertex_index)[0:12]},", file=f)
                print(f"current joint position[0:12]: {x_joint[0:12]},", file=f)
                print(f"next joint position[0:12]: {x_joint_[0:12]},", file=f)
                print(f"current joint rotation position: {x_joint_rotation},", file=f)
                print(f"next joint rotation position: {x_joint_rotation_}", file=f)
                print(f"distance after action {distance}", file=f)
                print(f"distance jaw after action {distance_jaw}", file=f)
                print(f"rotation action {base_action_prime[3*self.joint_num : 3*self.joint_num + 3*self.joint_rotate_num]}", file=f)
                print(f"distance ooc_r after action {distance_ooc_r}", file=f)
                print(f"distance ooc_l after action {distance_ooc_l}", file=f)
                print(f"distance frontalis after action {distance_frontalis}", file=f)
                print(f"reward {reward}", file=f)
                print(f"expected_total_steps, step_counter_internal, remaining_steps: {expected_total_steps}, {step_counter_internal}, {remaining_steps}", file=f)
                
                # --- Begin Estimated Steps Calculation ---
                
                # Note: self.initial_distance, self.distance_min, etc., 
                # are assumed to be accessible as class attributes.
                
                # Helper function for estimation
                def estimate_steps(current_dist, min_dist, initial_dist, steps_taken):
                    if current_dist <= min_dist:
                        return 0
                    # Avoid division by zero if no distance has been reduced
                    reduced_dist = initial_dist - current_dist
                    if reduced_dist <= 1e-9: # Check for near-zero
                        return float('inf') 
                    return int(round((current_dist - min_dist) * steps_taken / reduced_dist))

                # Estimate for OOR (Overall distance)
                steps_left_oor = estimate_steps(distance, self.distance_min, self.initial_distance, step_counter)

                # Estimate for OOC_R
                steps_left_ooc_r = estimate_steps(distance_ooc_r, self.distance_ooc_r_min, self.initial_ooc_r_distance, step_counter)

                # Estimate for OOC_L
                steps_left_ooc_l = estimate_steps(distance_ooc_l, self.distance_ooc_l_min, self.initial_ooc_l_distance, step_counter)
                
                print(f"estimated steps left for OOR: {steps_left_oor}", file=f)
                print(f"estimated steps left for OOC_R: {steps_left_ooc_r}", file=f)
                print(f"estimated steps left for OOC_L: {steps_left_ooc_l}", file=f)
                
                # --- End Estimated Steps Calculation ---
                
   

    def log_final_distances(self, current_phase=1, distance_jaw=None, distance=None,
                        distance_ooc_l=None, distance_ooc_r=None, distance_frontalis=None):
        """Log final distance information"""
        print(f"final distance_jaw is: {distance_jaw}")
        with open(self._get_log_filename(current_phase), 'a') as f:
            print(f"final distance_jaw is: {distance_jaw}, distance: {distance}, min dis: {self.distance_min},", file=f)
            print(f"distance_ooc_l: {distance_ooc_l}, min distance_ooc_l: {self.distance_ooc_l_min},", file=f)
            print(f"distance_ooc_r: {distance_ooc_r}, min distance_ooc_r: {self.distance_ooc_r_min},", file=f)
            print(f"distance_frontalis: {distance_frontalis}, min distance_frontalis: {self.distance_frontalis_min}", file=f)            

    # def plot_sigmoid_limit_progression(self, t_limit_sigmoid_1000_endList, phase_2_flag=0):
    #     """Plot how the limit changes over time"""
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     # Simulate progression
    #     # test_steps = np.linspace(0, 2000, 200)
    #     # test_steps = np.linspace(0, 1000, 200)
    #     # limits = [self.UNIT_PRIME * (0.25 + 0.75 * (min(s/2000,1)**2*(3-2*min(s/2000,1)))) for s in test_steps]
    #     steps = np.arange(0, len(t_limit_sigmoid_1000_endList))
    #     # steps = np.linspace(0, 1000, len(t_limit_sigmoid_1000_endList))
    #     # limits = t_limit_sigmoid_1000_endList[index_temp]
        
    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.plot(steps, t_limit_sigmoid_1000_endList, label='Sigmoid limit progression', color='red', linewidth=2)
    #     plt.xlabel('Steps', fontsize=14)
    #     plt.ylabel('Limit Value', fontsize=14)
    #     plt.title('Sigmoid Limit Progression', fontsize=16)
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.savefig(os.path.join(self.save_folder, f'sigmoid_limit_progression{phase_suffix}.png'),
    #             bbox_inches='tight', dpi=300)
    #     plt.close()            

    def plot_sigmoid_limit_progression(self, t_limit_sigmoid_1000_endList, current_phase=1):
        """Plot how the limit changes over time"""
        phase_suffix = f"_phase{current_phase}"  # Now supports any phase number
        
        steps = np.arange(0, len(t_limit_sigmoid_1000_endList))
        
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(steps, t_limit_sigmoid_1000_endList, 
                label=f'Phase {current_phase} Sigmoid limit progression', 
                color='red', linewidth=2)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Limit Value', fontsize=14)
        plt.title(f'Phase {current_phase} Sigmoid Limit Progression', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(self.save_folder, f'sigmoid_limit_progression{phase_suffix}.png'),
                bbox_inches='tight', dpi=300)
        plt.close()
        
    # def plot_true_translation_elements(self, phase_2_flag=0):
    #     """Plot the true translation elements over time"""
    #     data = self._get_phase_specific_data(phase_2_flag)
    #     phase_suffix = "_phase2" if phase_2_flag == 1 else ""
        
    #     # Calculate translation magnitudes by summing relevant action components
    #     if phase_2_flag == 0:
    #         translation_magnitudes = [np.sum(np.abs(action[0:self.joint_num * 3])) 
    #                                 for action in self.base_action_prime_log]
    #     else:
    #         translation_magnitudes = [np.sum(np.abs(action[0:self.joint_num * 3])) 
    #                                 for action in self.base_action_prime_log_phase2]            
            
    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
    #     plt.plot(data['step_counter_internal_log'], translation_magnitudes,
    #             label='True Translation Elements\nover optimization steps',
    #             color='green', linewidth=2, linestyle='-')
        
    #     plt.xlabel('Optimization steps', fontsize=14)
    #     plt.ylabel('Translation Magnitude', fontsize=14)
    #     plt.gca().spines[['top', 'right']].set_visible(False)
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
    #     plt.savefig(os.path.join(self.save_folder, f'true_translation_elements_vs_steps_plot{phase_suffix}.png'),
    #             bbox_inches='tight', dpi=300)
    #     plt.close()        

    # def plot_true_translation_elements(self, current_phase=1):
    #     """Plot the true translation elements over time for any phase (1, 2, 3, ...)"""
    #     data = self._get_phase_specific_data(current_phase)
    #     phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
    #     # Get phase-specific action log
    #     phase_data = self._get_phase_specific_data(current_phase)
    #     translation_magnitudes = [np.sum(np.abs(action[0:self.joint_num * 3])) 
    #                             for action in phase_data['base_action_prime']]
            
    #     plt.figure(figsize=(8, 6), dpi=300)
    #     plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
    #     plt.plot(data['step_counter_internal_log'], translation_magnitudes,
    #             label='True Translation Elements\nover optimization steps',
    #             color='green', linewidth=2, linestyle='-')
        
    def plot_true_translation_elements(self, current_phase=1):
        """Plot the true translation elements over time for any phase (1, 2, 3, ...)"""
        data = self._get_phase_specific_data(current_phase)
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        # Debug prints
        # print("\n=== DEBUGGING plot_true_translation_elements ===")
        # print(f"Phase: {current_phase}")
        # print(f"Length of step_counter_internal_log: {len(data['step_counter_internal_log'])}")
        
        # Get phase-specific action log
        translation_magnitudes = [np.sum(np.abs(action[0:self.joint_num * 3])) 
                                for action in data['base_action_prime']]
        
        # print(f"Length of translation_magnitudes: {len(translation_magnitudes)}")
        # print(f"Sample steps: {data['step_counter_internal_log'][:5]}") 
        # print(f"Sample magnitudes: {translation_magnitudes[:5]}")
        
        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        
        plt.plot(data['step_counter_internal_log'], translation_magnitudes,
                label='True Translation Elements\nover optimization steps',
                color='green', linewidth=2, linestyle='-')        
        
        plt.xlabel('Optimization steps', fontsize=14)
        plt.ylabel('Translation Magnitude', fontsize=14)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)
        
        plt.savefig(os.path.join(self.save_folder, f'true_translation_elements_vs_steps_plot{phase_suffix}.png'),
                bbox_inches='tight', dpi=300)
        plt.close()

    # def _get_log_filename_velocity(self, phase_2_flag):
    #     """Get appropriate log filename based on phase"""
    #     phase_suffix = "_phase2" if phase_2_flag else ""
    #     return os.path.join(self.save_folder, f"velocity_debug{phase_suffix}.log")        
    
    def _get_log_filename_velocity(self, current_phase=1):
        """Get appropriate log filename for any phase (1, 2, 3, ...)"""
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        return os.path.join(self.save_folder, f"velocity_debug{phase_suffix}.log")    

    # def _get_angle_storage(self, current_phase):
    #     """Returns the appropriate angle storage list for the current phase"""
    #     return {
    #         1: self.accumulated_rotated_angle,
    #         2: self.accumulated_rotated_angle_phase2
    #     }.get(current_phase, self.accumulated_rotated_angle)    
    
    def _get_angle_storage(self, current_phase):
        """Returns the angle storage list for the current phase, creating it if needed"""
        if current_phase not in self.phase_angle_storage:
            self.phase_angle_storage[current_phase] = []  # Initialize storage for new phase
        return self.phase_angle_storage[current_phase]
    

    def _get_phase_logs(self, phase):
        """Get or create log storage for a specific phase"""
        if phase not in self.phase_logs:
            self.phase_logs[phase] = {
                'distance': [],
                'step_counter_internal': [],
                'time_for_energy': [],
                'rotational_kinetic_energy': [],
                'time_for_rotational_kinetic_energy': [],
                'elastic_energy': [],
                'rotation_velocity': [],
                'base_action_prime': [],
                'angles': []
            }
        return self.phase_logs[phase]    
    
    def save_all_phase_data(self, current_phase):
        self.data_save_distance_vs_time(current_phase)
        self.data_save_energy_vs_time(current_phase)
        self.data_save_angular_velocity_vs_time(current_phase)
        self.data_save_rotational_kinetic_energy_vs_time(current_phase)
        self.data_save_elastic_energy_vs_time_plot(current_phase)
        self.data_save_accumulated_work_done(current_phase)    


    def calculate_accumulated_rotated_angle(self):
        """Incrementally update the accumulator with ONLY the newest action."""
        # Get the latest action (assumed to be the last element in action_real_list)
        new_action = self.action_real_list[-1]  # Directly access the newest action
        
        # Convert to NumPy array (if not already) and add to accumulator
        self._delta_b0_conca_accumulator += np.array(new_action, dtype=np.float32)
        
        return self._delta_b0_conca_accumulator                
    

    def debug_rotation_angle_speedup(self, current_phase, delta_b0_conca = None):

        if delta_b0_conca is None:
            old_angles = 9999
        else:
            old_angles = delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num]

        """Compare old and new rotation angle calculations"""
        phase_suffix = f"_phase{current_phase}" if current_phase != 1 else ""
        
        # Get both calculation methods
        delta_b0_conca = self.calculate_accumulated_rotated_angle()
        new_angles = delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num]
        # new_angles = self.accumulated_rotated_angle[-1] if self.accumulated_rotated_angle else [0]*3*self.joint_rotate_num
        # old_angles = 1
        
        # Calculate differences
        diff = np.array(old_angles) - np.array(new_angles)
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        
        # Log comparison
        with open(os.path.join(self.save_folder, f"rotation_angle_debug{phase_suffix}.txt"), 'a') as f:
            f.write(f"\n=== Rotation Angle Comparison ===\n")
            f.write(f"Step: {self.step_counter_internal}\n")
            f.write(f"Actions in history: {len(self.action_real_list)}\n")            
            # f.write(f"Actions: {self.action_real_list[:][self.n : self.n + 3*self.joint_rotate_num]}\n")     
            f.write(f"Actions: {[item[self.n : self.n + 3*self.joint_rotate_num] for item in self.action_real_list]}\n")            
            f.write(f"Current accumulator: {self._delta_b0_conca_accumulator[self.n:self.n+3]}\n")                   
            f.write(f"Old Calculation: {old_angles}\n")
            f.write(f"New Calculation: {new_angles}\n")
            f.write(f"Differences (max/mean): {max_diff:.6f}/{mean_diff:.6f}\n")
            
            if max_diff > 1e-6:  # Significant difference threshold
                f.write("WARNING: Significant difference detected!\n")
                f.write(f"Possible causes:\n")
                f.write("- Different phase tracking in new method\n")
                f.write("- Timing of when angles are captured\n")
                f.write("- Floating point accumulation differences\n")
            
            # Additional debug info
            f.write("\nDebug Context:\n")
            f.write(f"Current Phase: {current_phase}\n")
            f.write(f"Joint Rotate Num: {self.joint_rotate_num}\n")
            f.write(f"Slice Index n: {self.n}\n")
            f.write(f"Full delta_b0_conca length: {len(delta_b0_conca)}\n")    


    def apply_smooth_limits_to_actions(self, action, current_phase):
        """
        Applies the smooth, time-dependent limit (double sigmoid) to the
        translational movement components of the action vector (oor, ooc_l, ooc_r, frontalis).
        Each component uses its own configured UNIT_PRIME for its maximum limit.
        
        :param action: The NumPy array representing the current action vector.
        :return: The clipped action array.
        """

        # --- 1. Limit Calculation Function ---
        def get_smooth_limit(unit_prime):
            """
            Calculates the maximum allowable magnitude for the action components,
            eased in over time using a smooth step function, constrained by unit_prime.
            """
            # total_growth_steps is the length of the warm-up phase (e.g., 2000 steps).
            total_growth_steps = self.overlapping_length 
            
            # Normalize progress (t is between 0 and 1)
            t = min(self.step_counter_internal / total_growth_steps, 1.0)
            
            # Smooth step (ease-in-out function: t^2 * (3 - 2t))
            eased_t = t**2 * (3 - 2*t)
            
            # Scale the limit: starts at 0.25 * unit_prime and grows up to unit_prime.
            t_limit = unit_prime * (0.25 + 0.75 * eased_t)
            
            # Final limit ensures it never exceeds unit_prime
            final_limit = min(t_limit, unit_prime)
            
            return final_limit

        self.UNIT_PRIME_OOR = self.phase_params['translation_anatomy_limit'][current_phase]

        self.UNIT_PRIME_OOC_L = self.phase_params['ooc_l_translation_limit'][current_phase]
        self.UNIT_PRIME_OOC_R = self.phase_params['ooc_r_translation_limit'][current_phase] 
        self.UNIT_PRIME_FRONTALIS = self.phase_params['frontalis_translation_limit'][current_phase]        
        # print("self.UNIT_PRIME_OOR", self.UNIT_PRIME_OOR)
        # print("self.UNIT_PRIME_OOC_L", self.UNIT_PRIME_OOC_L)
        # print("self.UNIT_PRIME_OOC_R", self.UNIT_PRIME_OOC_R)
        # print("self.UNIT_PRIME_FRONTALIS", self.UNIT_PRIME_FRONTALIS)
        # --- 2. Calculate the Limit Values (One for each component) ---
        t_limit_oor = get_smooth_limit(self.UNIT_PRIME_OOR)
        # print("t_limit_oor", t_limit_oor)
        t_limit_ooc_l = get_smooth_limit(self.UNIT_PRIME_OOC_L)
        t_limit_ooc_r = get_smooth_limit(self.UNIT_PRIME_OOC_R)
        t_limit_frontalis = get_smooth_limit(self.UNIT_PRIME_FRONTALIS)

        # --- 3. Define Index Offsets ---
        # The action vector is structured as:
        # [OOR_TRANS] + [JOINT_ROTATE] + [OOC_L_TRANS] + [OOC_R_TRANS] + [FRONTALIS_TRANS] + ...

        # Lengths (each movement has 3 dimensions: X, Y, Z)
        len_oor = self.joint_num * 3
        len_rotate = self.joint_rotate_num * 3
        len_ooc_l = self.joint_ooc_l_num * 3
        len_ooc_r = self.joint_ooc_r_num * 3
        len_frontalis = self.joint_frontalis_num * 3

        # --- 4. Apply Clipping to All Segments ---

        # A. OOR / Main Translation 
        action_oor = action[0:len_oor]
        action_oor = np.clip(action_oor, -t_limit_oor, t_limit_oor)
        action[0:len_oor] = action_oor

        # Calculate starting point for the OOC/Frontalis group
        start_offset = len_oor + len_rotate

        # B. OOC_L Translation
        start_ooc_l = start_offset
        end_ooc_l = start_ooc_l + len_ooc_l
        action_ooc_l = action[start_ooc_l:end_ooc_l]
        action_ooc_l = np.clip(action_ooc_l, -t_limit_ooc_l, t_limit_ooc_l)
        action[start_ooc_l:end_ooc_l] = action_ooc_l

        # C. OOC_R Translation
        start_ooc_r = end_ooc_l
        end_ooc_r = start_ooc_r + len_ooc_r
        action_ooc_r = action[start_ooc_r:end_ooc_r]
        action_ooc_r = np.clip(action_ooc_r, -t_limit_ooc_r, t_limit_ooc_r)
        action[start_ooc_r:end_ooc_r] = action_ooc_r

        # D. Frontalis Translation
        start_frontalis = end_ooc_r
        end_frontalis = start_frontalis + len_frontalis
        action_frontalis = action[start_frontalis:end_frontalis]
        action_frontalis = np.clip(action_frontalis, -t_limit_frontalis, t_limit_frontalis)
        action[start_frontalis:end_frontalis] = action_frontalis

        return action, t_limit_oor            