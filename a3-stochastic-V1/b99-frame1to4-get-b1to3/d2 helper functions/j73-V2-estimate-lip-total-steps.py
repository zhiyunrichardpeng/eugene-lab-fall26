import time

import numpy as np
# import torch
import math
import os

from os import path

# from utils.initialization import create_env
# from utils.common_utils import set_seed

import sys
sys.path.insert(0, r"C:\Users\zpa18\keyLabLocal\a2-DSAC-T-for-pipelineV1")
sys.path.insert(0, r"C:\Users\zpa18\keyLabLocal\a2-DSAC-T-for-pipelineV1\example_train")
sys.path.insert(0, r"C:\Users\zpa18\keyLabLocal\a2-DSAC-T-for-pipelineV1\env_gym")
sys.path.insert(0, r"C:\Users\zpa18\keyLabLocal\a1-blendshape-guided-animation-pipeline\database")
sys.path.insert(0, r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub")
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo")


sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\b98-sampling-joint\300joints_suite_good_old")

import importlib

import joint_position
import joint_position_first_wave
import joint_position_third_wave
importlib.reload(joint_position)
importlib.reload(joint_position_first_wave)
importlib.reload(joint_position_third_wave)

from utils.distance_calculator import MeshDistanceCalculator
distance_calculator = MeshDistanceCalculator()



import dataV22_phoneme_N

importlib.reload(dataV22_phoneme_N)
import dataV22_phoneme_T

importlib.reload(dataV22_phoneme_T)
import dataV22_phoneme_AA

importlib.reload(dataV22_phoneme_AA)

import dataV22_phoneme_IY

importlib.reload(dataV22_phoneme_IY)

import dataV22_phoneme_V

importlib.reload(dataV22_phoneme_V)


import dataV22_phoneme_Y

importlib.reload(dataV22_phoneme_Y)


import dataV22_phoneme_UW

importlib.reload(dataV22_phoneme_UW)

import dataV22_AU0_neutral

importlib.reload(dataV22_AU0_neutral)


class JawAngleCalculator:
    def __init__(self):
        # Initialize with your actual vertex indices
        self.vertex_jaw_index = [29, 177, 350]
        self.save_folder = "./debug_results"  # For debug output
        
    def control_vertices_slice(self, target_state, vertex_index):
        """Extract vertex coordinates from target state"""
        combined_slices = []
        for index in vertex_index:
            slice_part = target_state[index*3:index*3+3]
            combined_slices.extend(slice_part)
        return combined_slices
    
    def calculate_normal_vector(self, vertices):
        """
        Calculate normal vector for a triangle defined by 3 vertices
        vertices: list of 9 elements [x1,y1,z1, x2,y2,z2, x3,y3,z3]
        """
        if len(vertices) != 9:
            raise ValueError("Exactly 3 vertices (9 values) required for normal calculation")
        
        # Reshape into 3 points
        points = np.array(vertices).reshape(3, 3)
        
        # Calculate two vectors from the points
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        
        # Calculate cross product to get normal
        normal = np.cross(v1, v2)
        
        # Normalize the vector
        normal = normal / np.linalg.norm(normal)
        return normal
    
    def calculate_angle_between_normals(self, normal1, normal2):
        """Calculate angle between two normal vectors in degrees"""
        dot_product = np.dot(normal1, normal2)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def debug_output(self, normal1, normal2, angle):
        """Save debug information to file"""
        os.makedirs(self.save_folder, exist_ok=True)
        
        with open(os.path.join(self.save_folder, "jaw_angle_debug.txt"), 'a') as f:
            f.write("\n=== Jaw Angle Calculation ===\n")
            f.write(f"Normal vector 1: {normal1}\n")
            f.write(f"Normal vector 2: {normal2}\n")
            f.write(f"Angle between jaws: {angle:.2f} degrees\n")
            f.write("------------------------\n")
    

    def calculate_jaw_angle(self, x_jaw_pose1, x_jaw_pose2):
        """Main method to calculate jaw angle between two poses"""
        # Get vertex coordinates for both poses
        vertices1 = self.control_vertices_slice(x_jaw_pose1, self.vertex_jaw_index)
        vertices2 = self.control_vertices_slice(x_jaw_pose2, self.vertex_jaw_index)
        
        # DEBUG: Print the extracted vertices
        print(f"Vertices1 shape: {len(vertices1)}, values: {vertices1}")
        print(f"Vertices2 shape: {len(vertices2)}, values: {vertices2}")
        
        # Verify we have exactly 9 elements (3 vertices × 3 coordinates)
        if len(vertices1) != 9 or len(vertices2) != 9:
            print(f"Warning: Expected 9 elements (3 vertices × 3 coordinates), got {len(vertices1)} and {len(vertices2)}")
            print("Vertex indices used:", self.vertex_jaw_index)
            
            # Try to handle cases where we might have extra data
            vertices1 = vertices1[:9]  # Take first 9 elements if too many
            vertices2 = vertices2[:9]  # Take first 9 elements if too many
        
        # Calculate normal vectors
        try:
            normal1 = self.calculate_normal_vector(vertices1)
            normal2 = self.calculate_normal_vector(vertices2)
            
            # Calculate angle between normals
            angle = self.calculate_angle_between_normals(normal1, normal2)
            
            # Debug output
            self.debug_output(normal1, normal2, angle)
            
            return angle
            
        except Exception as e:
            print(f"Error calculating jaw angle: {str(e)}")
            print(f"Vertex data causing issue - vertices1: {vertices1}, vertices2: {vertices2}")
            return None  # or raise the exception if you prefer


def calculate_estimated_steps(pose1_module_name, pose2_module_name):
    """
    Calculate estimated completion steps between two poses specified by their module names
    
    Args:
        pose1_module_name (str): Name of first pose module (e.g. "dataV22_phoneme_N")
        pose2_module_name (str): Name of second pose module (e.g. "dataV22_AU0_neutral")
    
    Returns:
        int: Estimated completion steps between the poses
    """
    try:
        # Import the modules dynamically
        pose1_module = globals()[pose1_module_name]
        pose2_module = globals()[pose2_module_name]
        
        # Get the pose data
        target_state_temp1 = getattr(pose1_module, "x_middleres_OrbicularisOris" + str(1))
        target_state_temp2 = getattr(pose2_module, "x_middleres_OrbicularisOris" + str(1))
        
        # Flatten the pose data
        x_conca1 = [item for sublist in target_state_temp1 for item in sublist]
        x_conca2 = [item for sublist in target_state_temp2 for item in sublist]
        target_state1 = np.asarray(x_conca1)
        target_state2 = np.asarray(x_conca2)
        
        # Initialize calculator and get vertex indices
        calculator = JawAngleCalculator()
        vertex_index_combined = (joint_position_first_wave.sampled_points_vertex_idx + 
                                joint_position.sampled_points_vertex_idx + 
                                joint_position_third_wave.sampled_points_vertex_idx)
        vertex_index = vertex_index_combined
        
        # Calculate initial distance
        method = 'euclidean'
        initial_distance = distance_calculator.calculate_distance(
            method, 
            calculator.control_vertices_slice(target_state1, vertex_index),
            calculator.control_vertices_slice(target_state2, vertex_index)
        )
        
        # Calculate estimated steps
        estimated_completion_steps = math.floor(916 * initial_distance)
        return estimated_completion_steps
        
    except Exception as e:
        print(f"Error calculating estimated steps: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    estimated_steps = calculate_estimated_steps("dataV22_phoneme_N", "dataV22_AU0_neutral")
    if estimated_steps is not None:
        print(f"Estimated completion steps: {estimated_steps}")
    else:
        print("Failed to calculate estimated steps")