import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
from matplotlib.colors import Normalize  # Import Normalize from matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation 
# from IPython import display
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection


sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\a1-blendshape-guided-animation-pipeline\database")
sys.path.insert(0, r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub")

import importlib

import dataV22_phoneme_N
importlib.reload(dataV22_phoneme_N)

import dataV22
importlib.reload(dataV22)

import dataV22_for_rest_adhoc
importlib.reload(dataV22_for_rest_adhoc)

import pickle

debug_mode_vsCode = 0 # maya platform online

# if debug_mode_vsCode == 0:
#     import maya.cmds as cmds
#     import maya.OpenMaya as OpenMaya      

def returnCoordinates(object_name, vertex_indices=None):
    # Select the object
    cmds.select(object_name, visible=True)
    
    # Get the active selection list
    selection = OpenMaya.MSelectionList()
    OpenMaya.MGlobal.getActiveSelectionList(selection)
    
    # Create an iterator for the selection list
    iterSel = OpenMaya.MItSelectionList(selection, OpenMaya.MFn.kMesh)
    
    # Get the DAG path and MFnMesh object
    dagPath = OpenMaya.MDagPath()
    iterSel.getDagPath(dagPath)
    currentInMeshMFnMesh = OpenMaya.MFnMesh(dagPath)
    
    # Get the mesh points in world space
    inMeshMPointArray = OpenMaya.MPointArray()
    currentInMeshMFnMesh.getPoints(inMeshMPointArray, OpenMaya.MSpace.kWorld)
    
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

visualize_organ = 1
if visualize_organ == 1:
    keyframe_num_temp = 2
    # x_bar_j = 
    # x_bar_j = x_bar_j
    # x_bar_conca = [item for sublist in getattr(dataV22_phoneme_N, "x_middleres_OrbicularisOris" + str(1)) for item in sublist]
    # x_bar_conca_rest_adhoc= [item for sublist in getattr(dataV22_for_rest_adhoc, "x_middleres_Jaw" + str(0)) for item in sublist]
    
    x_bar_conca_collected = returnCoordinates("middleres_OrbicularisOris")
    x_bar_conca = x_bar_conca_collected
    coords_rest = x_bar_conca

    previous_pose_folder_2 = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250502-204611\250502-204627"
    try:
        # Try to open and read data from the first file
        with open(previous_pose_folder_2 + "/deltaB_temp.pkl", 'rb') as f:
            data_t = pickle.load(f)
            deltaB = data_t['delta_b0_conca']
            x_joint_ = data_t['x_joint_']
            x_ooc_l_joint_ = data_t['x_ooc_l_joint_']
            x_ooc_r_joint_ = data_t['x_ooc_r_joint_']
            x_frontalis_joint_ = data_t['x_frontalis_joint_']

            x_joint_rotation_ = data_t['x_joint_rotation_']
            s_next = data_t['s_next']
            s_ooc_r_next = data_t['s_ooc_r_next']
            s_ooc_l_next = data_t['s_ooc_l_next']
            s_frontalis_next = data_t['s_frontalis_next']
            s_jaw_next = data_t['s_jaw_next']
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Trying to open and read data from the second file...")


    x_bar = s_jaw_next


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
        output_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\e68 backed to e71 no speed up and visualize debug"
        output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()        

    rest_vertices = coords_rest # np.array(dataV22.x_middleres_OrbicularisOris0)
    


    # call:
    # new_coords_for_energy_non_concatenate = [new_coords_for_energy[i:i+3] for i in range(0, len(new_coords_for_energy), 3)]
    coords_rest_non_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]
    # x_bar_conca_rest_adhoc = [x_bar_conca_rest_adhoc[i:i+3] for i in range(0, len(x_bar_conca_rest_adhoc), 3)]

    # rest_vertices = np.array(dataV22.x_middleres_OrbicularisOris0)
    visualize(coords_rest_non_concatenate, "T_mb_file_without_ziva")
    # visualize(coords_rest_non_concatenate, "neutral-in-g10-10")
    # visualize(coords_rest_non_concatenate, "jaw-after-deform-online")
    # visualize(new_coords_for_energy_non_concatenate, "current")

    # visualize(x_bar_conca_rest_adhoc, "neutral-in-rest_adhoc")
    # visualize(x_bar, "jaw-after-deform")