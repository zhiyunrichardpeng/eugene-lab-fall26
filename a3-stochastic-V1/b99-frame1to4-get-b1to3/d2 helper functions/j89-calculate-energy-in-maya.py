import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
from scipy.spatial import ConvexHull, Delaunay
# import trimesh

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# run this line first: 

i_frame = 14
self_time_for_energy_log = []
self_strain_energy = []


import sys
from paths import *

sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo"))
sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo" / "example_train"))
sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo" / "env_gym"))
sys.path.insert(0, str(a1_blendshape_path / "database"))
sys.path.insert(0, str(b98_sampling_path))
sys.path.insert(0, str(b98_sampling_path / "300joints_suite_good_old"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_ooc_l"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_ooc_r"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_frontalis"))

import numpy as np

import dataV22_for_rest_adhoc
import importlib
from importlib import reload
importlib.reload(dataV22_for_rest_adhoc)

debug_mode_vsCode = 0

if debug_mode_vsCode == 0:
    # import maya.standalone
    # maya.standalone.initialize()    
    import maya.cmds as cmds
    import maya.OpenMaya as OpenMaya    
    import maya.mel as mel


def returnCoordinates(object):
    # Select the object
    cmds.select(object, visible=True)
    
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
    
    # Pre-allocate NumPy array for points
    num_points = inMeshMPointArray.length()
    pointArray = np.zeros((num_points, 3))
    
    # Copy points from MPointArray to NumPy array
    for i in range(num_points):
        point = inMeshMPointArray[i]
        pointArray[i, 0] = point.x
        pointArray[i, 1] = point.y
        pointArray[i, 2] = point.z
    
    # Flatten the NumPy array to 1D
    one_d_list = pointArray.flatten()
    
    return one_d_list


def strain_energy_calculator(new_coords, coords_rest):

    import numpy as np
    import sys
    sys.path.insert(0, str(a1_blendshape_path / "database"))
    import importlib

    # import dataV22_for_rest_adhoc_after_deform
    # import dataV22_for_rest_adhoc
    # importlib.reload(dataV22_for_rest_adhoc_after_deform)
    # importlib.reload(dataV22_for_rest_adhoc)


    # Material properties (Lamé parameters)
    mu = 1.0      # Shear modulus
    lambda_ = 1.0  # Lamé's first parameter

    # Rest pose: mixed mesh (triangles and quadrangles) in 3D
    vertices_rest = np.array(coords_rest)

    # Deformed pose for hybrid method
    vertices_deformed_hybrid = np.array(new_coords)

    # Deformed pose for pure kinematic method
    # vertices_deformed_kinematic = np.array(dataV22_for_rest_adhoc.x_middleres_OrbicularisOris1)

    # Elements: mix of triangles and quadrangles (vertex indices)
    # elements = dataV22_for_rest_adhoc.x_middleres_OrbicularisOris_face_0
    elements = dataV22_for_rest_adhoc.x_middleres_OrbicularisOris_face_0
    # elements = elements

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
        

        # debug the "F = np.column_stack((E1_deformed, E2_def" dimonsion error
        # log_path =str(results_path / "240423-150653 log files/log_debug25-2-20.txt")


        # new_coords_for_energy = new_coords
        # with open(log_path, 'w') as f:
        #     # Print new_coords_for_energy
        #     print("new_coords_for_energy shape:", np.shape(new_coords_for_energy), file=f)
        #     print("new_coords_for_energy first 5 elements:\n", new_coords_for_energy[:5], file=f)
            
        #     # Print coords_rest
        #     print("\ncoords_rest shape:", np.shape(coords_rest), file=f)
        #     print("coords_rest first 5 elements:\n", coords_rest[:5], file=f)

        #     # Print elements
        #     print("\nelements shape:", np.shape(elements), file=f)
        #     print("elements first 5 elements:\n", elements[:5], file=f)

        #     # Compute deformation vectors
        #     # E1_deformed = new_coords_for_energy[elements[:, 1]] - new_coords_for_energy[elements[:, 0]]
        #     # E2_deformed = new_coords_for_energy[elements[:, 2]] - new_coords_for_energy[elements[:, 0]]
            
        #     # E1_rest = coords_rest[elements[:, 1]] - coords_rest[elements[:, 0]]
        #     # E2_rest = coords_rest[elements[:, 2]] - coords_rest[elements[:, 0]]

        #     # Print E1_deformed and E2_deformed
        #     print("\nE1_deformed shape:", np.shape(E1_deformed), file=f)
        #     print("E1_deformed first 1 elements:\n", E1_deformed, file=f)
            
        #     print("\nE2_deformed shape:", np.shape(E2_deformed), file=f)
        #     print("E2_deformed first 1 elements:\n", E2_deformed, file=f)

        #     # Print E1_rest and E2_rest
        #     print("\nE1_rest shape:", np.shape(E1_rest), file=f)
        #     print("E1_rest first 1 elements:\n", E1_rest, file=f)

        #     print("\nE2_rest shape:", np.shape(E2_rest), file=f)
        #     print("E2_rest first 1 elements:\n", E2_rest, file=f)

        #     # Check if any of these arrays are empty (which might cause the np.cross error)
        #     if E1_deformed.size == 0 or E2_deformed.size == 0:
        #         print("\nERROR: One of the deformation vectors (E1_deformed, E2_deformed) is empty!", file=f)

        #     if E1_rest.size == 0 or E2_rest.size == 0:
        #         print("\nERROR: One of the rest vectors (E1_rest, E2_rest) is empty!", file=f)

        #     # Try computing the transformation matrix F and catch errors
        #     try:
        #         F = np.column_stack((E1_deformed, E2_deformed, np.cross(E1_deformed, E2_deformed))) @ \
        #             np.linalg.inv(np.column_stack((E1_rest, E2_rest, np.cross(E1_rest, E2_rest))))
        #         print("\nF matrix computed successfully!", file=f)
        #     except Exception as e:
        #         print("\nERROR while computing F:", str(e), file=f)



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



# if debug_mode_vsCode == 0:
#     # import maya.standalone
#     # maya.standalone.initialize()    
#     import maya.cmds as cmds
#     import maya.OpenMaya as OpenMaya    
#     import maya.mel as mel

def step_forward_one_frame():
    current_frame = cmds.currentTime(query=True)
    cmds.currentTime(current_frame + 1)

current_frame = 14
cmds.currentTime(current_frame)


# Loop from i = 14 to i = 49
for i_frame in range(14, 50):  # range(14, 50) includes 14 to 49
    i_frame = i_frame+1
    start_frame = i_frame + 0

    # Step forward one frame in Maya
    step_forward_one_frame()
    
    self_time_for_energy_log.append(i_frame)


    keyframe_num = 2

    x_bar_conca = [item for sublist in getattr(dataV22_for_rest_adhoc, "x_middleres_OrbicularisOris" + str(keyframe_num - 2)) for item in sublist]
    x_bar_j = x_bar_conca
    new_coords_for_energy = returnCoordinates("middleres_OrbicularisOris")
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
        output_folder = str(backup_path / "e46 adjustable overlap")
        output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()        

    rest_vertices = np.array(dataV22_for_rest_adhoc.x_middleres_OrbicularisOris0)
    # visualize(rest_vertices, "rest")


    # call:
    new_coords_for_energy_non_concatenate = [new_coords_for_energy[i:i+3] for i in range(0, len(new_coords_for_energy), 3)]
    coords_rest_non_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]

    # rest_vertices = np.array(dataV22_for_rest_adhoc.x_middleres_OrbicularisOris0)
    # visualize(new_coords_for_energy_non_concatenate, "current")


    strain_enegy = strain_energy_calculator(new_coords_for_energy_non_concatenate, coords_rest_non_concatenate)



    self_strain_energy.append(strain_enegy)

    print("self_strain_energy", self_strain_energy)
    print("No. ", i_frame, "Frame energy done")

# Final output (optional)
print("self_strain_energy log:", self_strain_energy)
print("Time log:", self_time_for_energy_log)

