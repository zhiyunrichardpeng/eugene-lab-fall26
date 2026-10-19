import datetime
import os
import sys
from paths import b98_sampling_path, a1_blendshape_path, backup_path, results_path
sys.path.insert(0, str(b98_sampling_path))
sys.path.insert(0, str(a1_blendshape_path / "database"))

sys.path.insert(0, str(backup_path / "b31 dataV22 backup hub"))



# print("debug plot l7")
# print("loadPathRotation", loadPathRotation)

run_c96_finalized_datav21generating = 1
run_a2_fromFrameGetDeltaB = 0
run_both = 0
run_rotation = 0
run_translation = 0

# keyframe_num = 20 # 50 # 100 # 10 # 4  # 10

use_incremental_NN_flag = 0  # 1  # 0

import argparse
import pickle
import numpy as np

import cProfile
import pstats





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
parser.add_argument("--unit_denominator", type=float, default=100000.0, help="unit_denominator")
parser.add_argument("--sample_batch_size", type=int, default=20, help="unit_denominator")

args = vars(parser.parse_args())
# import a2_fromFrameGetDeltaB as a2_fromFrameGetDeltaB


keyframe_num = args["keyframe_num"]
keyframe_translation_num = args["keyframe_translation_num"]

print("debug plot l55")

if 1 == run_c96_finalized_datav21generating:
    import c96_finalized_datav21generating_for_b98 as c96_finalized_datav21generating_for_b98
    c96_finalized_datav21generating_for_b98.c96_finalized_datav21generating(keyframe_num)

# from PyQt5.QtCore import Qt, QCoreApplication
# from PyQt5.QtWidgets import QApplication, QLabel        

# # Set the attribute before creating the QApplication
# QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

# app = QApplication(sys.argv)

print("debug plot l69")

import dataV22_AU7 as dataV22

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("debug plot l76")

def plot_target_state(target_state, object_name):
    vertices_num = int(len(target_state) / 3)
    
    # Extracting X, Y, Z coordinates
    x_coords = [target_state[3 * i] for i in range(vertices_num)]
    y_coords = [target_state[3 * i + 1] for i in range(vertices_num)]
    z_coords = [target_state[3 * i + 2] for i in range(vertices_num)]
    print("debug plot l82")

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o', s=1)  # Set s=1 for smaller dots
    ax.view_init(elev=(180 + 30), azim=0, roll=90)  # Change these values to set the desired view
    
    print("debug plot l90")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Target State Visualization')

    # Save the plot to a file
    output_folder =str(results_path / "240423-150653 log files")
    output_file = os.path.join(output_folder, f'3d_mesh_plot_{object_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print("debug plot l103")


# middleres_OrbicularisOculi_r
target_state_temp = getattr(dataV22, "x_middleres_OrbicularisOculi_l" + str(1))        # x_middleres_OrbicularisOculi_l   x_middleres_OrbicularisOris
# temporary = getattr(dataV22_rest, "x_middleres_OrbicularisOris" + str(j_func))
x_conca = [item for sublist in target_state_temp for item in sublist]
object_name = "phoneme_AU2L_ooc_1200landmark_v1"

plot_target_state(x_conca, object_name)
print("job done")