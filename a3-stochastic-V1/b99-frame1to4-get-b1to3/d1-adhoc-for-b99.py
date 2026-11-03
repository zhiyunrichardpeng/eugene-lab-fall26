import datetime
import os
import sys
from paths import (
    a3_stochastic_path,
    b98_sampling_path,
    a1_blendshape_path,
    backup_path,
    results_path
)

sys.path.insert(0, str(a3_stochastic_path / "a3-stochastic-V1-algo"/ "example_train"))
sys.path.insert(0, str(backup_path / "database"))
sys.path.insert(0, str(b98_sampling_path))
sys.path.insert(0, str(b98_sampling_path / "300joints_suite_good_old"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/"joint_position_ooc_l"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/"joint_position_ooc_r"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/"joint_position_frontalis"))
sys.path.insert(0, str(backup_path / "b31 dataV22 backup hub"))



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


import dataV22
importlib.reload(dataV22)
import dataV22_for_rest_adhoc
importlib.reload(dataV22_for_rest_adhoc)

def main():
    # Your code here
    loadPathRotation = str(results_path / "job_240612-203822 rotation 0.05" / "240612-203830")

    if 1==run_a2_fromFrameGetDeltaB:
        if 1 == run_both:
            deltaBList = []
            
            save_folder = r"C:\Users\zpa18\keyLabLocalplaceHolder"

            # action_mode = "translation"    
            # action_mode = "rotation"    
            action_mode = "both"    
            for i in range(0, 1):  # keyframe_num - 1
            
                log_path = None

                
                x_bar = getattr(dataV22_for_rest_adhoc, "x_middleres_OrbicularisOris" + str(i))

                previous_pose_folder = str(results_path / "job_250428-184837\250428-184853")
                previous_pose_folder_2 = str(results_path / "job_250429-210906\250429-210922")

                AA_to_rest_flag = 0  # 0
                if AA_to_rest_flag == 1:

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


                    x_bar = s_next

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
                    output_folder = str(backup_path / "d83 initiate rest to AU12 smile"
                    output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()     

                coords_rest_non_concatenate = x_bar
                # coords_rest_non_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]
                # visualize(coords_rest_non_concatenate, "rest")

                x_jaw_bar = getattr(dataV22_for_rest_adhoc, "x_middleres_Jaw" + str(i))  # not used in this experiment
                x_ooc_l_bar = getattr(dataV22, "x_middleres_OrbicularisOculi_l" + str(i))
                x_ooc_r_bar = getattr(dataV22, "x_middleres_OrbicularisOculi_r" + str(i))
                x_frontalis_bar = getattr(dataV22, "x_middleres_Frontalis" + str(i))

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
                x_joint = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave

                
                # x_jaw_joint = getattr(dataV22, "x_jaw_joint_concatenate" + str(i))           
                x_jaw_joint = getattr(dataV22_for_rest_adhoc, "x_joint_cancatenate" + str(i))  # temporaty use a dummy one
                # useless

                x_jaw_joint_rotation = getattr(dataV22_for_rest_adhoc, "x_joint_rotation_cancatenate" + str(i)) # x_joint_rotation_cancatenate0 = [[0, 0, 0]]

                # to update the joint position to the AA pose, as the starting pose.
                def update_joint_position_to_AA_pose():
                    deltaBList_temp_file_folder = previous_pose_folder
                    with open(deltaBList_temp_file_folder + "/deltaBList_temp.pkl", 'rb') as f:
                        data_t = pickle.load(f)
                        deltaBList = data_t['deltaBList']

                    x_joint_conca = [item for sublist in x_joint for item in sublist]
                    x_joint_rotation_conca = [item for sublist in x_jaw_joint_rotation for item in sublist]

                    x_joint_init_conca = x_joint_conca
                    x_joint_init = np.array(x_joint_init_conca)
                    x_joint_rotation_init_conca = x_joint_rotation_conca
                    x_joint_rotation_init = np.array(x_joint_rotation_init_conca)                    
                    delta_b0_conca = deltaBList[i-1]
                    x_joint_ = np.array(x_joint_init_conca)


                    n = len(x_joint_conca)
                    joint_num = int(n/3) #153
                    n_rotate = len(x_joint_rotation_conca)
                    joint_rotate_num = int(n_rotate/3) #1
                    x_joint_ = x_joint_init + delta_b0_conca[0:n]
                    x_joint_rotation_ = x_joint_rotation_init + delta_b0_conca[n : n + 3*joint_rotate_num]

                    multiple_previous_folder = 1
                    if multiple_previous_folder == 1:
                        deltaBList_temp_file_folder = previous_pose_folder_2
                        with open(deltaBList_temp_file_folder + "/deltaBList_temp.pkl", 'rb') as f:
                            data_t = pickle.load(f)
                            deltaBList = data_t['deltaBList']
            
                        delta_b0_conca = deltaBList[i-1]

                        x_joint_ = x_joint_ + delta_b0_conca[0:n]
                        x_joint_rotation_ = x_joint_rotation_ + delta_b0_conca[n : n + 3*joint_rotate_num]                            

                    x_joint_non_concatenate = [x_joint_[i:i+3] for i in range(0, len(x_joint_), 3)]
                    x_joint_rotation_non_concatenate = [x_joint_rotation_[i:i+3] for i in range(0, len(x_joint_rotation_), 3)]


                    return x_joint_non_concatenate, x_joint_rotation_non_concatenate
                
                AA_to_rest_flag = 0  # 0
                if AA_to_rest_flag == 1:
                    x_joint, x_jaw_joint_rotation = update_joint_position_to_AA_pose()
                
                # # Example usage: print the combined length of the positions
                
                x_joint_first_wave = getattr(joint_position_ooc_l_base_0, "x_joint_position_1")
                x_joint_second_wave = getattr(joint_position_ooc_l_base_pose11_0, "x_joint_position_1")
                x_joint_third_wave = getattr(joint_position_ooc_l_base_pose11_AU1_0, "x_joint_position_1")
                x_joint_fourth_wave = getattr(joint_position_ooc_l_base_pose11_AU1_AU2_0, "x_joint_position_1")
                x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave

                # vertex_index_combined = joint_position_ooc_l_base_0.sampled_points_vertex_idx + joint_position_ooc_l_base_pose11_0.sampled_points_vertex_idx  + joint_position_ooc_l_base_pose11_AU1_0.sampled_points_vertex_idx  + joint_position_ooc_l_base_pose11_AU1_AU2_0.sampled_points_vertex_idx

                x_ooc_l_joint = x_joint_combined
                print(f"Total length of combined x_ooc_l_joint: {len(x_joint_combined)}")
                
                x_joint_first_wave = getattr(joint_position_ooc_r_base_0, "x_joint_position_1")
                x_joint_second_wave = getattr(joint_position_ooc_r_base_pose11_0, "x_joint_position_1")
                x_joint_third_wave = getattr(joint_position_ooc_r_base_pose11_AU1_0, "x_joint_position_1")
                x_joint_fourth_wave = getattr(joint_position_ooc_r_base_pose11_AU1_AU2_0, "x_joint_position_1")
                x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave                    

                # folder_path = str(b98_sampling_path / "joint_positions\joint_position_ooc_r"
                # sys.path.insert(0, folder_path)
                # # Call the function and get the combined x_joint_position_1
                # x_joint_combined = combine_x_joint_positions_from_folder(folder_path)
                
                # Example usage: print the combined length of the positions
                print(f"Total length of combined x_ooc_r_joint: {len(x_joint_combined)}")

                

                x_ooc_r_joint = x_joint_combined

                
                x_joint_first_wave = getattr(joint_position_frontalis_base_0, "x_joint_position_1")
                x_joint_second_wave = getattr(joint_position_frontalis_base_AU1_0, "x_joint_position_1")
                x_joint_third_wave = getattr(joint_position_frontalis_base_AU1_AU2_0, "x_joint_position_1")
                
                x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave

                # folder_path = str(b98_sampling_path / "joint_positions\joint_position_ooc_r"
                # sys.path.insert(0, folder_path)
                # # Call the function and get the combined x_joint_position_1
                # x_joint_combined = combine_x_joint_positions_from_folder(folder_path)
                
                # Example usage: print the combined length of the positions
                print(f"Total length of combined x_frontalis_joint: {len(x_joint_combined)}")

                

                x_frontalis_joint = x_joint_combined


                s_next = x_bar

                s_jaw_next = x_jaw_bar

                s_ooc_l_next = x_ooc_l_bar

                s_ooc_r_next = x_ooc_r_bar
                s_frontalis_next = x_frontalis_bar                    

                x_joint_ = x_joint
                x_jaw_joint_ = x_jaw_joint # useless
                x_ooc_l_joint_ = x_ooc_l_joint
                x_ooc_r_joint_ = x_ooc_r_joint
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

            # with open(str(results_path / \240423-150653 log files\log.txt", 'w') as f:
            #     print("deltaBList: \n \n", deltaBList, file = f)  

            # with open(str(results_path / \240423-150653 log files" + "/deltaBList_temp.pkl", 'wb') as f:
            #     pickle.dump({'deltaBList': deltaBList}, f)


# Run the profiler
cProfile.run('main()', 'output.prof')

# Create a Stats object
p = pstats.Stats('output.prof')

import datetime

saveFolder = str(results_path / "240423-150653 log files" / "computingTime_analysis")
filename = saveFolder + r"\computingTime_analysis_file"+str("_")+ datetime.datetime.now().strftime("%y%m%d-%H%M%S") #  +".mv"
os.makedirs(saveFolder, exist_ok=True)

# Open a text file for writing
with open(filename + ".txt", 'w') as f:
    # Redirect the output to the text file
    p.stream = f
    # Sort and print statistics to the text file
    p.sort_stats('cumulative').print_stats(60)

print("Profiling results saved to output.txt")
