# have to run this code in maya
# open this maya file: 
# C:\keyLabLocal_D\c99-maya-ziva-workshop\experiment_files\g10-10 rest pose 1200 landmarks v9.mb


# commandPort -name "localhost:7001" -sourceType "mel";

# run locally offline: 
# "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" "C:\Users\zpa18\keyLabLocal\b99-frame1to4-get-b1to3\d2-adhoc-for-b99-part2.py"

# path2_combined_checkpoints =str(results_path / "merged_deformation_parameters_hub/merged_padded_250623-144522_251110-124637")
path2_combined_checkpoints = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\keyLabCloud\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003_25-7-18\job_250623-144458_experiment\job_250623-144458_40pct"
previous_pose_folder = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\SFU_PhD_cloud\2-Projects\1-physically based facial animation\3-Experiments\a3-stochastic-V1-backup-and-results\results\job_251128-140733\251128-140743"
neutral_to_other_pose_flag = 1


debug_mode_vsCode = 0

import sys
import os
from paths import a3_stochastic_relative_path, b98_sampling_path, a1_blendshape_path

sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo"))
sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo"/ "example_train"))
sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo"/ "env_gym"))
sys.path.insert(0, str(a1_blendshape_path / "database"))
sys.path.insert(0, str(b98_sampling_path ))
sys.path.insert(0, str(b98_sampling_path / "300joints_suite_good_old"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/ "joint_position_ooc_l"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/ "joint_position_ooc_r"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions"/ "joint_position_frontalis"))
sys.path.insert(0, r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub")


import numpy as np
import dataV22
import dataV22_AU22

import dataV22_deformed_phoneme_AA

# import dataV22_phoneme_T
# import dataV22_phoneme_IY
# import dataV22_phoneme_V
# import dataV22_phoneme_Y
# import dataV22_phoneme_UW

import importlib
from importlib import reload
importlib.reload(dataV22)
importlib.reload(dataV22_AU22)
importlib.reload(dataV22_deformed_phoneme_AA)

# importlib.reload(dataV22_phoneme_T)
# importlib.reload(dataV22_phoneme_IY)
# importlib.reload(dataV22_phoneme_V)
# importlib.reload(dataV22_phoneme_Y)
# importlib.reload(dataV22_phoneme_UW)

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

import argparse
import pickle
import math
import importlib

import sys
import os
import j77_V3_estimate_r_limit_newest

# target_rotation_deg, estimated_steps, t_limit, r_limit, ooc_l_limit, ooc_r_limit, frontalis_limit, this_task, x_bar, maya_file_to_open, estimated_completion_steps_ooc_l, estimated_completion_steps_ooc_r = j77_V3_estimate_r_limit_newest.calculate_rotation_parameters()
maya_file_to_open = r"dummy.mb"


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
    # try:
    #     # Redirect stdout and stderr to null
    #     # sys.stdout = open(os.devnull, 'w')
    #     # sys.stderr = open(os.devnull, 'w')        
    #     maya.standalone.initialize()
    #     print("Maya standalone initialized.")
    # except Exception as e:
    #     print(f"Maya standalone already initialized or failed to initialize: {e}")

    # if not os.path.exists(maya_file_to_open):
    #     raise RuntimeError(f"File not found: {maya_file_to_open}")

    # try:
    #     opened_file = cmds.file(maya_file_to_open, open=True, force=True)
        
    #     print("Opened file:", maya_file_to_open)
    # except Exception as e:
    #     print(f"Failed to open Maya file: {e}")
    #     if 'maya.standalone' in sys.modules:
    #          maya.standalone.uninitialize()
    #     return {}

    # # Function to delete existing cache nodes
    # def delete_cache_nodes():
    #     cache_nodes = cmds.ls(type='cacheFile')
    #     for node in cache_nodes:
    #         cmds.delete(node)

    # # Delete existing cache nodes
    # delete_cache_nodes()



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

    # cmds.select(cl=True)
    # cmds.currentCtx()
    # Close the reference    



    # try:
    #     maya.standalone.uninitialize()
    #     print("Maya standalone uninitialized.")
    # except Exception as e:
    #     print(f"Failed to uninitialize Maya standalone: {e}")

    return retrieved_positions

if debug_mode_vsCode == 0:
    # import maya.standalone
    # maya.standalone.initialize()    
    import maya.cmds as cmds
    import maya.OpenMaya as OpenMaya    
    import maya.mel as mel
    


from utils.distance_calculator import MeshDistanceCalculator
distance_calculator = MeshDistanceCalculator()

run_set_keyframes = 1
run_calculate_distance = 0

# mayaFilePath = r"C:\keyLabLocal_D\c99-maya-ziva-workshop\f97 use base and au12 and au22 joints binded and ziva muscle activation good.mb"
saveMayaTo =str(results_path / "240423-150653 log files")
save_folder = saveMayaTo
keyframe_num = 2 # 50  # 100
keyframe_translation_num = 2
# keframe_max = 50  # 1000


# vertex_index = [1137, 4468, 453, 1025, 1992, 3267, 352, 6568, 95, 5858, 4498, 9904, 6102, 1217, 99, 251, 3920, 5436, 6081, 4453, 9554, 3359, 6178, 4798, 5843, 5379, 7601, 3155, 7689, 10132, 8715, 431, 6067, 4519, 5942, 2601, 1605, 66, 6588, 2318, 7630, 8310, 3786, 9371, 7257, 2005, 7989, 4040, 8074, 6936, 5817, 7705, 5095, 6350, 3078, 3834, 7257, 7257, 8556, 3546, 9219, 6343, 1302, 2100, 7164, 5137, 329, 6696, 1218, 3247, 6974, 221, 9198, 5377, 3454, 7445, 2080, 6568, 9895, 2606, 8330, 7972, 971, 2195, 4221, 5525, 7780, 4363, 935, 7733, 5907, 2043, 7189, 6323, 9254, 9338, 1296, 3923, 5997, 9871, 5563, 1404, 7550, 6316, 1220, 5990, 5836, 8626, 3097, 7715, 9735, 2715, 7773, 3710, 2818, 6779, 7013, 7708, 9377, 3174, 6322, 6003, 5124, 7150, 6095, 4850, 8263, 9280, 6265, 4055, 5560, 3777, 505, 6547, 685, 6707, 85, 9207, 3901, 3901, 3274, 6990, 3773, 8738, 1360, 8084, 3640, 8603, 8953, 2760, 3423, 6648, 7257, 2120, 4766, 1309, 4724, 5682, 3161, 7701, 40, 3198, 9137, 1000, 7598, 65, 8506, 2559, 9859, 8451, 6679, 1244, 4116, 2868, 8294, 3507, 3124, 6333, 7257, 2876, 1956, 1412, 5547, 3851, 7160, 1534, 2285, 8861, 7988, 3359, 1007, 8923, 4035, 1744, 3105, 9795, 2718, 9404, 6801, 979, 7708, 5709, 4212, 7637, 7649, 6279, 3359, 8690, 986, 4023, 9367, 4493, 6273, 461, 7811, 8008, 359, 3299, 4667, 4452, 3945, 93, 8611, 3088, 780, 8213, 9172, 8203, 7550, 3945, 3898, 9371, 3518, 5426, 3089, 3348, 1439, 5601, 7967, 7302, 10220, 7361, 6311, 9921, 5434, 8412, 6790, 6639, 2470, 605, 9446, 2321, 5237, 5729, 5737, 4786, 7257, 9924, 4578, 9056, 8605, 1968, 7677, 6407, 7291, 3538, 10173, 9204, 453, 7001, 2422, 3945, 9826, 267, 7652, 5074, 6974, 7245, 7823, 7825, 9647, 2033, 1474, 589, 6470, 2968, 6971, 9152, 9434, 492, 8183, 3768, 4048, 970, 4237, 7006, 601, 2785, 6118, 7431, 4148, 2726, 9677, 9803, 5454, 5052, 5547, 7232, 6635, 3964, 2703, 2253, 3963, 7748, 1963, 10217, 902, 2, 1581, 7924, 7908, 4854, 8524, 9982, 1756, 10163, 6179, 703, 5038, 2105, 3124, 8065, 9253, 1832, 1354, 2981, 6920, 6531, 1285, 5009, 4869, 9464, 9843, 2275, 2995, 1571, 9698, 9881, 9932, 10210, 8011, 6090, 1633, 3286, 3945, 7540, 3899, 2811, 10210, 7039, 6033, 7669, 3055, 4801, 4610, 9626, 10149, 9667, 7257, 305, 1007, 5736, 9405, 118, 5127, 6074, 9486, 6958, 9291, 350, 5293, 2481, 2380, 6036, 10, 6771, 18, 956, 3331, 5813, 8906, 9323, 41, 8070, 9204, 9716, 8065, 3880, 2463, 9516, 393, 7108, 9295, 1425, 1265, 1543, 8036, 8580, 1542, 5210, 7639, 6879, 1890, 6178, 9898, 4278, 3665, 4443, 6748, 3166, 1377, 6080, 5412, 9826, 212, 8422, 1505, 7688, 6494, 2411, 6195, 8521, 3586, 533, 7770, 8793, 3567, 111, 5476, 451, 9187, 9309, 952, 1247, 8251, 5640, 793, 7651, 5378, 1356, 10147, 8083, 5326, 1920, 8661, 7860, 9664, 9442, 7897, 1375, 5090, 6380, 7257, 7682, 5003, 2067, 600, 3185, 5554, 6603, 1929, 529, 2442, 9274, 2341, 9689, 4396, 3359, 1967, 2197, 8330, 4185, 6540, 6020, 2627, 604, 3633, 636, 9772, 9545, 7590, 1842, 3289, 5981, 8309, 5508, 4099, 7925, 2256, 2098, 3574, 6290, 6005, 6182, 1922, 7958, 3293, 5678, 2879, 6049, 3359, 8087, 5986, 6077, 6406, 6190, 5918, 9210, 6287, 7647, 4384, 7673, 1461, 1464, 6257, 3751, 1738, 7257, 6064, 6937, 6312, 9419, 9060, 3669, 1185, 1705, 2703, 8761, 4249, 2771, 2071, 2314, 1381, 5799, 7781, 8344, 3557, 9712, 279, 6390, 5987, 5943, 2361, 6382, 5880, 6929, 7480, 3907, 8522, 7894, 4551, 8847, 9868, 2524, 10086, 5264, 4292, 262, 9779, 1002, 5369, 9206, 109, 9055, 3359, 7711, 6098, 2254, 379, 2143, 6780, 5396, 1383, 2099, 6633, 3813, 7095, 1462, 6386, 9284, 7244, 8399, 2265, 3340, 6099, 7504, 383, 3698, 783, 9299, 3746, 3945, 1434, 9819, 8384, 6239, 7778, 9793, 5921, 2709, 4525, 3494, 7959, 9241, 4276, 7283, 7141, 8038, 8919, 10189, 2333, 7143, 6579, 149, 2030, 8179, 5455, 5495, 9903, 10221, 9531, 8965, 7540, 7630, 3555, 4653, 9288, 6836, 36, 6978, 1258, 5258, 9668, 9399, 9778, 5582, 5207, 5336, 3133, 244, 6017, 1412, 5569, 6350, 10174, 9064, 2826, 9976, 3945, 8202, 4226, 5778, 4312, 9479, 512, 5488, 5461, 5330, 1038, 8790, 3033, 6356, 6975, 3945, 6052, 6373, 9510, 5522, 493, 3102, 6346, 7514, 5225, 5483, 3774, 9872, 4529, 3244, 3321, 5194, 7018, 10027, 5008, 1617, 9671, 4933, 1408, 10216, 5719, 5815, 8047, 8146, 8117, 8123, 8081, 9867, 1508, 8398, 8381, 8775, 5604, 3659, 5982, 7702, 3945, 2024, 274, 2651, 8305, 1160, 646, 3195, 2437, 815, 4232, 3359, 4539, 9359, 421, 6049, 7892, 1114, 9317, 9790, 5094, 7761, 8169, 8878, 8894, 6014, 7784, 9124, 771, 8087, 8925, 5327, 8950, 8573, 9437, 6590, 6476, 7907, 1923, 6030, 5534, 9086, 6255, 1911, 384, 6399, 2235, 8227, 9514, 8978, 3980, 9385, 8346, 6036, 9759, 8296, 355, 6553, 3062, 2442, 9896, 9813, 2122, 1386, 1353, 9450, 6552, 6224, 2162, 211, 4719, 6934, 5328, 7829, 8294, 9548, 9966, 10113, 5479, 2653, 4056, 7892, 4302, 4495, 3501, 7867, 1298, 8187, 1775, 3148, 6084, 7257, 7544, 7544, 7792, 7715, 3876, 9539, 5213, 7740, 10022, 8941, 4176, 6059, 6676, 9812, 8017, 2169, 4763, 584, 2048, 9461, 2418, 572, 7616, 8538, 78, 2430, 6116, 880, 6389, 6812, 337, 4826, 5654, 3111, 3945, 2460, 4550, 8309, 6330, 3761, 4507, 7574, 2834, 6566, 242, 7641, 599, 5965, 4829, 4509, 2706, 6796, 7693, 6916, 6762, 1298, 633, 9287, 1418, 9591, 9862, 6343, 1966, 8312, 5911, 2111, 2317, 3768, 5901, 281, 9227, 7758, 626, 8508, 6141, 1352, 6968, 4141, 2809, 276, 1097, 70, 6842, 2567, 7000, 5628, 3729, 6584, 3965, 263, 6059, 5726, 374, 5565, 2050, 383, 9288, 9193, 2048, 7793, 5302, 6324, 7600, 9869, 5572, 9889, 1278, 607, 9437, 2256, 3736, 5484, 8109, 8210, 1796, 7426, 7583, 7480, 1799, 5103, 8288, 9344, 5459, 5442, 7997, 3040, 7908, 7893, 9386, 178, 5794, 5185, 173, 8317, 3136, 2385, 1062, 8310, 4340, 7531, 10034, 3359, 7036, 6976, 3832, 5977, 6659, 5600, 3945, 9661, 8355, 292, 6477, 1924, 4493, 5910, 9311, 4862, 9363, 5040, 6287, 2180, 6034, 6720, 3174, 1386, 7022, 893, 9418, 3040, 8717, 1403, 5053, 5462, 7257, 5883, 9448, 1115, 1950, 7696, 8935, 9900, 8124, 3981, 6062, 6500, 9059, 2426, 1775, 6623, 3480, 3526, 9202, 3945]
vertex_index = joint_position_first_wave.sampled_points_vertex_idx + joint_position.sampled_points_vertex_idx + joint_position_third_wave.sampled_points_vertex_idx

vertex_jaw_index = [29, 177, 350]



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




action_mode = "both"
start_frame = 15




## new added start1
target_state_temp = getattr(dataV22, "x_middleres_OrbicularisOris" + str(keyframe_num - 1))        
# temporary = getattr(dataV22, "x_middleres_Jaw" + str(j_func))
x_conca = [item for sublist in target_state_temp for item in sublist]
x_conca = np.asarray(x_conca)
target_state = x_conca





def initial_joint_position():
    # This part is from the original d2 file, to get some base poses.
    x_joint_first_wave = getattr(joint_position_first_wave, "x_joint_position_1")
    x_joint_second_wave = getattr(joint_position, "x_joint_position_1")
    x_joint_third_wave = getattr(joint_position_third_wave, "x_joint_position_1")
    x_joint = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave

    x_joint_rotation = getattr(dataV22_deformed_phoneme_AA, "x_joint_rotation_cancatenate" + str(0))
    x_jaw_joint = x_joint_rotation

    x_joint_first_wave = getattr(joint_position_ooc_l_base_0, "x_joint_position_1")
    x_joint_second_wave = getattr(joint_position_ooc_l_base_pose11_0, "x_joint_position_1")
    x_joint_third_wave = getattr(joint_position_ooc_l_base_pose11_AU1_0, "x_joint_position_1")
    x_joint_fourth_wave = getattr(joint_position_ooc_l_base_pose11_AU1_AU2_0, "x_joint_position_1")
    x_ooc_l_joint = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave

    x_joint_first_wave = getattr(joint_position_ooc_r_base_0, "x_joint_position_1")
    x_joint_second_wave = getattr(joint_position_ooc_r_base_pose11_0, "x_joint_position_1")
    x_joint_third_wave = getattr(joint_position_ooc_r_base_pose11_AU1_0, "x_joint_position_1")
    x_joint_fourth_wave = getattr(joint_position_ooc_r_base_pose11_AU1_AU2_0, "x_joint_position_1")
    x_ooc_r_joint = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave

    x_joint_first_wave = getattr(joint_position_frontalis_base_0, "x_joint_position_1")
    x_joint_second_wave = getattr(joint_position_frontalis_base_AU1_0, "x_joint_position_1")
    x_joint_third_wave = getattr(joint_position_frontalis_base_AU1_AU2_0, "x_joint_position_1")
    x_frontalis_joint = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave

    # This is the new logic from d1
    
    if 1 == 1:
        # previous_pose_folder =str(results_path / "job_251028-122724/251028-122739")
        with open(previous_pose_folder + "/deltaB_temp.pkl", 'rb') as f:
            data_t = pickle.load(f)

        joint_groups_to_query = {
            'oor': 300,
            'ooc_l': 300,
            'ooc_r': 300,
            'frontalis': 300
        }
        retrieved_positions = retrieve_landmark_positions(maya_file_to_open, joint_groups_to_query)
        
        x_joint = retrieved_positions.get('oor', [])
        x_ooc_l_joint = retrieved_positions.get('ooc_l', [])
        x_ooc_r_joint = retrieved_positions.get('ooc_r', [])
        x_frontalis_joint = retrieved_positions.get('frontalis', [])
        
        if neutral_to_other_pose_flag == 0:
            x_joint_rotation = data_t['x_joint_rotation_']
        else:
            # x_joint_rotation = data_t['x_joint_rotation_neutral_to_other_pose_']
            pass

    x_joint_conca = [item for sublist in x_joint for item in sublist]
    x_joint_rotation_conca = [item for sublist in x_joint_rotation for item in sublist]
    x_ooc_l_joint_conca = [item for sublist in x_ooc_l_joint for item in sublist]
    x_ooc_r_joint_conca = [item for sublist in x_ooc_r_joint for item in sublist]
    x_frontalis_joint_conca = [item for sublist in x_frontalis_joint for item in sublist]

    return x_joint_conca, x_joint_rotation_conca, x_ooc_l_joint_conca, x_ooc_r_joint_conca, x_frontalis_joint_conca
# for frame_number in range(15,16):
for frame_number in range(15):
    cmds.currentTime(str(frame_number), edit=True)
    time_temp = str(frame_number)

x_joint_conca0, x_joint_rotation_conca0, x_ooc_l_joint_conca, x_ooc_r_joint_conca, x_frontalis_joint_conca = initial_joint_position()



import numpy as np

# Example initialization (ensure `x_joint_conca0` and `x_joint_rotation_conca0` are NumPy arrays)
x_joint = np.array(x_joint_conca0)
x_joint_rotation = np.array(x_joint_rotation_conca0)

n = 900
joint_rotate_num = 1


# initial 0 keyframe position.
frame_number = 0
cmds.currentTime(str(frame_number), edit=True)

time_temp = str(frame_number)

# x_joint = x_joint_conca
# x_joint_rotation_conca is given.

# del x_joint


n = 900 # len(x_joint_conca)
joint_num = int(n/3) #153
n_rotate = 3 # len(x_joint_rotation_conca)
joint_rotate_num = int(n_rotate/3) #1



joint_ooc_l_num = len(vertex_ooc_l_index)
joint_ooc_r_num = len(vertex_ooc_r_index)
joint_frontalis_num = len(vertex_frontalis_index)


joints = []

if debug_mode_vsCode == 0:
    for i_temp2 in range(0,1):
        joints.append('jaw0'+str(i_temp2+1))

for joint in joints:
    cmds.setKeyframe(joint, time=time_temp)

joints = []

if debug_mode_vsCode == 0:
    for i in range(joint_num):
        if i < 99:
            joints.append(f'oor_{i+1:02d}')
        elif i < 999:
            joints.append(f'oor_{i+1}')
        else:
            joints.append(f'oor_{i+1}')

for joint in joints:
    cmds.setKeyframe(joint, time=time_temp)







# Example initialization (ensure `x_joint_conca0` and `x_joint_rotation_conca0` are NumPy arrays)
x_joint_init = np.array(x_joint_conca0)
x_joint_rotation_init = np.array(x_joint_rotation_conca0)
x_ooc_l_joint_init = np.array(x_ooc_l_joint_conca)

x_ooc_r_joint_init = np.array(x_ooc_r_joint_conca)
x_frontalis_joint_init = np.array(x_frontalis_joint_conca)



def control_vertices_slice(target_state, vertex_index):
    combined_slices = []
    
    for index in vertex_index:
        slice_part = target_state[index*3:index*3+3]
        combined_slices.extend(slice_part)
    
    return combined_slices



# cmds.playbackOptions(min=0, max=50, animationStartTime=1, animationEndTime=50)
# # Function to delete existing cache nodes
# def delete_cache_nodes():
#     cache_nodes = cmds.ls(type='cacheFile')
#     for node in cache_nodes:
#         cmds.delete(node)

# # Delete existing cache nodes
# delete_cache_nodes()

# cmds.cacheFile(createCacheNode=True, singleCache=True, startTime=0, endTime=50, fileName='myCache', dir='cacheDir')



# Function to return coordinates at a specific frame
def get_coordinates_at_frame(frame, obj):
    # cmds.currentTime(frame)
    return returnCoordinates(obj)
    
all_coords = {}  # Dictionary to store coordinates at each frame
# Set up and run the logging function
start_frame = 0
end_frame = 32        
obj = "middleres_OrbicularisOris"
# Gather coordinates at each frame
for frame in range(start_frame, end_frame + 1):
    all_coords[frame] = get_coordinates_at_frame(frame, obj)


# move to the begining keyframe.
frame_number = 15 # keyframe_num
# cmds.currentTime(str(frame_number), edit=True)

s_next = all_coords[frame_number]  # returnCoordinates("middleres_OrbicularisOris")
method = 'euclidean'
distance_before = distance_calculator.calculate_distance(method, control_vertices_slice(target_state, vertex_index), control_vertices_slice(s_next, vertex_index))           

next_coords_rest = np.array(control_vertices_slice(s_next, vertex_index))

i = 0
s_next_interFunctional = getattr(dataV22, "x_middleres_OrbicularisOris" + str(i))
x_bar = s_next_interFunctional        
x_bar_conca = [item for sublist in x_bar for item in sublist]
x_bar_conca = np.asarray(x_bar_conca)
s_next = x_bar_conca

distance_before_from_file = distance_calculator.calculate_distance(method, control_vertices_slice(target_state, vertex_index), control_vertices_slice(s_next, vertex_index))           

current_coords_target = np.array(control_vertices_slice(target_state, vertex_index))


diff = next_coords_rest - current_coords_target

## new added end1

# for au12
deltaBList_temp_file_folder_first =str(results_path / "job_240904-093213/240904-093228")

# for au22
deltaBList_temp_file_folder_second =str(results_path / "job_240904-105216/240904-105217")

# for phoneme T
# deltaBList_temp_file_folder_3 =str(results_path / "job_241209-143126/241209-143141")

save_joint_path =str(results_path / "job_240904-093213/240904-093228")





# x_joint_conca, x_joint_rotation_conca = deform_rest_au12(deltaBList_temp_file_folder_first, start_frame, keframe_max, rest_x_joint_conca0, rest_x_joint_rotation_conca0, au12_x_joint_conca, au12_x_joint_rotation_conca)
#bk2



def calculate_normal(vertex1, vertex2, vertex3):
    # Create vectors from the vertices
    v1 = np.array(vertex2) - np.array(vertex1)
    v2 = np.array(vertex3) - np.array(vertex1)
    
    # Calculate the normal using the cross product
    normal = np.cross(v1, v2)
    # print("normal", normal)
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    return normal

def compute_direction_reward_normal(s_next, target_state):
    # Extract vertices from the state lists
    A1, B1, C1 = s_next[0:3], s_next[3:6], s_next[6:9]
    A2, B2, C2 = target_state[0:3], target_state[3:6], target_state[6:9]
    
    # Calculate the normal vectors
    real_direction = calculate_normal(A1, B1, C1)
    desired_direction = calculate_normal(A2, B2, C2)
    # print("real_direction:", real_direction)
    # print("desired_direction:", desired_direction)
    # Normalize the vectors (already normalized in calculate_normal)
    real_direction = real_direction / np.linalg.norm(real_direction)
    desired_direction = desired_direction / np.linalg.norm(desired_direction)
    
    # Compute the cosine of the angle between the two vectors
    cos_theta = np.dot(desired_direction, real_direction)
            
    return cos_theta    

def cos_theta_to_degrees(cos_theta):
    # Compute the arccosine of cos_theta to get the angle in radians
    angle_radians = np.arccos(cos_theta)
    
    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def control_vertices_slice(target_state, vertex_index):
    combined_slices = []
    
    for index in vertex_index:
        slice_part = target_state[index*3:index*3+3]
        combined_slices.extend(slice_part)
    
    return combined_slices


def measureDistanceInMayaFile(data_module, keframe_max):

    import importlib
    importlib.reload(data_module)
    
    # Get the target state from the dynamically passed module
    target_state_temp = getattr(data_module, "x_middleres_OrbicularisOris" + str(keyframe_num - 1))        
    x_conca = [item for sublist in target_state_temp for item in sublist]
    x_conca = np.asarray(x_conca)
    target_state = x_conca

    # Get jaw's target position
    x_jaw = getattr(data_module, "x_middleres_Jaw" + str(keyframe_num - 1))
    x_jaw_conca = [item for sublist in x_jaw for item in sublist]
    x_jaw_conca = np.asarray(x_jaw_conca)
    target_jaw_state = x_jaw_conca

    # Intermediate target (not used later)
    x_bar_conca = [item for sublist in getattr(data_module, "x_middleres_OrbicularisOris" + str(keyframe_num - 2)) for item in sublist]
    
    print("target is, ", "x_lips", str(keyframe_num - 1))
    
    # Access joint data
    x_joint = getattr(data_module, "x_joint_cancatenate" + str(keyframe_num - 2))
    # x_joint = self.x_joint_interFunctional

    x_joint_conca = [item for sublist in x_joint for item in sublist]

    del x_joint

    # x_joint_init_conca = [item for sublist in x_joint_init for item in sublist]
    x_joint_init_conca = x_joint_conca

    x_bar_conca = np.asarray(x_bar_conca)
    x_conca = np.asarray(x_conca)
    x_joint_conca = np.asarray(x_joint_conca)
    x_joint_init_conca = np.asarray(x_joint_init_conca)


    target_state = x_conca

    # move to the final keyframe.
    frame_number = keframe_max # keyframe_num
    cmds.currentTime(str(frame_number), edit=True)

    s_next = returnCoordinates("middleres_OrbicularisOris")
    s_jaw_next = returnCoordinates("middleres_Jaw")




    # distance = 0.0
    # distance, distance_ship = euclidean_distance(target_state[-3:], s_next[-3:])
    method = 'euclidean'
    distance = distance_calculator.calculate_distance(method, control_vertices_slice(target_state, vertex_index), control_vertices_slice(s_next, vertex_index))           






    cos_theta = compute_direction_reward_normal(control_vertices_slice(s_jaw_next, vertex_jaw_index), control_vertices_slice(target_jaw_state, vertex_jaw_index))
    angle_degrees = cos_theta_to_degrees(cos_theta)
    print("distance", distance)
    print("angle_degrees", angle_degrees)
    return distance, s_next, angle_degrees

    
    # # debug target state
    # with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files" + "/s_next_target_debugV1.pkl", 'wb') as f:
    #     pickle.dump({'target_state': target_state, 's_next': s_next}, f)


    # print("final distance is: ", distance)
    # return distance, s_next


# save as output file
def mayaSave2file():
    file_to_save = save_folder + "/cp26-a001 after-deform-task b2.mb"
    cmds.file(rename = file_to_save)
    cmds.file( save=True,f=True, type='mayaBinary' )


def calculate_distance(data_module, keframe_max):
    ## measure the distance. 

    # maya_file_name = "/cp26-a001 after-deform-task b2.mb"
    # maya_file_to_open = self.save_folder + maya_file_name
    file_to_save = save_folder + "/cp26-a001 after-deform-task b2.mb"
    maya_file_to_open = file_to_save
    # maya_file_to_open = r"C:\Users\zpa18\keyLabLocal\DSAC-T\results\DSAC_V2_m1_vertexMove_Va001\240419-182813\cp26-a001 after-deform-task b2.mb"
    # distance_in_MayaFile, s_next = measureDistanceInMayaFile(maya_file_to_open)
    # print("distance_in_MayaFile: ", distance_in_MayaFile)
    distance_in_MayaFile, s_next, angle_degrees = measureDistanceInMayaFile(data_module, keframe_max)
    print("distance_in_MayaFile: ", distance_in_MayaFile)
    print("angle_degrees_in_MayaFile: ", angle_degrees)


    # # debug target state
    # maya_file_to_open = r"D:\keyLabLocal_D\a2-DSAC-T-for-pipelineV1\results\DSAC_V2_m1_vertexMove_Va002\240426-043751\cp26-a001 after-deform-task b2.mb"
    # distance_in_MayaFile, s_next = measureDistanceInMayaFile(maya_file_to_open)        
    # print("distance_in_MayaFile IN a2: ", distance_in_MayaFile)

    # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
    #     print("\n \n distance_in_MayaFile: ", distance_in_MayaFile, "\n \n", file = f) 
    with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\log_v2.txt", 'a') as f:
        print("\n \n distance_in_MayaFile: ", distance_in_MayaFile, "\n \n", file = f)  


# Example usage of run_deformation_sequence
sequence = [1] * 36 + [0]
# sequence = [1,2,2,2,3,3,3,1,1,1,2,2,3,3,2,1,1,1,2,2,2]
# sequence = [2,3,4,5,6,2,7,8,0]  # by default, the begining pose is rest pose. so sequence[0] will be the first "next pose" to deform. and the last number is a dumpy end mark. I assign a 0 to it.
# pose_intensity_sequence = [0.6, 0.9, 1, 0.5, 0.8, 0]  # Corresponding intensity for each pose
pose_intensity_sequence = [1] * 36 + [0]  # Corresponding intensity for each pose

# muscle_activation_sequence = [0.8, 0.3, 0, 0.2, 0.9, 0]  # Corresponding intensity for each pose
muscle_activation_sequence = [0] * 36 + [0]  # Corresponding intensity for each pose

start_frame = 15
keframe_max = 50

x_muscle_activation_ = 0

# muscle
muscle_num = 2
muscles = []

if debug_mode_vsCode == 0:
    for i in range(muscle_num):
        if i < 90:
            muscles.append(f'zFiber{i+1}')
            
for muscle in muscles:
    # cmds.setKeyframe(muscle, time=time_temp)

    cmds.select(muscle, addFirst=True)
    cmds.setAttr(muscle + '.excitation', x_muscle_activation_)

    cmds.setKeyframe(muscle, time=start_frame)



keyframe_num = len(sequence)
keframe_max_sequence = np.round(np.linspace(start_frame,keframe_max,keyframe_num)).astype(int)

print("keframe_max_sequence:", keframe_max_sequence)





# DEBUG: Print the full path of the file being loaded to ensure it's the correct one.
loading_path = path2_combined_checkpoints + r"\combined_checkpoints.pkl"
print(f"--- DEBUG: Loading checkpoints from: {loading_path} ---")

with open(loading_path, "rb") as file:
    combined_checkpoints = pickle.load(file)

print("len of combined_checkpoints", len(combined_checkpoints))
print("combined_checkpoints", combined_checkpoints[0:1])




def beta_function(x_joint_conca, x_joint_rotation_conca, au22_x_joint_conca3, au22_x_joint_rotation_conca3, au22_x_ooc_l_joint_conca3, au22_x_ooc_r_joint_conca3, au22_x_frontalis_joint_conca3, start_frame, keyframe_max):
        for i in range(1, 2):  #takes 1 2 3


            current_start_frame = keyframe_max

            
            # print("keframe_max_sequence:", keframe_max_sequence)
            print("current_start_frame is:", keyframe_max)
            print("keyframe_max is:", keyframe_max)
            print("i", i)

        
            # frame_number = start_frame
            # cmds.currentTime(str(frame_number), edit=True)

            # time_temp = str(frame_number)

            # x_joint_init_conca = [item for sublist in x_joint_init for item in sublist]
            x_joint_init_conca = x_joint_conca

            x_joint_rotation_init_conca = x_joint_rotation_conca

            
            x_ooc_l_joint_init_conca = np.array(x_ooc_l_joint_init)

            x_ooc_r_joint_init_conca = np.array(x_ooc_r_joint_init)

            x_frontalis_joint_init_conca = np.array(x_frontalis_joint_init)
            

            # delta_b0_conca = deltaBList[i-1]
            x_joint_ = np.array(x_joint_init_conca)

            # au22_x_joint_conca3, au22_x_joint_rotation_conca3            
            
            if "both" == action_mode:                        
                x_joint_ = au22_x_joint_conca3 # x_joint_init + delta_b0_conca[0:n]
                x_joint_rotation_ = au22_x_joint_rotation_conca3 # x_joint_rotation_init + delta_b0_conca[n : n + 3*joint_rotate_num]
                x_ooc_l_joint_ = au22_x_ooc_l_joint_conca3
                x_ooc_r_joint_ = au22_x_ooc_r_joint_conca3
                x_frontalis_joint_ = au22_x_frontalis_joint_conca3

            # x_joint_ = np.array(x_joint_init_conca) + np.array(delta_b0_conca)
            
            # x_joint_rotation_ = np.array(x_joint_rotation_init_conca) + delta_b0_conca[n : n + 3*joint_rotate_num]
            # x_joint_rotation_ = np.array(x_joint_rotation_init_conca)
                    
            # with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\log_v2.txt", 'a') as f:
            #     print(" \n shape of x_joint_", len(x_joint_),  "\n x_joint_ =  \n", x_joint_, file = f)  
            #     print(" \n shape of x_joint_init_conca", len(x_joint_init_conca)," \n x_joint_init_conca =  \n", x_joint_init_conca, file = f)  
            #     print(" \n x_joint_ - np.array(x_joint_init_conca)", np.array(x_joint_) - np.array(x_joint_init_conca), "\n", file = f)  
                
                # print(" \n  \n delta_b0_conca =  \n", delta_b0_conca, file = f)  



            # trainer = create_trainer(alg, sampler, buffer, evaluator, **args)
            # self.env.moveJointToFinal(x_joint_)
            # i_temp = 0
            # a2_for_move_joint.a2_for_move_joint(i_temp, i_temp+1, x_joint_, s_next, **args)

            print("******************", keframe_max)
            frame_number = keframe_max
            cmds.currentTime(str(frame_number), edit=True)
            time_temp = str(frame_number)
            # for joint in joints:
            #     cmds.setKeyframe(joint, time=time_temp)


            # Ensure both variables are numpy arrays before performing the subtraction
            x_joint_rotation_ = np.array(x_joint_rotation_)
            x_joint_rotation_init_conca = np.array(x_joint_rotation_init_conca)

            # Now you can subtract the two arrays
            relative_rotation_value = x_joint_rotation_ - x_joint_rotation_init_conca

            # relative_rotation_value = relative_rotation_value * 0.6

            # Deformation calculations with intensity
            relative_rotation_value = relative_rotation_value  #  * intensity            

            # relative_rotation_value = delta_b0_conca[n : n + 3*joint_rotate_num]
            # relative_rotation_value = x_joint_rotation_ - x_joint_rotation_init_conca

            # if debug_mode_vsCode == 1:
            # # if debug_mode_vsCode == 0:        
            #     for i_temp2 in range(0,joint_rotate_num):
            #         cmds.rotate( x_joint_rotation_[3*i_temp2], x_joint_rotation_[3*i_temp2+1], x_joint_rotation_[3*i_temp2+2], 'ref:joint'+str(i_temp2+2), absolute=True )

            # x_joint_init, x_joint_rotation_init
            if debug_mode_vsCode == 0:
                for i_temp2 in range(0,joint_rotate_num):
                    flag_temp1 = np.array_equal(x_joint_rotation_init_conca[3*i_temp2],x_joint_rotation_[3*i_temp2])
                    flag_temp2 = np.array_equal(x_joint_rotation_init_conca[3*i_temp2+1],x_joint_rotation_[3*i_temp2+1])
                    flag_temp3 = np.array_equal(x_joint_rotation_init_conca[3*i_temp2+2],x_joint_rotation_[3*i_temp2+2])

                    if (not flag_temp1) or (not flag_temp2) or (not flag_temp3):                  
                        cmds.rotate( relative_rotation_value[3*i_temp2], relative_rotation_value[3*i_temp2+1], relative_rotation_value[3*i_temp2+2], 'jaw0'+str(i_temp2+1), relative=True, ws = True)
                    
                    cmds.setKeyframe('jaw0'+str(i_temp2+1), time=time_temp)


            # if debug_mode_vsCode == 1:
            # # if debug_mode_vsCode == 0:        
            #     for i_temp2 in range(0,joint_num):
            #         cmds.move( x_joint_[3*i_temp2], x_joint_[3*i_temp2+1], x_joint_[3*i_temp2+2], 'ref:joint'+str(i_temp2+2), absolute=True )


            # Ensure both variables are numpy arrays before performing the subtraction
            x_joint_ = np.array(x_joint_)
            x_joint_init_conca = np.array(x_joint_init_conca)

            # Now you can subtract the two arrays
            relative_rotation_value = x_joint_ - x_joint_init_conca

            # relative_rotation_value = relative_rotation_value * 0.6
            # Deformation calculations with intensity
            relative_rotation_value = relative_rotation_value #  * intensity                        

            joint_num = 300

            # translate oor

            if debug_mode_vsCode == 0:
                for i_temp2 in range(0,joint_num):
                    flag_temp1 = np.array_equal(x_joint_init_conca[3*i_temp2],x_joint_[3*i_temp2])
                    flag_temp2 = np.array_equal(x_joint_init_conca[3*i_temp2+1],x_joint_[3*i_temp2+1])
                    flag_temp3 = np.array_equal(x_joint_init_conca[3*i_temp2+2],x_joint_[3*i_temp2+2])
                    
                    if i_temp2 + 1 <= 9:
                        joint_name = f'oor_0{i_temp2 + 1}'  # Leading zero for single-digit numbers
                    elif i_temp2 + 1 <= 99:
                        joint_name = f'oor_{i_temp2 + 1}'
                    elif i_temp2 + 1 <= 999:
                        joint_name = f'oor_{i_temp2 + 1}'
                    else:
                        joint_name = f'oor_{i_temp2 + 1}'

                    if (not flag_temp1) or (not flag_temp2) or (not flag_temp3):                  
                        cmds.move( relative_rotation_value[3*i_temp2], relative_rotation_value[3*i_temp2+1], relative_rotation_value[3*i_temp2+2], joint_name, relative=True)
                    cmds.setKeyframe(joint_name, time=time_temp)
                    


            # translate ooc l
            if debug_mode_vsCode == 0:
                for i_temp2 in range(joint_ooc_l_num):
                    flag_temp1 = np.array_equal(x_ooc_l_joint_init_conca[3 * i_temp2], x_ooc_l_joint_[3 * i_temp2])
                    flag_temp2 = np.array_equal(x_ooc_l_joint_init_conca[3 * i_temp2 + 1], x_ooc_l_joint_[3 * i_temp2 + 1])
                    flag_temp3 = np.array_equal(x_ooc_l_joint_init_conca[3 * i_temp2 + 2], x_ooc_l_joint_[3 * i_temp2 + 2])
                    if not flag_temp1 or not flag_temp2 or not flag_temp3:
                        # Determine the joint name based on the index i_temp2
                        if i_temp2 + 1 <= 9:
                            joint_name = f'ooc_l_0{i_temp2 + 1}'  # Leading zero for single-digit numbers
                        elif i_temp2 + 1 <= 99:
                            joint_name = f'ooc_l_{i_temp2 + 1}'
                        elif i_temp2 + 1 <= 999:
                            joint_name = f'ooc_l_{i_temp2 + 1}'
                        else:
                            joint_name = f'ooc_l_{i_temp2 + 1}'

                        # Move the joint
                        if 1==0: # joint_name == "lip_lower191":
                            pass                    

                        else: 
                            # print("joint_name to deform this time", joint_name)
                            cmds.move(x_ooc_l_joint_[3 * i_temp2], x_ooc_l_joint_[3 * i_temp2 + 1], x_ooc_l_joint_[3 * i_temp2 + 2], joint_name, absolute=True)

                    cmds.setKeyframe(joint_name, time=time_temp)



            # translate ooc r
            if debug_mode_vsCode == 0 or debug_mode_vsCode == 1:
                for i_temp2 in range(joint_ooc_r_num):
                    flag_temp1 = np.array_equal(x_ooc_r_joint_init_conca[3 * i_temp2], x_ooc_r_joint_[3 * i_temp2])
                    flag_temp2 = np.array_equal(x_ooc_r_joint_init_conca[3 * i_temp2 + 1], x_ooc_r_joint_[3 * i_temp2 + 1])
                    flag_temp3 = np.array_equal(x_ooc_r_joint_init_conca[3 * i_temp2 + 2], x_ooc_r_joint_[3 * i_temp2 + 2])
                    if not flag_temp1 or not flag_temp2 or not flag_temp3:
                        # Determine the joint name based on the index i_temp2
                        if i_temp2 + 1 <= 9:
                            joint_name = f'ooc_r_0{i_temp2 + 1}'  # Leading zero for single-digit numbers
                        elif i_temp2 + 1 <= 99:
                            joint_name = f'ooc_r_{i_temp2 + 1}'
                        elif i_temp2 + 1 <= 999:
                            joint_name = f'ooc_r_{i_temp2 + 1}'
                        else:
                            joint_name = f'ooc_r_{i_temp2 + 1}'

                        # Move the joint
                        if 1==0: # joint_name == "lip_lower191":
                            pass                    

                        else: 
                            # print("joint_name to deform this time", joint_name)
                            cmds.move(x_ooc_r_joint_[3 * i_temp2], x_ooc_r_joint_[3 * i_temp2 + 1], x_ooc_r_joint_[3 * i_temp2 + 2], joint_name, absolute=True)

                    cmds.setKeyframe(joint_name, time=time_temp)


            # translate frontalis
            if debug_mode_vsCode == 0 or debug_mode_vsCode == 1:
                for i_temp2 in range(joint_frontalis_num):
                    flag_temp1 = np.array_equal(x_frontalis_joint_init_conca[3 * i_temp2], x_frontalis_joint_[3 * i_temp2])
                    flag_temp2 = np.array_equal(x_frontalis_joint_init_conca[3 * i_temp2 + 1], x_frontalis_joint_[3 * i_temp2 + 1])
                    flag_temp3 = np.array_equal(x_frontalis_joint_init_conca[3 * i_temp2 + 2], x_frontalis_joint_[3 * i_temp2 + 2])
                    if not flag_temp1 or not flag_temp2 or not flag_temp3:
                        # Determine the joint name based on the index i_temp2
                        if i_temp2 + 1 <= 9:
                            joint_name = f'frontalis_0{i_temp2 + 1}'  # Leading zero for single-digit numbers
                        elif i_temp2 + 1 <= 99:
                            joint_name = f'frontalis_{i_temp2 + 1}'
                        elif i_temp2 + 1 <= 999:
                            joint_name = f'frontalis_{i_temp2 + 1}'
                        else:
                            joint_name = f'frontalis_{i_temp2 + 1}'

                        # Move the joint
                        if 1==0: # joint_name == "lip_lower191":
                            pass                    

                        else: 
                            # print("joint_name to deform this time", joint_name)
                            cmds.move(x_frontalis_joint_[3 * i_temp2], x_frontalis_joint_[3 * i_temp2 + 1], x_frontalis_joint_[3 * i_temp2 + 2], joint_name, absolute=True)

                    cmds.setKeyframe(joint_name, time=time_temp)



            # x_muscle_activation_ = muscle_activation

            # for muscle in muscles:
                

            #     cmds.select(muscle, addFirst=True)
            #     cmds.setAttr(muscle + '.excitation', x_muscle_activation_)

            #     cmds.setKeyframe(muscle, time=time_temp)

            # cmds.select('zFiber1', addFirst=True)
            # cmds.setAttr('zFiber1.excitation', x_muscle_activation_)

            x_joint_conca = x_joint_
            x_joint_rotation_conca = x_joint_rotation_



        mayaSave2file()



# for i in range(10):
# for i in range(36):
for i in range(len(combined_checkpoints)):    
    print("i", i)
    delta_b0_conca = np.array(combined_checkpoints[i])  # Ensure this is a NumPy array
    # print("len of delta_b0_conca", len(delta_b0_conca))

    # # Add detailed debugging before the problematic section
    # print(f"\n=== Debugging Array Shapes ===")
    # print(f"Length of delta_b0_conca: {len(delta_b0_conca)}")
    # print(f"n (3*joint_num): {n}")
    # print(f"3 * joint_rotate_num: {3 * joint_rotate_num}")
    # print(f"3 * joint_ooc_l_num: {3 * joint_ooc_l_num}")
    # print(f"3 * joint_ooc_r_num: {3 * joint_ooc_r_num}")
    # print(f"3 * joint_frontalis_num: {3 * joint_frontalis_num}")



    # # Calculate all the slice indices
    # slice1_end = n
    # slice2_start = n
    # slice2_end = n + 3 * joint_rotate_num
    # slice3_start = 3*joint_num + 3*joint_rotate_num
    # slice3_end = 3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num
    # slice4_start = 3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num
    # slice4_end = 3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num + 3 * joint_ooc_r_num
    # slice5_start = 3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num + 3 * joint_ooc_r_num
    # slice5_end = 3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num + 3 * joint_ooc_r_num + 3 * joint_frontalis_num

    # print(f"\nSlice indices:")
    # print(f"slice1: 0 to {slice1_end} (length: {slice1_end - 0})")
    # print(f"slice2: {slice2_start} to {slice2_end} (length: {slice2_end - slice2_start})")
    # print(f"slice3: {slice3_start} to {slice3_end} (length: {slice3_end - slice3_start})")
    # print(f"slice4: {slice4_start} to {slice4_end} (length: {slice4_end - slice4_start})")
    # print(f"slice5: {slice5_start} to {slice5_end} (length: {slice5_end - slice5_start})")

    # # Check each slice individually
    # print(f"\nActual slice lengths:")
    # print(f"delta_b0_conca[0:{slice1_end}]: {len(delta_b0_conca[0:slice1_end])}")
    # print(f"delta_b0_conca[{slice2_start}:{slice2_end}]: {len(delta_b0_conca[slice2_start:slice2_end])}")
    # print(f"delta_b0_conca[{slice3_start}:{slice3_end}]: {len(delta_b0_conca[slice3_start:slice3_end])}")
    # print(f"delta_b0_conca[{slice4_start}:{slice4_end}]: {len(delta_b0_conca[slice4_start:slice4_end])}")
    # print(f"delta_b0_conca[{slice5_start}:{slice5_end}]: {len(delta_b0_conca[slice5_start:slice5_end])}")

    # # Check initial array shapes
    # print(f"\nInitial array shapes:")
    # print(f"x_joint_init shape: {x_joint_init.shape if hasattr(x_joint_init, 'shape') else len(x_joint_init)}")
    # print(f"x_joint_rotation_init shape: {x_joint_rotation_init.shape if hasattr(x_joint_rotation_init, 'shape') else len(x_joint_rotation_init)}")
    # print(f"x_ooc_l_joint_init shape: {x_ooc_l_joint_init.shape if hasattr(x_ooc_l_joint_init, 'shape') else len(x_ooc_l_joint_init)}")
    # print(f"x_ooc_r_joint_init shape: {x_ooc_r_joint_init.shape if hasattr(x_ooc_r_joint_init, 'shape') else len(x_ooc_r_joint_init)}")
    # print(f"x_frontalis_joint_init shape: {x_frontalis_joint_init.shape if hasattr(x_frontalis_joint_init, 'shape') else len(x_frontalis_joint_init)}")


    # --- DEBUGGING BLOCK ---
    print("--- DEBUGGING BEFORE LINE 1027 ---")
    try:
        # Print values of variables used in slicing
        print("\n[INFO] Slicing variables:")
        if 'joint_num' in locals(): print(f"  - joint_num = {joint_num}")
        if 'n' in locals(): print(f"  - n = {n}")
        if 'joint_rotate_num' in locals(): print(f"  - joint_rotate_num = {joint_rotate_num}")
        if 'joint_ooc_l_num' in locals(): print(f"  - joint_ooc_l_num = {joint_ooc_l_num}")

        # Print info about the main array
        print("\n[INFO] Main array to be sliced:")
        print(f"  - Type of delta_b0_conca: {type(delta_b0_conca)}")
        print(f"  - Length of delta_b0_conca: {len(delta_b0_conca)}")

        # Analyze the specific slice for x_ooc_l_joint_
        slice_start = 3*joint_num + 3*joint_rotate_num
        slice_end = 3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num
        the_slice = delta_b0_conca[slice_start:slice_end]
        
        print("\n[INFO] Analysis of the problematic slice:")
        print(f"  - Slice start index: {slice_start}")
        print(f"  - Slice end index: {slice_end}")
        print(f"  - Length of the resulting slice: {len(the_slice)}")

        # Analyze the operands for the failing line
        print("\n[INFO] Operands for the addition `x_ooc_l_joint_init + slice`:")
        if hasattr(x_ooc_l_joint_init, 'shape'):
            print(f"  - Shape of x_ooc_l_joint_init: {x_ooc_l_joint_init.shape}")
        else:
            print(f"  - Length of x_ooc_l_joint_init: {len(x_ooc_l_joint_init)}")
        print(f"  - Length of the slice: {len(the_slice)}")

    except NameError as e:
        print(f"\n[ERROR] A variable needed for debugging is not defined: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during debugging: {e}")

    # print("\n--- EXITING SCRIPT FOR DEBUGGING ---")
    # import sys
    # sys.exit()
    # --- END DEBUGGING BLOCK ---

    # Element-wise addition    
    x_joint_ = x_joint_init + delta_b0_conca[0:n]
    x_joint_rotation_ = x_joint_rotation_init + delta_b0_conca[n:n + 3 * joint_rotate_num]
    x_ooc_l_joint_ = x_ooc_l_joint_init + delta_b0_conca[3*joint_num + 3*joint_rotate_num:3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num]
    x_ooc_r_joint_ = x_ooc_r_joint_init + delta_b0_conca[3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num : 3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num + 3 * joint_ooc_r_num]
    x_frontalis_joint_ = x_frontalis_joint_init + delta_b0_conca[3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num + 3 * joint_ooc_r_num : 3*joint_num + 3*joint_rotate_num + 3 * joint_ooc_l_num + 3 * joint_ooc_r_num + 3 * joint_frontalis_num]

    # Debugging: Print the variables to observe their values
    # print("x_joint_init:", x_joint_init)
    # print("delta_b0_conca[0:n]:", delta_b0_conca[0:n])
    # print("x_joint_:", x_joint_)


    print("len of x_joint_", len(x_joint_))
    print("len of x_joint_rotation_", len(x_joint_rotation_))


    
    start_frame = i + 14
    keframe_max = i + 15
    beta_function(x_joint, x_joint_rotation, x_joint_, x_joint_rotation_, x_ooc_l_joint_, x_ooc_r_joint_, x_frontalis_joint_, start_frame, keframe_max)

    x_joint = x_joint_
    x_joint_rotation = x_joint_rotation_
    # x_joint = np.array(x_joint_conca0)
    # x_joint_rotation = np.array(x_joint_rotation_conca0)    
    x_ooc_l_joint = x_ooc_l_joint_
    x_ooc_r_joint = x_ooc_r_joint_
    x_frontalis_joint = x_frontalis_joint_

    # print("L1031 passed the test, now quit.")

    # sys.exit(0)  # Exit with status code 0  

    if keframe_max == 32:

        # x_joint_init = x_joint
        # x_joint_rotation_init = x_joint_rotation
        pass



mayaSave2file()

print("job done")