# have to run this code in maya
# open this maya file: 
# "C:\keyLabLocal_D\c99-maya-ziva-workshop\experiment_files\g10-5 rest pose 1200 landmarks v5.mb"
# commandPort -name "localhost:7001" -sourceType "mel";

# run locally offline: 
# "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" "C:\Users\zpa18\keyLabLocal\b99-frame1to4-get-b1to3\d2-adhoc-for-b99-part2.py"


debug_mode_vsCode = 0

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
sys.path.insert(0, str(backup_path / "b31 dataV22 backup hub"))


import numpy as np
import dataV22
# import dataV22_AU22
import importlib
from importlib import reload
importlib.reload(dataV22)
# importlib.reload(dataV22_AU22)


import dataV22_phoneme_T
importlib.reload(dataV22_phoneme_T)

import joint_position
import joint_position_first_wave
import joint_position_third_wave
importlib.reload(joint_position)
importlib.reload(joint_position_first_wave)
importlib.reload(joint_position_third_wave)

import dataV22_phoneme_UW

importlib.reload(dataV22_phoneme_UW)


import argparse
import pickle
import math
import importlib



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

action_mode = "both"
start_frame = 15




## new added start1
target_state_temp = getattr(dataV22_phoneme_UW, "x_middleres_OrbicularisOris" + str(keyframe_num - 1))        
# temporary = getattr(dataV22_phoneme_UW, "x_middleres_Jaw" + str(j_func))
x_conca = [item for sublist in target_state_temp for item in sublist]
x_conca = np.asarray(x_conca)
target_state = x_conca


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
s_next_interFunctional = getattr(dataV22_phoneme_UW, "x_middleres_OrbicularisOris" + str(i))
x_bar = s_next_interFunctional        
x_bar_conca = [item for sublist in x_bar for item in sublist]
x_bar_conca = np.asarray(x_bar_conca)
s_next = x_bar_conca

distance_before_from_file = distance_calculator.calculate_distance(method, control_vertices_slice(target_state, vertex_index), control_vertices_slice(s_next, vertex_index))           

current_coords_target = np.array(control_vertices_slice(target_state, vertex_index))


diff = next_coords_rest - current_coords_target

## new added end1

# for phoneme N
deltaBList_temp_file_folder_first =str(results_path / "job_241209-091306/241209-091322")

# for phoneme AA
deltaBList_temp_file_folder_second =str(results_path / "job_241209-130715/241209-130731")

save_joint_path =str(results_path / "job_240904-093213/240904-093228")



def load_joint_data(save_joint_path):

    # Load the data from the pickle file
    with open(save_joint_path +'\joint_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    # Accessing the variables after loading
    rest_x_joint_conca0 = loaded_data['rest']['x_joint_conca0']
    rest_x_joint_rotation_conca0 = loaded_data['rest']['x_joint_rotation_conca0']

    au12_x_joint_conca = loaded_data['au12']['x_joint_conca']
    au12_x_joint_rotation_conca = loaded_data['au12']['x_joint_rotation_conca']

    au22_x_joint_conca3 = loaded_data['au22']['x_joint_conca3']
    au22_x_joint_rotation_conca3 = loaded_data['au22']['x_joint_rotation_conca3']

    print("Data successfully loaded and ready for use.")    

    return rest_x_joint_conca0, rest_x_joint_rotation_conca0, au12_x_joint_conca, au12_x_joint_rotation_conca, au22_x_joint_conca3, au22_x_joint_rotation_conca3

rest_x_joint_conca0, rest_x_joint_rotation_conca0, au12_x_joint_conca, au12_x_joint_rotation_conca, au22_x_joint_conca3, au22_x_joint_rotation_conca3 = load_joint_data(save_joint_path)




start_frame = 15
keframe_max = 32  # 1000

start_frame_2 = 32
keframe_max_2 = 50  # 1000


# def euclidean_distance(point1, point2):

#     distance = np.linalg.norm(np.asarray(point1) - np.asarray(point2))

#     vertices_num = int(len(point1)/3) #172
#     distance_ship = [0] * vertices_num * 3
#     for i in range(0,vertices_num * 3):
#         distance_ship[i] = math.sqrt(  (point1[i] - point2[i])**2  )  # + (point1[3*i+1] - point2[3*i+1])**2 + (point1[3*i+2] - point2[3*i+2])**2 

#     return distance, distance_ship


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
    
    keyframe_num = 2

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
    # with open(str(results_path / "\240423-150653 log files" + "/s_next_target_debugV1.pkl", 'wb') as f:
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
    with open(str(results_path / "240423-150653 log files " / "log_v2.txt"), 'a') as f:
        print("\n \n distance_in_MayaFile: ", distance_in_MayaFile, "\n \n", file = f)  


# def deform_au12_au22(start_frame, keframe_max, x_joint_conca, x_joint_rotation_conca, au22_x_joint_conca3, au22_x_joint_rotation_conca3, data_module):


def calculate_distance_adhoc(data_module, keframe_max):
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
    # with open(str(results_path / "\240423-150653 log files\log_v2.txt", 'a') as f:
    #     print("\n \n distance_in_MayaFile: ", distance_in_MayaFile, "\n \n", file = f)  

    return distance_in_MayaFile

distance_log1 = []
time_log1 = []
distance_log2 = []
time_log2 = []
