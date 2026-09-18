import numpy as np
import pickle

only_run_phase1_flag = 0  # Set this to 1 to only run phase 1

path1 = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250620-191133\250620-191151"

if only_run_phase1_flag == 0:
    path2 = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250620-191133\250620-191151"
    # path3 = same

# Load final_epoch_action_list_1 and optionally final_epoch_action_list_2 from pickle files
with open(path1 + r"\final_epoch_action_list.pkl", "rb") as file1:
    final_epoch_action_list_1 = pickle.load(file1)
    final_epoch_action_list = final_epoch_action_list_1['final_epoch_action_list']
    
    if only_run_phase1_flag == 0:
        with open(path2 + r"\final_epoch_action_list_phase2.pkl", "rb") as file2:
            final_epoch_action_list_2 = pickle.load(file2)
            final_epoch_action_list_phase2 = final_epoch_action_list_2['final_epoch_action_list']

        with open(path2 + r"\final_epoch_action_list_phase3.pkl", "rb") as file2:
            final_epoch_action_list_3 = pickle.load(file2)
            final_epoch_action_list_phase3 = final_epoch_action_list_3['final_epoch_action_list']            

# Initialize parameters
self_n = 300*3  # Replace with actual value of `self.n` (see config file.)
joint_rotate_num = 1  # Replace with actual value of `self.joint_rotate_num`
num_checkpoints_1 = 36  # 18   36

# Function to accumulate and save checkpoints
def accumulate_and_save_checkpoints(action_list, num_checkpoints, action_list2=None, action_list3=None):
    delta_b0_conca = [0] * self_n + [0] * 3 * joint_rotate_num
    total_length = len(action_list)
    
    if only_run_phase1_flag == 0 and action_list2 is not None:
        total_length2 = len(action_list2)
        total_length3 = len(action_list3)
        total_length_all = total_length + total_length2 + total_length3
        print("length action_list 1 and 2", total_length, total_length2, total_length3)
    else:
        total_length_all = total_length
    
    print("total length action_list", total_length_all)
    checkpoints = []

    # Compute evenly spaced indices
    # indices = np.linspace(0, total_length_all - 1, num=num_checkpoints, dtype=int)
    # indices = [0, 3343, 6687, 10031, 13375, 16719, 20062, 23406, 26750, 30094, 33438, 36782, 38122, 39463, 40803, 42144, 43484, 44825, 46165, 47506, 48846, 50187, 51527, 52868, 54208, 55549, 56890, 59893, 62896, 65899, 68902, 71906, 74909, 77912, 80915, 83919]
    indices = [0, 3678, 13486, 23295, 33104, 36782, 38981, 43381, 47781, 52181, 56582, 58781, 69756, 77738, 88713, 90651, 95820, 100989, 106159, 108097, 135373, 162649, 168034, 189577, 211120, 216505, 237327, 245008, 253787, 262566, 271345, 280124, 288903, 297682, 306462, 314143]
    # print("indices", indices)

    for i in range(total_length):
        delta_b0_conca = [delta_b0_conca[j] + action_list[i][j] for j in range(self_n + 3 * joint_rotate_num)]
      
        if i in indices:
            checkpoints.append(delta_b0_conca[:])  # Save a copy of the current state

    if only_run_phase1_flag == 0 and action_list2 is not None:
        i_outside = 0
        for i in range(total_length, total_length + total_length2): 
            delta_b0_conca = [delta_b0_conca[j] + action_list2[i_outside][j] for j in range(self_n + 3 * joint_rotate_num)]
     
            if i in indices:
                checkpoints.append(delta_b0_conca[:])  # Save a copy of the current state

            i_outside = i_outside + 1

    if only_run_phase1_flag == 0 and action_list3 is not None:
        i_outside = 0
        for i in range(total_length + total_length2, total_length_all): 
            delta_b0_conca = [delta_b0_conca[j] + action_list3[i_outside][j] for j in range(self_n + 3 * joint_rotate_num)]
     
            if i in indices:
                checkpoints.append(delta_b0_conca[:])  # Save a copy of the current state

            i_outside = i_outside + 1

    return checkpoints

# Accumulate checkpoints
if only_run_phase1_flag == 1:
    checkpoints_1 = accumulate_and_save_checkpoints(final_epoch_action_list, num_checkpoints_1)
else:
    checkpoints_1 = accumulate_and_save_checkpoints(final_epoch_action_list, num_checkpoints_1, final_epoch_action_list_phase2, final_epoch_action_list_phase3)

combined_checkpoints = checkpoints_1

# # Path to the pickle file
path3 = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250620-191133\250620-191151"

# Save the combined checkpoints to a pickle file
with open(path3 + "/combined_checkpoints.pkl", "wb") as file:
    pickle.dump(combined_checkpoints, file)

print("Checkpoints saved to 'combined_checkpoints.pkl'")

print("len of combined_checkpoints", len(combined_checkpoints))

try:
    # Load the combined checkpoints from the pickle file
    with open(path3 + "/combined_checkpoints.pkl", "rb") as file:
        combined_checkpoints = pickle.load(file)

    # Print the length of combined checkpoints
    print("Length of combined_checkpoints:", len(combined_checkpoints))

except FileNotFoundError:
    print("Error: File not found. Please check the path and try again.")
except EOFError:
    print("Error: File is empty or corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")