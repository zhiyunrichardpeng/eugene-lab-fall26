import numpy as np
import pickle

path1 = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250528-200118\250528-200133"
# path2 = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_240904-105216\240904-105217"

# Load final_epoch_action_list_1 and final_epoch_action_list_2 from pickle files
with open(path1 + "\deltaB_temp_final_epoch_action_list.pkl", "rb") as file1:  # , open(path2 + "\deltaB_temp_final_epoch_action_list.pkl", "rb") as file2
    final_epoch_action_list_1 = pickle.load(file1)
    # final_epoch_action_list_2 = pickle.load(file2)

# Initialize parameters
self_n = 300*3  # Replace with actual value of `self.n` (see config file. )
joint_rotate_num = 1  # Replace with actual value of `self.joint_rotate_num`
num_checkpoints_1 = 36  # 18   36
# num_checkpoints_2 = 19

# Function to accumulate and save checkpoints
def accumulate_and_save_checkpoints(action_list, num_checkpoints):
    delta_b0_conca = [0] * self_n + [0] * 3 * joint_rotate_num
    total_length = len(action_list)
    print("total length action_list", total_length)
    checkpoints = []

    # Compute evenly spaced indices
    indices = np.linspace(0, total_length - 1, num=num_checkpoints, dtype=int)

    for i in range(total_length):
        # for j in range(self_n + 3 * joint_rotate_num):
        # print("total length action_list element at i row", i, len(action_list[i]))
        # print("action_list[i]", action_list[i])
        # for j in range(self_n + 3 * joint_rotate_num):
            # print("i,j", i, j)
            # print("action_list[i][j]", action_list[i][j])
        delta_b0_conca = [delta_b0_conca[j] + action_list[i][j] for j in range(self_n + 3 * joint_rotate_num)]
        if i in indices:
            checkpoints.append(delta_b0_conca[:])  # Save a copy of the current state

    return checkpoints

# Accumulate checkpoints for both lists
checkpoints_1 = accumulate_and_save_checkpoints(final_epoch_action_list_1, num_checkpoints_1)
# checkpoints_2 = accumulate_and_save_checkpoints(final_epoch_action_list_2, num_checkpoints_2)

# Combine the checkpoints, removing the repeated middle element
# combined_checkpoints = checkpoints_1  + checkpoints_2[1:]  # # Exclude the first element of `checkpoints_2` to avoid repetition
combined_checkpoints = checkpoints_1
# Save the combined checkpoints to a pickle file
with open(path1 + "/combined_checkpoints.pkl", "wb") as file:
    pickle.dump(combined_checkpoints, file)

print("Checkpoints saved to 'combined_checkpoints.pkl'")


print("len of combined_checkpoints", len(combined_checkpoints))
# print("combined_checkpoints", combined_checkpoints[0:1])


import pickle

# Path to the pickle file
path1 = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250528-200118\250528-200133"

try:
    # Load the combined checkpoints from the pickle file
    with open(path1 + "/combined_checkpoints.pkl", "rb") as file:
        combined_checkpoints = pickle.load(file)

    # Print the length of combined checkpoints
    print("Length of combined_checkpoints:", len(combined_checkpoints))

    # Print the first 5 elements
    # print("First 5 elements of combined_checkpoints:")
    # for i, element in enumerate(combined_checkpoints[:2]):
    #     print(f"Element {i + 1}: {element}")

except FileNotFoundError:
    print("Error: File not found. Please check the path and try again.")
except EOFError:
    print("Error: File is empty or corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
