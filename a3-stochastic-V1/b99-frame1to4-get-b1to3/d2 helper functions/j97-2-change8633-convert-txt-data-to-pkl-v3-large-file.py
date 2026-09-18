import numpy as np
import pickle
import re

# file_path = r"C:\path\to\log0_final_list.txt"
# file_path_pick = r"C:\path\to\deltaB_temp_final_epoch_action_list.pkl"

file_path = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250528-200118\250528-200133\log0_final_list.txt"
file_path_pick = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250528-200118\250528-200133\deltaB_temp_final_epoch_action_list.pkl"


def parse_large_array_file(file_path):
    """Parse large array text files efficiently"""
    array_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip to the line containing the array data
        for line in f:
            if line.startswith('final_epoch_action_list:'):
                # Extract array content
                array_content = line.split(':', 1)[1].strip()
                break
        
        # Process remaining content if multi-line
        array_content += f.read()
    
    # Clean and parse
    array_content = array_content.replace('array(', '').replace(')', '')
    array_content = re.sub(r'[\[\],\n]', ' ', array_content)  # Remove brackets and commas
    
    # Convert to numpy array (memory efficient)
    try:
        data = np.fromstring(array_content, sep=' ', dtype=np.float32)
        data = np.fromstring(array_content, sep=' ', dtype=np.float32)
        # Reshape if you know the expected dimensions, e.g.:
        data = data.reshape((756, 3603))  # Adjust shape as needed   10355
        return data
    except Exception as e:
        print(f"Conversion error: {e}")
        return None

# Parse the file
final_epoch_action_list = parse_large_array_file(file_path)

if final_epoch_action_list is not None:
    # Save as pickle
    with open(file_path_pick, 'wb') as f:
        pickle.dump(final_epoch_action_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Successfully saved {len(final_epoch_action_list)} elements to {file_path_pick}")
    print(f"Memory usage: {final_epoch_action_list.nbytes/1e6:.2f} MB")
else:
    print("Failed to parse the file")