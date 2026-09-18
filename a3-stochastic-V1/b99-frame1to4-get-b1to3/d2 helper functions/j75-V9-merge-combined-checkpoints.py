import os
import pickle
import numpy as np

def main():
    # --- CONFIGURATION ---
    checkpoint_files_to_merge = [
        r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250623-144458-copy\250623-144522\combined_checkpoints.pkl",  # Speech "not even you" (older format, length 903)
        r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_251110-124621\251110-124637\combined_checkpoints.pkl",      # Expression "sadness" (newer format, length 3603)
    ]

    # --- Define Output Path ---
    base_output_dir = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\merged_deformation_parameters_hub"
    
    folder_names = [os.path.basename(os.path.dirname(p)) for p in checkpoint_files_to_merge]
    subfolder_name = "merged_padded_" + "_".join(folder_names)
    output_dir = os.path.join(base_output_dir, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = "combined_checkpoints.pkl"
    output_path = os.path.join(output_dir, output_filename)

    # --- Pre-scan to find max vector length and load raw data ---
    print("--- Pre-scanning files to determine maximum vector length ---")
    max_vector_len = 0
    raw_data_list = []
    valid_files_to_process = []

    for file_path in checkpoint_files_to_merge:
        print(f"Scanning: {file_path}")
        if not os.path.exists(file_path):
            print(f"  ERROR: File not found. Skipping.")
            continue
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list) and data:
                max_vector_len = max(max_vector_len, len(data[0]))
                raw_data_list.append(data)
                valid_files_to_process.append(file_path)
                print(f"  -> Found {len(data)} checkpoints with vector length {len(data[0])}.")
            else:
                print(f"  ERROR: File is empty or has invalid format. Skipping.")
        except Exception as e:
            print(f"  ERROR: Failed to parse file: {e}. Skipping.")

    print(f"\nDetected maximum vector length of {max_vector_len} across all valid files.")

    # --- Load, Pad, and Validate Checkpoints ---
    all_checkpoints_data = []
    print("\n--- Processing and padding files ---")

    for i, raw_data in enumerate(raw_data_list):
        file_path = valid_files_to_process[i]
        print(f"Processing: {file_path}")

        # Convert to list of numpy arrays
        data_np = [np.array(vec, dtype=np.float64) for vec in raw_data]
        
        # Pad vectors if they are shorter than the max length
        current_vector_len = data_np[0].shape[0]
        if current_vector_len < max_vector_len:
            pad_width = max_vector_len - current_vector_len
            print(f"  -> Padding {len(data_np)} vectors from {current_vector_len} to {max_vector_len}...")
            padded_data = [np.pad(vec, (0, pad_width), 'constant', constant_values=0) for vec in data_np]
            all_checkpoints_data.append(padded_data)
            print(f"  -> OK. Padded. New vector length: {padded_data[0].shape[0]}, dtype: {padded_data[0].dtype}")
        else:
            all_checkpoints_data.append(data_np)
            print(f"  -> OK. No padding needed. Vector length: {current_vector_len}, dtype: {data_np[0].dtype}")

    if len(all_checkpoints_data) < 2:
        print("\nError: Need at least two valid checkpoint files to merge. Exiting.")
        return

    # Verify that all checkpoint lists have the same number of checkpoints
    first_len = len(all_checkpoints_data[0])
    if not all(len(x) == first_len for x in all_checkpoints_data):
        print("\nError: Not all checkpoint files have the same number of checkpoints after processing.")
        return

    print(f"\nAll files validated. Ready to merge {len(all_checkpoints_data)} sources, each with {first_len} checkpoints of length {max_vector_len}.")

    # --- Merge Checkpoints using NumPy ---
    sum_checkpoints = [v.copy() for v in all_checkpoints_data[0]]
    for i in range(1, len(all_checkpoints_data)):
        for j in range(first_len):
            sum_checkpoints[j] += all_checkpoints_data[i][j]

    print("\nMerging complete.")
    if sum_checkpoints:
        print(f"  - Type of first merged vector: {type(sum_checkpoints[0])}")
        print(f"  - Dtype of first merged vector: {sum_checkpoints[0].dtype}")
        print(f"  - Length of first merged vector: {sum_checkpoints[0].shape[0]}")


    # --- Save Combined Checkpoints ---
    with open(output_path, 'wb') as f:
        pickle.dump(sum_checkpoints, f)
    print(f"\nSuccessfully combined and saved to: {output_path}")

    # --- Verification ---
    try:
        with open(output_path, "rb") as f:
            loaded_checkpoints = pickle.load(f)
        print("\n--- Verification ---")
        print(f"Loaded {len(loaded_checkpoints)} combined checkpoints from file.")
        if loaded_checkpoints:
            print(f"  - Dimension of first checkpoint vector: {len(loaded_checkpoints[0])}")
            print(f"  - Type of first checkpoint vector: {type(loaded_checkpoints[0])}")
            first_is_zero = np.all(loaded_checkpoints[0] == 0)
            print(f"  - First checkpoint is zero vector: {first_is_zero}")
            last_checkpoint = loaded_checkpoints[-1]
            print(f"  - Final deformation stats: Max={np.max(last_checkpoint):.4f}, Min={np.min(last_checkpoint):.4f}, Mean={np.mean(last_checkpoint):.4f}")
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    main()
