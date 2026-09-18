import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_landmarks(landmarks, title, filename):
    """
    Creates a 3D scatter plot of the given landmarks and saves it to a file.
    landmarks: A list of [name, x, y, z]
    """
    if not landmarks:
        print(f"No landmarks to plot for {title}")
        return

    # Extract coordinates
    x_coords = [lm[1] for lm in landmarks]
    y_coords = [lm[2] for lm in landmarks]
    z_coords = [lm[3] for lm in landmarks]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='Blues', marker='o', s=20)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title(title)
    
    ax.view_init(elev=(180 + 30 - 90), azim=0, roll=90)
    
    max_range = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords)) / 2.0
    mid_x = (max(x_coords) + min(x_coords)) / 2.0
    mid_y = (max(y_coords) + min(y_coords)) / 2.0
    mid_z = (max(z_coords) + min(z_coords)) / 2.0
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {filename}")

def visualize_and_save_landmarks(landmark_data, group_name, save_folder):
    """
    Saves landmark data to a CSV file and creates a 3D plot.
    
    Args:
        landmark_data (list or np.array): A list of 3D coordinates, e.g., [[x1,y1,z1], [x2,y2,z2], ...].
        group_name (str): The name of the landmark group (e.g., 'ooc_r').
        save_folder (str): The folder path to save the CSV and plot.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    num_landmarks = len(landmark_data)
    if num_landmarks == 0:
        print(f"No landmark data provided for group '{group_name}'.")
        return

    # Setup CSV for landmark locations
    landmark_locations_file = os.path.join(save_folder, f"a99_debug_landmark_locations_{group_name}.csv")
    
    with open(landmark_locations_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['landmark_name', 'x', 'y', 'z'])

        print(f"\n--- Saving Landmark Locations for {group_name} ---")

        landmark_locations_for_plot = []
        for i in range(num_landmarks):
            # Determine the joint name based on the index i
            if i + 1 <= 9:
                joint_name = f'{group_name}_0{i + 1}'
            else:
                joint_name = f'{group_name}_{i + 1}'
            
            location = landmark_data[i]
            
            # Save to CSV
            csv_writer.writerow([joint_name, location[0], location[1], location[2]])
            landmark_locations_for_plot.append([joint_name, location[0], location[1], location[2]])
    
    print(f"Saved {num_landmarks} landmarks for group '{group_name}' to {landmark_locations_file}.")
    
    # Plotting
    plot_filename = os.path.join(save_folder, f"plot_{group_name}_landmarks.png")
    plot_landmarks(landmark_locations_for_plot, f"Landmark Locations: {group_name}", plot_filename)
