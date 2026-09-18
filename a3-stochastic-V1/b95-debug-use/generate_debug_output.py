
import os
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D

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

def generate_output(x_ooc_r_bar, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    
    landmark_locations_file = os.path.join(save_folder, "a99_debug_landmark_locations_from_d1.csv")
    
    with open(landmark_locations_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['landmark_name', 'x', 'y', 'z'])

        print("\n--- Retrieving and Saving Landmark Locations from d1-adhoc-for-b99.py ---")

        group_name = 'ooc_r'
        
        # x_ooc_r_bar is a list of lists/tuples, each being a 3D coordinate.
        # Let's reshape if it's a flat list. Based on d1-adhoc-for-b99.py, it should be a list of lists.
        if x_ooc_r_bar and isinstance(x_ooc_r_bar[0], (int, float)):
             x_ooc_r_bar = [x_ooc_r_bar[i:i+3] for i in range(0, len(x_ooc_r_bar), 3)]

        num_landmarks = len(x_ooc_r_bar)
        
        print(f"\nProcessing group: {group_name}")
        landmark_locations_for_plot = []
        for i in range(num_landmarks):
            if i + 1 <= 9:
                joint_name = f'{group_name}_0{i + 1}'
            else:
                joint_name = f'{group_name}_{i + 1}'

            location = x_ooc_r_bar[i]
            
            if i < 3:
                print(f"  Location of {joint_name} is: {location}")
            
            csv_writer.writerow([joint_name, location[0], location[1], location[2]])
            landmark_locations_for_plot.append([joint_name, location[0], location[1], location[2]])
        
        print(f"  Found and saved {num_landmarks} landmarks for group '{group_name}'.")
        plot_filename = os.path.join(save_folder, f"plot_{group_name}_landmarks_from_d1.png")
        plot_landmarks(landmark_locations_for_plot, f"Landmark Locations: {group_name}", plot_filename)

    print(f"\nSaved all found landmark locations to: {landmark_locations_file}")
