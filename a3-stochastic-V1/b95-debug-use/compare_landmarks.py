import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def compare_landmark_files():
    # Define file paths
    file1_path =str(results_path / "240423-150653 log files/AU2_right_as_rest_pose/mb_file_offline/a99_debug_landmark_locations.csv")
    # file2_path =str(results_path / "240423-150653 log files/joint_file_ooc_r/a99_debug_landmark_locations.csv")
    file2_path =str(results_path / "240423-150653 log files/AU2_right_as_rest_pose/mb_file_offline_from_d1/a99_debug_landmark_locations_from_d1.csv")
    
    output_folder =str(results_path / "240423-150653 log files/comparison/mb_file_offlineAU2R_VS_joint_plus_deformation_from_d1")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read data from files
    def read_landmarks(filepath):
        landmarks = {}
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                landmark_name, x, y, z = row
                landmarks[landmark_name] = np.array([float(x), float(y), float(z)])
        return landmarks

    landmarks1 = read_landmarks(file1_path)
    landmarks2 = read_landmarks(file2_path)

    # Compare landmarks
    differences = {}
    landmark_names = []
    distance_diffs = []

    print("Landmark Comparison:")
    print("-" * 50)
    print(f"{ 'Landmark':<15} {'Distance Difference':<25}")
    print("-" * 50)

    for name in landmarks1:
        if name in landmarks2:
            diff_vector = landmarks1[name] - landmarks2[name]
            distance = np.linalg.norm(diff_vector)
            differences[name] = {
                "vector": diff_vector,
                "distance": distance
            }
            landmark_names.append(name)
            distance_diffs.append(distance)
            print(f"{name:<15} {distance:<25.6f}")

    if not differences:
        print("No common landmarks found to compare.")
        return

    # Plotting
    plt.figure(figsize=(15, 8))
    plt.bar(landmark_names, distance_diffs, color='skyblue')
    plt.xlabel("Landmark")
    plt.ylabel("Euclidean Distance Difference")
    plt.title("Difference in Landmark Locations")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(output_folder, "landmark_comparison.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    plt.close()

    # 3D Scatter plot of differences
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    positions1 = np.array(list(landmarks1.values()))
    positions2 = np.array(list(landmarks2.values()))

    ax.scatter(positions1[:,0], positions1[:,1], positions1[:,2], c='b', marker='o', label='File 1 (mb_file_offline)')
    ax.scatter(positions2[:,0], positions2[:,1], positions2[:,2], c='r', marker='^', label='File 2 (joint_file_ooc_r)')

    for name in differences:
        pos1 = landmarks1[name]
        pos2 = landmarks2[name]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 'g-')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Comparison of Landmark Locations')
    ax.legend()
    
    plot_3d_path = os.path.join(output_folder, "landmark_comparison_3d.png")
    plt.savefig(plot_3d_path)
    print(f"3D plot saved to {plot_3d_path}")
    plt.close()


if __name__ == "__main__":
    compare_landmark_files()
