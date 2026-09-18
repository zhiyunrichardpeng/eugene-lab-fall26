import maya.standalone
maya.standalone.initialize()    
import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import sys
import os
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtWidgets import QApplication, QLabel        

class MayaDebug:
    def __init__(self):
        # Set the attribute before creating the QApplication
        QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

        self.app = QApplication(sys.argv)
        self.maya_file_to_open = r"C:\keyLabLocal_D\c99-maya-ziva-workshop\different pose as rest pose\AU2_right as rest pose_V2.mb"
        self.opened_file = None
        # self.save_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\AU2_right_as_rest_pose\mb_file_offline"
        # os.makedirs(self.save_folder, exist_ok=True)

        # Making the script self-contained
        self.cmds = cmds
        self.debug_mode_vsCode = 1  # Assuming this is a debug script
        
        # Set the number of landmarks to check for each group
        self.joint_ooc_r_num = 300
        self.joint_ooc_l_num = 300
        self.joint_frontalis_num = 300
        self.joint_oor_num = 300

    def plot_landmarks(self, landmarks, title, filename):
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
            
            # === MODIFICATION START ===
            # Adjusted view:
            # elev: 30 degrees (slightly higher elevation for a better top-down curve)
            # azim: -130 degrees (rotated further to show more of the "front" of the eye structure)
            # ax.view_init(elev=30, azim=-130)
            ax.view_init(elev=(180 + 30 - 90), azim=0, roll=90)  # Change these values to set the desired view
            # === MODIFICATION END ===
            
            # Set axes limits to be equal to enforce a cubic feel, improving the 3D perception
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

    def run(self):
        # cmds.loadPlugin("Mayatomr") # Load all plugins you might need

        # Have a namespace if you want (recommended)
        # self.namespace = "ref" 

        if not os.path.exists(self.maya_file_to_open):
            raise RuntimeError(f"File not found: {self.maya_file_to_open}")

        # Check for unsaved changes and handle them
        if cmds.file(query=True, modified=True):
            # Save or discard changes as needed
            print("Unsaved changes detected.")
            # save_changes = input("You have unsaved changes. Do you want to save them before opening the new file? (yes/no): ").strip().lower()
            if 1==0:  # save_changes in ['yes', 'y']:
                cmds.file(save=True)
                print("Changes saved.")
            else:
                cmds.file(newFile=True, force=True)  # Discard changes

        # Open the Maya file
        try:
            self.opened_file = cmds.file(self.maya_file_to_open, open=True, force=True)
            print("Opened file:", self.opened_file)

            cmds.playbackOptions(min=1, max=50, animationStartTime=1, animationEndTime=50)

            # Function to delete existing cache nodes
            def delete_cache_nodes():
                cache_nodes = cmds.ls(type='cacheFile')
                for node in cache_nodes:
                    cmds.delete(node)

            # Delete existing cache nodes
            delete_cache_nodes()

            cmds.cacheFile(createCacheNode=True, singleCache=True, startTime=1, endTime=50, fileName='myCache', dir='cacheDir')

            cmds.currentTime(15)

        except Exception as e:
            print(f"Failed to open Maya file: {e}")

        solvers = cmds.ziva(s=True)
        if solvers:
            solver = solvers[0]
            print(f"Found zSolver: {solver}")
        ziva_nodes = cmds.ls(type='zSolverTransform')
        print(f"Ziva solver nodes: {ziva_nodes}")


        # --- Debugging Job: Retrieve and Save Landmark Locations ---
        # This section saves the visualized locations of landmarks to a file for comparison.

        # Setup CSV for landmark locations
        landmark_locations_file = os.path.join(self.save_folder, "a99_debug_landmark_locations.csv")
        
        with open(landmark_locations_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['landmark_name', 'x', 'y', 'z'])

            print("\n--- Retrieving and Saving Landmark Locations ---")

            # Define landmark groups to process
            landmark_groups = {
                'ooc_r': self.joint_ooc_r_num,
                # 'ooc_l': self.joint_ooc_l_num,
                # 'frontalis': self.joint_frontalis_num,
                # 'oor': self.joint_oor_num
            }

            for group_name, num_landmarks in landmark_groups.items():
                print(f"\nProcessing group: {group_name}")
                found_count = 0
                landmark_locations_for_plot = []
                for i in range(num_landmarks):
                    # Determine the joint name based on the index i
                    if i + 1 <= 9:
                        joint_name = f'{group_name}_0{i + 1}'
                    else:
                        joint_name = f'{group_name}_{i + 1}'

                    # Check if the landmark object exists
                    if self.cmds.objExists(joint_name):
                        found_count += 1
                        location = self.cmds.xform(joint_name, query=True, translation=True, worldSpace=True)
                        
                        # Print a small sample to console for each group
                        if found_count <= 3:
                            print(f"  Location of {joint_name} is: {location}")
                        
                        # Save all to CSV
                        # csv_writer.writerow([joint_name, location[0], location[1], location[2]])
                        # landmark_locations_for_plot.append([joint_name, location[0], location[1], location[2]])
                
                # if found_count == 0:
                #     print(f"  No landmarks found for group '{group_name}'.")
                # else:
                #     print(f"  Found and saved {found_count} landmarks for group '{group_name}'.")
                #     plot_filename = os.path.join(self.save_folder, f"plot_{group_name}_landmarks.png")
                #     self.plot_landmarks(landmark_locations_for_plot, f"Landmark Locations: {group_name}", plot_filename)

            # print(f"\nSaved all found landmark locations to: {landmark_locations_file}")


if __name__ == '__main__':
    debug_app = MayaDebug()
    debug_app.run()
