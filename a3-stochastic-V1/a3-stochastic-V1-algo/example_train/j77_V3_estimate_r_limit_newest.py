def calculate_rotation_parameters():

    import shutil
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    import time
    import math
    import os
    import sys
    from os import path
    import importlib
    from paths import b98_sampling_path, a3_stochastic_relative_path, a1_blendshape_path

    # --- Task Setup (Kept as is) ---
    # this_task = 'dataV22_AU0_neutral to dataV22_phoneme_N'
    this_task = 'dataV22_AU0_neutral to dataV22_phoneme_N'
    # from_pose = "dataV22_AU0_neutral"
    # to_pose = "dataV22_phoneme_N"    

    from_pose = "dataV22_AU0_neutral"
    to_pose = "dataV22_phoneme_N"    

    # Copy file from source to destination (Kept as is)
    # copy_from = r"C:/keyLabLocal_D/a3-stochastic-V1-backup-and-results/backup/b31 dataV22 backup hub/dataV22_phoneme_N.py"
    copy_from = r"C:/keyLabLocal_D/a3-stochastic-V1-backup-and-results/backup/b31 dataV22 backup hub/dataV22_phoneme_N.py"
    copy_to = str(a1_blendshape_path / "database"/ "dataV22.py")
    shutil.copy(copy_from, copy_to)

    # maya_file_to_open = r"C:/keyLabLocal_D/c99-maya-ziva-workshop/experiment_files/g10-10 rest pose 1200 landmarks v9.mb"
    maya_file_to_open = r"C:/keyLabLocal_D/c99-maya-ziva-workshop/experiment_files/g10-10 rest pose 1200 landmarks v9.mb"

    # --- System Paths and Module Imports (Kept as is) ---
    sys.path.insert(0, r"C:\Users\zpa18\keyLabLocal\a2-DSAC-T-for-pipelineV1")
    sys.path.insert(0, r"C:\Users\zpa18\keyLabLocal\a2-DSAC-T-for-pipelineV1\example_train")
    sys.path.insert(0, r"C:\Users\zpa18\keyLabLocal\a2-DSAC-T-for-pipelineV1\env_gym")
    sys.path.insert(0, str(a1_blendshape_path / "database"))
    sys.path.insert(0, r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub")
    sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo"))
    sys.path.insert(0, str(b98_sampling_path / "300joints_suite_good_old"))

    # Module imports (Kept as is)
    import joint_position
    import joint_position_first_wave
    import joint_position_third_wave
    importlib.reload(joint_position)
    importlib.reload(joint_position_first_wave)
    importlib.reload(joint_position_third_wave)

    from utils.distance_calculator import MeshDistanceCalculator
    distance_calculator = MeshDistanceCalculator()

    import dataV22_phoneme_N
    importlib.reload(dataV22_phoneme_N)
    import dataV22_phoneme_T
    importlib.reload(dataV22_phoneme_T)
    import dataV22_phoneme_AA
    importlib.reload(dataV22_phoneme_AA)
    import dataV22_phoneme_IY
    importlib.reload(dataV22_phoneme_IY)
    import dataV22_phoneme_V
    importlib.reload(dataV22_phoneme_V)
    import dataV22_phoneme_Y
    importlib.reload(dataV22_phoneme_Y)
    import dataV22_phoneme_UW
    importlib.reload(dataV22_phoneme_UW)
    import dataV22_AU0_neutral
    importlib.reload(dataV22_AU0_neutral)
    import dataV22_phoneme_EH
    importlib.reload(dataV22_phoneme_EH)    
    import dataV22_AU12
    importlib.reload(dataV22_AU12)
    import dataV22_AU22
    importlib.reload(dataV22_AU22)
    import dataV22_AU2_left
    importlib.reload(dataV22_AU2_left)
    import dataV22_AU2_right
    importlib.reload(dataV22_AU2_right)
    import dataV22_AU6_12_25
    importlib.reload(dataV22_AU6_12_25)    

    import dataV22_AU4
    importlib.reload(dataV22_AU4)

    import dataV22_AU1
    importlib.reload(dataV22_AU1)

    import dataV22_AU2
    importlib.reload(dataV22_AU2)

    import dataV22_AU25
    importlib.reload(dataV22_AU25)    

    import dataV22_AU6
    importlib.reload(dataV22_AU6)    

    import dataV22_AU7
    importlib.reload(dataV22_AU7)    






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


    # --- JawAngleCalculator Class (Kept as is) ---
    class JawAngleCalculator:
        def __init__(self):
            # Initialize with your actual vertex indices
            self.vertex_jaw_index = [29, 177, 350]
            self.muscle_vertex_indices = {
                'jaw': [29, 177, 350],
                'OrbicularisOculi_l': vertex_ooc_l_index,  #  vertex indices for left orbicularis oculi
                'OrbicularisOculi_r': vertex_ooc_r_index,  #  right orbicularis oculi
                'Frontalis': vertex_frontalis_index  # vertex indices for frontalis
            }                   
            self.save_folder = "./debug_results"

        def control_vertices_slice(self, target_state, vertex_index):
            """Extract vertex coordinates from target state"""
            combined_slices = []
            for index in vertex_index:
                slice_part = target_state[index*3:index*3+3]
                combined_slices.extend(slice_part)
            return combined_slices

        def calculate_normal_vector(self, vertices):
            """
            Calculate normal vector for a triangle defined by 3 vertices
            vertices: list of 9 elements [x1,y1,z1, x2,y2,z2, x3,y3,z3]
            """
            if len(vertices) != 9:
                raise ValueError("Exactly 3 vertices (9 values) required for normal calculation")
            
            points = np.array(vertices).reshape(3, 3)
            v1 = points[1] - points[0]
            v2 = points[2] - points[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            return normal

        def calculate_angle_between_normals(self, normal1, normal2):
            """Calculate angle between two normal vectors in degrees"""
            dot_product = np.dot(normal1, normal2)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            return angle_deg

        def debug_output(self, normal1, normal2, angle):
            """Save debug information to file"""
            os.makedirs(self.save_folder, exist_ok=True)
            
            with open(os.path.join(self.save_folder, "jaw_angle_debug.txt"), 'a') as f:
                f.write("\n=== Jaw Angle Calculation ===\n")
                f.write(f"Normal vector 1: {normal1}\n")
                f.write(f"Normal vector 2: {normal2}\n")
                f.write(f"Angle between jaws: {angle:.2f} degrees\n")
                f.write("------------------------\n")

        def calculate_jaw_angle(self, x_jaw_pose1, x_jaw_pose2):
            """Main method to calculate jaw angle between two poses"""
            vertices1 = self.control_vertices_slice(x_jaw_pose1, self.vertex_jaw_index)
            vertices2 = self.control_vertices_slice(x_jaw_pose2, self.vertex_jaw_index)
            
            print(f"Vertices1 shape: {len(vertices1)}, values: {vertices1}")
            print(f"Vertices2 shape: {len(vertices2)}, values: {vertices2}")
            
            if len(vertices1) != 9 or len(vertices2) != 9:
                print(f"Warning: Expected 9 elements (3 vertices × 3 coordinates), got {len(vertices1)} and {len(vertices2)}")
                print("Vertex indices used:", self.vertex_jaw_index)
                vertices1 = vertices1[:9]
                vertices2 = vertices2[:9]
            
            try:
                normal1 = self.calculate_normal_vector(vertices1)
                normal2 = self.calculate_normal_vector(vertices2)
                angle = self.calculate_angle_between_normals(normal1, normal2)
                self.debug_output(normal1, normal2, angle)
                return angle
                
            except Exception as e:
                print(f"Error calculating jaw angle: {str(e)}")
                print(f"Vertex data causing issue - vertices1: {vertices1}, vertices2: {vertices2}")
                return None

    # --- Jaw Angle Function (Kept as is) ---
    def calculate_jaw_angle_between_poses(pose1_module_name, pose2_module_name):
        """
        Calculate jaw angle between two poses specified by their module names
        """
        try:
            pose1_module = sys.modules[pose1_module_name]
            pose2_module = sys.modules[pose2_module_name]            
        except KeyError as e:
            print(f"Error: Could not find module {str(e)}")
            return None
        
        calculator = JawAngleCalculator()
        
        try:
            x_jaw_pose1 = getattr(pose1_module, "x_middleres_Jaw" + str(1))
            x_jaw_pose2 = getattr(pose2_module, "x_middleres_Jaw" + str(1))
            
            x_jaw_pose1_conca = [item for sublist in x_jaw_pose1 for item in sublist]
            x_jaw_pose2_conca = [item for sublist in x_jaw_pose2 for item in sublist]
            
            angle = calculator.calculate_jaw_angle(x_jaw_pose1_conca, x_jaw_pose2_conca)
            return angle
            
        except Exception as e:
            print(f"Error calculating jaw angle: {str(e)}")
            return None

    # --- Estimated Steps Function (Modified to return all relevant pose data) ---
    def calculate_estimated_steps(pose1_module_name, pose2_module_name):
        """
        Calculate estimated completion steps between two poses specified by their module names
        """
        try:
            pose1_module = sys.modules[pose1_module_name]
            pose2_module = sys.modules[pose2_module_name]            
            
            # Get the pose data for distance calculation (OrbicularisOris)
            target_state_temp1 = getattr(pose1_module, "x_middleres_OrbicularisOris" + str(1))
            target_state_temp2 = getattr(pose2_module, "x_middleres_OrbicularisOris" + str(1))
            
            # Flatten the pose data
            x_conca1 = [item for sublist in target_state_temp1 for item in sublist]
            x_conca2 = [item for sublist in target_state_temp2 for item in sublist]
            target_state1 = np.asarray(x_conca1)
            target_state2 = np.asarray(x_conca2)
            
            calculator = JawAngleCalculator()
            vertex_index_combined = (joint_position_first_wave.sampled_points_vertex_idx +
                                     joint_position.sampled_points_vertex_idx +
                                     joint_position_third_wave.sampled_points_vertex_idx)
            vertex_index = vertex_index_combined
            
            # Calculate initial distance
            method = 'euclidean'
            initial_distance = distance_calculator.calculate_distance(
                method, 
                calculator.control_vertices_slice(target_state1, vertex_index),
                calculator.control_vertices_slice(target_state2, vertex_index)
            )
            
            # Calculate estimated steps
            estimated_completion_steps = math.floor(43.0073594825 * initial_distance)
            
            # Return full pose data for later use
            x_bar_pose1 = target_state_temp1
            x_bar_pose2 = target_state_temp2 # You might need this for displacement

            return estimated_completion_steps, x_bar_pose1, x_bar_pose2
            
        except Exception as e:
            print(f"Error calculating estimated steps: {str(e)}")
            return None, None, None

    # --- New Function to Calculate Translation Distance for a Region ---
    def calculate_region_translation_distance(pose1_module_name, pose2_module_name, region_name):
        """
        Calculate the Euclidean distance between a specific region's vertices in two poses.
        
        Args:
            region_name (str): e.g., 'OrbicularisOculiLeft', 'Frontalis'
        
        Returns:
            float: Euclidean distance, or None if calculation fails.
        """
        calculator = JawAngleCalculator()
        
        try:
            pose1_module = sys.modules[pose1_module_name]
            pose2_module = sys.modules[pose2_module_name]
            
            # Assuming pose data is stored in 'x_middleres_<RegionName>1' format
            data_name = "x_middleres_" + region_name + str(1)
            
            x_pose1 = getattr(pose1_module, data_name)
            x_pose2 = getattr(pose2_module, data_name)
            
            # Flatten the pose data
            x_conca1 = np.asarray([item for sublist in x_pose1 for item in sublist])
            x_conca2 = np.asarray([item for sublist in x_pose2 for item in sublist])
            
            # Calculate L2 (Euclidean) distance
            # distance = np.linalg.norm(x_conca1 - x_conca2)
            # Calculate initial distance
            
            method = 'euclidean'
            distance = distance_calculator.calculate_distance(
                method, 
                calculator.control_vertices_slice(x_conca1, calculator.muscle_vertex_indices[region_name]),
                calculator.control_vertices_slice(x_conca2, calculator.muscle_vertex_indices[region_name])
            )

            return distance
            
        except Exception as e:
            print(f"Error calculating distance for {region_name}: {str(e)}")
            return None

    # --- PART 1: Jaw Angle and Estimated Steps Calculation ---
    angle = calculate_jaw_angle_between_poses(from_pose, to_pose)
    if abs(angle - 0) < 1e-6:
        angle = 1e-6
        angle_0 = 0
        print(f"Jaw angle between poses: {angle_0:.2f} degrees (adjusted for optimization)")
    
    if angle is not None:
        print(f"Jaw angle between poses: {angle:.2f} degrees")
    else:
        print("Failed to calculate jaw angle")

    estimated_steps, x_bar, _ = calculate_estimated_steps(from_pose, to_pose)
    
    if estimated_steps is not None:
        print(f"Estimated completion steps: {estimated_steps}")
    else:
        print("Failed to calculate estimated steps")
        # Handle the case where estimated_steps is None/0, potentially by exiting or setting a default
        estimated_steps = 1 # Fallback to prevent division by zero, but actual error handling is better

    # --- PART 2: Translation Distance Calculations for Other Regions ---
    target_ooc_l_translation = calculate_region_translation_distance(from_pose, to_pose, "OrbicularisOculi_l")
    target_ooc_r_translation = calculate_region_translation_distance(from_pose, to_pose, "OrbicularisOculi_r")
    target_frontalis_translation = calculate_region_translation_distance(from_pose, to_pose, "Frontalis")
    
    if target_ooc_l_translation is not None:
        print(f"Target OOC_L translation distance: {target_ooc_l_translation:.6f}")
    if target_ooc_r_translation is not None:
        print(f"Target OOC_R translation distance: {target_ooc_r_translation:.6f}")
    if target_frontalis_translation is not None:
        print(f"Target Frontalis translation distance: {target_frontalis_translation:.6f}")

    # --- PART 3: Optimization Functions (Modified) ---

    def omega(t, T, magnitude, k=0.004):
        """Velocity profile function: sigmoid transition with an envelope"""
        if T <= 0: return np.zeros_like(t)
        t1 = 0.07 * T
        t2 = 0.93 * T
        sigmoid1 = 1 / (1 + np.exp(-k * (t - t1)))
        sigmoid2 = 1 / (1 + np.exp(-k * (t - t2)))
        envelope = (1 - np.exp(-k * t)) * (1 - np.exp(-k * (T - t)))
        return magnitude * (sigmoid1 - sigmoid2) * envelope

    def calculate_movement_characteristics(magnitude, target_steps, target_movement):
        """
        Calculate the total movement for given parameters using the omega profile.
        Movement can be rotation (degrees) or translation (units).
        """
        if target_steps <= 0:
            print(f"ERROR: target_steps must be positive, got {target_steps}")
            return 0, 0, np.array([]), np.array([]), np.array([])

        time_points = np.arange(0, target_steps)
        velocities = omega(time_points, target_steps, magnitude)
        
        # Guard against NaN/Inf
        velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
        
        movement_per_step = velocities
        cumulative_movement = np.cumsum(movement_per_step)
        
        if len(cumulative_movement) == 0:
            total_movement = 0
            completed_steps = 0
        else:
            total_movement = cumulative_movement[-1]
            # Find when 99% of movement is complete
            threshold = 0.99 * target_movement
            condition = cumulative_movement >= threshold
            completed_steps = np.argmax(condition)
            if not np.any(condition):
                # If 99% is never reached, use the full step count
                completed_steps = target_steps

        return total_movement, completed_steps, time_points, velocities, cumulative_movement

    def plot_results(time_points, velocities, cumulative_movement, target_movement, target_steps, optimal_magnitude, label):
        """Visualize the velocity profile and cumulative movement"""
        plt.figure(figsize=(12, 6))
        steps = np.arange(len(time_points))
        save_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files"

        # Velocity profile
        plt.subplot(1, 2, 1)
        plt.plot(steps, velocities, 'b-', label='Velocity')
        plt.xlabel('Steps')
        plt.ylabel(f'{label} Velocity')
        plt.title(f'{label} Velocity Profile\nOptimal Magnitude = {optimal_magnitude:.6e}')
        plt.grid(True)
        
        # Cumulative movement
        plt.subplot(1, 2, 2)
        plt.plot(steps, cumulative_movement, 'r-', label=f'Cumulative {label}')
        plt.axhline(y=target_movement, color='g', linestyle='--', label=f'Target {label}')
        plt.axhline(y=0.99*target_movement, color='m', linestyle=':', label='99% Target')
        plt.xlabel('Steps')
        plt.ylabel(f'{label}')
        plt.title(f'Cumulative {label}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, f'{label.lower().replace(" ", "_")}_optimization_plot.png'), bbox_inches='tight', dpi=300)
        plt.close()

    def objective_function_template(params, target_steps, target_movement):
        magnitude = params[0]
        total_movement, completed_steps, _, _, _ = calculate_movement_characteristics(
            magnitude, target_steps, target_movement)
        
        if target_movement == 0:
             # If target is 0, goal is for total_movement to be 0
             return abs(total_movement) 

        # Weights: prioritize achieving the target rotation/translation, then matching steps
        movement_error = abs(total_movement - target_movement) / target_movement
        steps_error = abs(completed_steps - target_steps) / target_steps
        
        return movement_error + 0.3 * steps_error


    def find_optimal_parameters(target_steps, target_movement, region_name):
        # Safely handle zero or None target movement
        if target_movement is None or target_movement <= 1e-6:
            print(f"Target movement for {region_name} is near zero ({target_movement}). Returning 0.0 limit.")
            return 0.0

        # Initial bounds search, adjust based on expected magnitude (rotation vs translation)
        if region_name == 'Jaw Rotation':
            # Rotation is in degrees, likely a smaller magnitude
            up_limit = 0.1 
            down_limit = 0.0000001
        else:
            # Translation is in world units, may need a wider/higher bound
            # Initial guess: translation limits are often smaller than rotation velocity in this context
            up_limit = 0.1 
            down_limit = 0.000001
            
        middle = (down_limit + up_limit) / 2
        bounds = [(down_limit, up_limit)]

        result = minimize(
            objective_function_template,
            x0=[middle],
            args=(target_steps, target_movement),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        optimal_magnitude = result.x[0]

        final_movement, final_steps, time_points, velocities, cum_movement = calculate_movement_characteristics(
            optimal_magnitude, target_steps, target_movement)
        
        plot_results(time_points, velocities, cum_movement, target_movement, target_steps, optimal_magnitude, region_name)

        return optimal_magnitude

    # --- PART 4: Optimization Execution ---
    print("\n--- Starting Optimization ---")
    
    # 0. t_limit. for oor
    # heuristic value
    # expected_total_steps = 5200  # 6000
    translation_anatomy_limit = 0.014  # 35/30/(expected_total_steps/300)*0.208
    t_limit = translation_anatomy_limit
    print(f"Recommended t_limit: {t_limit:.8e}")
    # --- Smart Synchronization Preparation ---
    # Estimate steps for all regions using the same formula (43.0073594825 * distance)
    # and find the maximum steps needed to synchronize all tissues smoothly.
    
    estimated_completion_steps_ooc_l = 0
    if target_ooc_l_translation is not None:
        estimated_completion_steps_ooc_l = math.floor(43.0073594825 * target_ooc_l_translation)
        
    estimated_completion_steps_ooc_r = 0
    if target_ooc_r_translation is not None:
        estimated_completion_steps_ooc_r = math.floor(43.0073594825 * target_ooc_r_translation)
        
    estimated_completion_steps_frontalis = 0
    if target_frontalis_translation is not None:
        estimated_completion_steps_frontalis = math.floor(43.0073594825 * target_frontalis_translation)
        
    all_steps = [estimated_steps, estimated_completion_steps_ooc_l, estimated_completion_steps_ooc_r, estimated_completion_steps_frontalis]
    sync_target_steps = max(all_steps)
    if sync_target_steps < 1:
        sync_target_steps = 1
        
    print(f"\n--- Smart Synchronization ---")
    print(f"Individual estimated steps: OOR={estimated_steps}, OOC_L={estimated_completion_steps_ooc_l}, OOC_R={estimated_completion_steps_ooc_r}, Frontalis={estimated_completion_steps_frontalis}")
    print(f"Synchronized global target steps: {sync_target_steps}")

    # 1. Jaw Rotation (r_limit)
    target_rotation_deg = angle * 0.95 
    r_limit = 0.0
    if sync_target_steps > 0 and target_rotation_deg > 1e-6:
        print(f"\nOptimizing Jaw Rotation...")
        r_limit = find_optimal_parameters(sync_target_steps, target_rotation_deg, 'Jaw Rotation')
    print(f"Recommended r_limit: {r_limit:.8e}")

    # 2. OOC_L Translation (ooc_l_limit)
    ooc_l_limit = 0.0
    if target_ooc_l_translation is not None and target_ooc_l_translation > 1e-6:
        print(f"\nOptimizing OOC_L Translation...")
        ooc_l_limit = find_optimal_parameters(sync_target_steps, target_ooc_l_translation, 'OOC_L Translation')
    print(f"Recommended ooc_l_limit: {ooc_l_limit:.8e}")

    # 3. OOC_R Translation (ooc_r_limit)
    ooc_r_limit = 0.0
    if target_ooc_r_translation is not None and target_ooc_r_translation > 1e-6:
        print(f"\nOptimizing OOC_R Translation...")
        ooc_r_limit = find_optimal_parameters(sync_target_steps, target_ooc_r_translation, 'OOC_R Translation')
    print(f"Recommended ooc_r_limit: {ooc_r_limit:.8e}")

    # 4. Frontalis Translation (frontalis_limit)
    frontalis_limit = 0.0
    if target_frontalis_translation is not None and target_frontalis_translation > 1e-6:
        print(f"\nOptimizing Frontalis Translation...")
        frontalis_limit = find_optimal_parameters(sync_target_steps, target_frontalis_translation, 'Frontalis Translation')
    print(f"Recommended frontalis_limit: {frontalis_limit:.8e}")

    # Update global estimated_steps to the synchronized max so downstream processes and logging use it
    estimated_steps = sync_target_steps

    # --- Log Results to File ---
    log_dir = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\debug nonsync"
    os.makedirs(log_dir, exist_ok=True)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"j77_calculation_{from_pose}_to_{to_pose}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, 'w') as log_file:
        log_file.write(f"Task: {this_task}\n")
        log_file.write(f"From Pose: {from_pose}\n")
        log_file.write(f"To Pose: {to_pose}\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Maya File: {maya_file_to_open}\n")
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"CALCULATION RESULTS\n")
        log_file.write(f"{'='*60}\n\n")
        log_file.write(f"                        'degree': {target_rotation_deg}, # rotation angle\n")
        log_file.write(f"                        'expected_steps': {estimated_steps},\n")
        log_file.write(f"                        'rotation_limit': {r_limit:.8f},\n")
        log_file.write(f"                        'translation_limit': {t_limit:.8f},\n")
        log_file.write(f"                        'ooc_l_translation_limit': {ooc_l_limit:.8f},\n")
        log_file.write(f"                        'ooc_r_translation_limit': {ooc_r_limit:.8f},\n")
        log_file.write(f"                        'frontalis_translation_limit': {frontalis_limit:.8f},\n")
        log_file.write(f"\n")
        log_file.write(f"Estimated OOC_L completion steps: {estimated_completion_steps_ooc_l}\n")
        log_file.write(f"Estimated OOC_R completion steps: {estimated_completion_steps_ooc_r}\n")

    print(f"\n[LOG] Results saved to: {log_path}")

    # --- Final Return ---
    return target_rotation_deg, estimated_steps, t_limit, r_limit, ooc_l_limit, ooc_r_limit, frontalis_limit, this_task, x_bar, maya_file_to_open, estimated_completion_steps_ooc_l, estimated_completion_steps_ooc_r

# Example of how to call the function outside the definition:
# target_rotation_deg, estimated_steps, r_limit, ooc_l_limit, ooc_r_limit, frontalis_limit, this_task, x_bar, maya_file_to_open = r"C:/keyLabLocal_D/c99-maya-ziva-workshop/experiment_files/g10-10 rest pose 1200 landmarks v9.mb"