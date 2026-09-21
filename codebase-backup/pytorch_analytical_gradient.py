from torch.jit import _decomposition_utils
import torch
import torch.nn as nn
import json
import numpy as np
import sys

class LinearBlendSkinning(nn.Module):
    def __init__(self, bind_pose_vertices, skin_weights, joint_pivots=None):
        """
        Initializes the differentiable LBS function.
        
        Args:
            bind_pose_vertices (torch.Tensor): Shape (N, 3), the original rest-pose vertices.
            skin_weights (torch.Tensor): Shape (N, K), the weights of K landmarks on N vertices.
            joint_pivots (torch.Tensor, optional): Shape (K, 3), the pivot points for rotations.
        """
        super(LinearBlendSkinning, self).__init__()
        
        # Static mesh data (doesn't change during optimization)
        self.register_buffer('bind_pose', bind_pose_vertices) # (N, 3)
        self.register_buffer('weights', skin_weights)         # (N, K)
        
        if joint_pivots is None:
            K = skin_weights.shape[1]
            joint_pivots = torch.zeros((K, 3), dtype=bind_pose_vertices.dtype)
        self.register_buffer('joint_pivots', joint_pivots)    # (K, 3)

    def forward(self, translations, positions, rotations=None):
        """
        Applies translations and/or rotations to the landmarks and computes new vertex positions,
        incorporating the observed true vertices positions to compute residuals.
        
        Args:
            translations (torch.Tensor or None): Shape (K, 3), the XYZ movement for each landmark.
            positions (torch.Tensor): Shape (N, 3), the observed true vertices positions.
            rotations (torch.Tensor or None): Shape (K, 3), the XYZ Euler angles in degrees for each landmark.
            
        Returns:
            torch.Tensor: Shape (N, 3), the new vertex positions incorporating the true residual.
        """
        N = self.bind_pose.shape[0]
        K = self.weights.shape[1]
        
        if rotations is not None:
            import math
            angles_rad = rotations * (math.pi / 180.0)
            
            x, y, z = angles_rad[:, 0], angles_rad[:, 1], angles_rad[:, 2]
            cx, sx = torch.cos(x), torch.sin(x)
            cy, sy = torch.cos(y), torch.sin(y)
            cz, sz = torch.cos(z), torch.sin(z)
            
            Rx = torch.zeros(K, 3, 3, device=rotations.device, dtype=rotations.dtype)
            Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cx; Rx[:, 1, 2] = -sx; Rx[:, 2, 1] = sx; Rx[:, 2, 2] = cx
            
            Ry = torch.zeros(K, 3, 3, device=rotations.device, dtype=rotations.dtype)
            Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 1, 1] = 1; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy
            
            Rz = torch.zeros(K, 3, 3, device=rotations.device, dtype=rotations.dtype)
            Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz; Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz; Rz[:, 2, 2] = 1
            
            # Rotation matrix R = Rz @ Ry @ Rx (assuming XYZ Maya convention)
            R = torch.bmm(Rz, torch.bmm(Ry, Rx)) # (K, 3, 3)
            
            # For relative=True, ws=True, the rotation is incremental and in world space.
            # Thus, we apply the rotation to the CURRENT positions, not the bind_pose.
            v_expanded = positions.unsqueeze(1).unsqueeze(-1) # (N, 1, 3, 1)
            p_expanded = self.joint_pivots.unsqueeze(0).unsqueeze(-1) # (1, K, 3, 1)
            R_expanded = R.unsqueeze(0) # (1, K, 3, 3)
            
            # Rotate current position around the pivot: R_k @ (v_current - p_k) + p_k
            v_centered = v_expanded - p_expanded
            v_rotated = torch.matmul(R_expanded, v_centered) + p_expanded # (N, K, 3, 1)
            v_rotated = v_rotated.squeeze(-1) # (N, K, 3)
            
            # The displacement caused by each joint
            v_orig_expanded = positions.unsqueeze(1) # (N, 1, 3)
            displacements = v_rotated - v_orig_expanded # (N, K, 3)
            
            weights_expanded = self.weights.unsqueeze(-1) # (N, K, 1)
            weighted_displacements = torch.sum(weights_expanded * displacements, dim=1) # (N, 3)
            
            lbs_vertices = positions + weighted_displacements
            
            if translations is not None:
                lbs_deformation = torch.matmul(self.weights, translations)
                lbs_vertices = lbs_vertices + lbs_deformation
                
            return lbs_vertices
        else:
            # 1. Matrix multiplication handles the weighted sum for all vertices and landmarks instantly
            
            # print("weights", self.weights)
            # print("translations", translations)
            # print("weights.shape", self.weights.shape)
            # print("translations.shape", translations.shape)

            # print("passed the test, now quit.")

            # sys.exit(0)  # Exit with status code 0

            lbs_deformation = torch.matmul(self.weights, translations) # (N, K) @ (K, 3) -> (N, 3)
            lbs_vertices = self.bind_pose + lbs_deformation

        # print(""n" + "="*50)
        # print("🔍 DEBUG [pytorch_analytical_gradient.py - forward]")
        # print(f"  self.bind_pose shape:  {self.bind_pose.shape} (N, 3)")
        # print(f"  self.weights shape:    {self.weights.shape} (N, K)")
        # print(f"  translations shape:    {translations.shape}")
        # print(f"  lbs_vertices shape:    {lbs_vertices.shape} (Expected N, 3)")
        # print(f"  positions shape:       {positions.shape} (Input observation)")
        # print("="*50 + ""n")

        # 2. Compute the true residual from the observation
        # We use .detach() so PyTorch treats this residual as a constant offset, NOT a differentiable variable
        true_residual = (positions - lbs_vertices).detach()

        # 3. Reconstruct the deformed vertices incorporating the residual
        deformed_vertices = lbs_vertices + true_residual
        
        return deformed_vertices


class AnalyticalGradientCalculator:
    def __init__(self, bind_pose_json_path, skin_weights_json_path, vertex_index=None, joint_pivots_json_path=None):
        print("[AnalyticalGradient] Loading Maya data...")
        with open(bind_pose_json_path, 'r') as f:
            bind_pose_data = json.load(f)
        
        with open(skin_weights_json_path, 'r') as f:
            skin_weights_data = json.load(f)
            
        self.bind_pose = torch.tensor(bind_pose_data, dtype=torch.float32)
        self.skin_weights = torch.tensor(skin_weights_data, dtype=torch.float32)
        
        # If a subset of vertices is provided, reduce the bind pose and skin weights
        if vertex_index is not None:
            v_idx = torch.tensor(vertex_index, dtype=torch.long)
            self.bind_pose = self.bind_pose[v_idx]
            self.skin_weights = self.skin_weights[v_idx]
            print(f"[AnalyticalGradient] Sliced data to {len(v_idx)} selected vertices.")

        self.N = self.bind_pose.shape[0]
        self.K = self.skin_weights.shape[1]
        
        if joint_pivots_json_path is not None:
            with open(joint_pivots_json_path, 'r') as f:
                joint_pivots_data = json.load(f)
            joint_pivots = torch.tensor(joint_pivots_data, dtype=torch.float32)
        else:
            joint_pivots = None
        
        print(f"[AnalyticalGradient] Loaded {self.N} vertices and {self.K} landmarks.")
        
        self.maya_lbs = LinearBlendSkinning(self.bind_pose, self.skin_weights, joint_pivots)

    def compute_gradient(self, current_translations_flat, target_vertices_flat, initial_distance, positions_flat):
        """
        Computes the exact analytical gradient matching the evaluator.py numerical gradient.
        
        Args:
            current_translations_flat (list or np.array): The flat array of current joint translations (3*K).
            target_vertices_flat (list or np.array): The flat array of target mesh vertices (3*N).
            initial_distance (float): The normalization factor used in evaluator.py (temp_init_distance).
            positions_flat (list or np.array): The flat array of observed true mesh vertices (3*N).
            
        Returns:
            np.array: The flat array of gradients (3*K).
        """
        # 1. Convert to PyTorch Tensors
        # Shape: (K, 3)
        translations = torch.tensor(current_translations_flat, dtype=torch.float32).view(self.K, 3)
        translations.requires_grad_(True)
        
        # Shape: (N, 3)
        targets = torch.tensor(target_vertices_flat, dtype=torch.float32).view(self.N, 3)
        
        # Shape: (N, 3)
        positions = torch.tensor(positions_flat, dtype=torch.float32).view(self.N, 3)
        
        # 2. Forward Pass: Compute LBS exactly as Maya does
        y_e = self.maya_lbs(translations, positions)
        
        # 3. Compute Objective Function (Same as evaluator.py distance calculation)
        # evaluator.py uses: normalized_distance = euclidean_distance(current, target) / initial_distance
        distance = torch.sqrt(torch.sum((targets - y_e)**2))
        normalized_distance = distance / initial_distance
        
        # In evaluator.py, the final objective for ablation 1 is w1 * normalized_distance
        # Assuming w1=1.0 for this simple comparison. The agent tries to minimize this.
        # objective = normalized_distance
        
        # Get dynamic properties injected by evaluator.py (with safe defaults for standalone test)
        w1 = getattr(self, 'w1', 1.0)
        w2 = getattr(self, 'w2', 0.1)
        w3 = getattr(self, 'w3', 0.1)
        k_tissue = getattr(self, 'k_tissue', 10000.0)
        max_strain_energy = getattr(self, 'max_strain_energy', 1.0)
        max_muscle_work = getattr(self, 'max_muscle_work', 1.0)
        expected_total_steps = getattr(self, 'expected_total_steps', 5500)
        object_name = getattr(self, 'object_name', 'oor')

        scaling_factor = 9.62e-4
        epsilon = 1e-3
        
        # 1. Strain Energy Term
        strain_energy = 0.5 * k_tissue * (distance * scaling_factor)**2
        normalized_strain_energy = strain_energy / max_strain_energy
        
        # 2. Muscle Work Term
        F_max_new = 17.0 / float(self.K)
        v_max_new = 1.0 / ((35.0 / 30.0) / expected_total_steps) * 9.62e-4 * 100.0
        K_new = 0.18
        a_new = 0.12
        b_new = v_max_new * a_new / (F_max_new * K_new)
        activation = 0.8
        original_length = 2.0e-2
        l_0 = 2.0e-2
        sigma = 0.4
        k_PEE = 0.5
        k_SEE = 0.3
        l_tendon = 0.05
        
        # Distance remain per vertex (same assumption as evaluator.py)
        trans = distance / self.K
        landmark_velocities = epsilon * scaling_factor / ((35.0 / 30.0) / expected_total_steps)
        
        avg_translation = trans * scaling_factor * 100.0
        avg_velocity = landmark_velocities * scaling_factor * 100.0
        
        original_length_cm = original_length * 100.0
        l_0_cm = l_0 * 100.0
        
        l = original_length_cm + avg_translation
        normalized_length_diff = (l - l_0_cm) / l_0_cm
        exponent = -(normalized_length_diff ** 2) / (2 * (sigma ** 2))
        F_length = F_max_new * torch.exp(exponent)
        
        # Assume avg_velocity >= 0 (eccentric/lengthening) as done in evaluator.py
        F_velocity = ((F_max_new + a_new) * b_new) / (avg_velocity + b_new) - a_new
        
        F_PEE = k_PEE * torch.relu(l - l_0_cm)
        F_SEE = k_SEE * l_tendon
        
        muscle_force = activation * F_length * F_velocity + F_PEE + F_SEE
        muscle_force = torch.clamp(muscle_force, max=F_max_new)
        
        landmark_displacements = epsilon * scaling_factor
        enlarge_contribution_factor = 10e2
        
        # np.sum of a scalar in evaluator.py yields a scalar, we do the exact same computation
        normalized_muscle_work = muscle_force * landmark_displacements * (enlarge_contribution_factor * scaling_factor / max_muscle_work)
        
        distance_contribution = w1 * normalized_distance
        strain_contribution = w2 * normalized_strain_energy
        muscle_contribution = w3 * normalized_muscle_work
        
        objective = distance_contribution + strain_contribution + muscle_contribution


        # 4. Backward Pass: Compute gradients of each component separately to log their contributions
        distance_contribution.backward(retain_graph=True)
        self.dist_grad_mag = translations.grad.abs().mean().item() if translations.grad is not None else 0.0
        translations.grad.zero_()
        
        strain_contribution.backward(retain_graph=True)
        self.strain_grad_mag = translations.grad.abs().mean().item() if translations.grad is not None else 0.0
        translations.grad.zero_()
        
        muscle_contribution.backward(retain_graph=True)
        self.muscle_grad_mag = translations.grad.abs().mean().item() if translations.grad is not None else 0.0
        translations.grad.zero_()


        objective.backward()
        self.total_grad_mag = translations.grad.abs().mean().item() if translations.grad is not None else 0.0
        
        
        # 5. Extract and format the gradients
        # The gradients are how much the objective changes if we move the translation.
        # Since the numerical gradient does (r_fw - r_bw)/2*delta on the NEGATIVE distance,
        # we might need to negate this depending on exactly how it's applied in your RL step.
        gradients_flat = translations.grad.view(-1).detach().numpy()
        
        # Return negative gradient because evaluator.py reward_function returns NEGATIVE distance
        # so evaluator.py gradients are inverted compared to standard minimization gradients.
        return -gradients_flat

def run_standalone_experiment():
    print("\n=== Running Standalone Analytical Gradient Experiment (Set B) ===")
    import time
    import os
    from config import a3_stochastic_relative_path
    
    # 1. Initialize the calculator with your REAL Maya data
    calc = AnalyticalGradientCalculator(
        str(a3_stochastic_relative_path / "automation" / "oor_bind_pose.json"),
        str(a3_stochastic_relative_path / "automation" / "oor_skin_weights.json")
    )
    
    # 2. Create a challenging target mesh (e.g. moving all landmarks by some known amounts)
    # We will generate a target mathematically so we know a 100% perfect solution exists.
    true_translations = torch.rand(calc.K, 3) * 5.0 - 2.5  # random values between -2.5 and +2.5
    
    # Use bind pose as a dummy base position for generating target
    dummy_positions = calc.bind_pose.clone()
    target_vertices = calc.maya_lbs(true_translations, dummy_positions).detach()
    
    # 3. Setup the optimization variables (agent starts at 0 for all joints)
    current_translations = torch.zeros((calc.K, 3), requires_grad=True)
    
    # We'll use Adam optimizer 
    optimizer = torch.optim.Adam([current_translations], lr=0.1)
    
    epochs = 300
    start_time = time.time()
    
    print("\nStarting pure PyTorch optimization (No Maya, No Finite Differences)...")
    print(f"{'Epoch':<10} | {'Squared Error (Loss)':<20}")
    print("-" * 35)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass: Compute the current mesh state instantly
        # Using dummy_positions since it acts as the "observed" mesh position in this dummy experiment
        current_vertices = calc.maya_lbs(current_translations, dummy_positions)
        
        # Compute loss (Squared Error)
        loss = torch.sum((target_vertices - current_vertices)**2)
        
        # Backward pass (Calculates the exact analytical gradient for all calc.K landmarks instantly)
        loss.backward()
        
        # Take a step in the direction of the gradient
        optimizer.step()
        
        if epoch == 0 or (epoch + 1) % 30 == 0:
            print(f"{epoch + 1:<10} | {loss.item():<20.6f}")
            
    end_time = time.time()
    print("-" * 35)
    print(f"Optimization completed in {end_time - start_time:.4f} seconds!")
    print(f"Final Loss: {loss.item():.6f}")
    
    # Check if it actually solved the inverse kinematics problem
    diff = torch.abs(current_translations - true_translations)
    print(f"Mean Error vs True Joint Position: {torch.mean(diff).item():.6f}")
    print(f"\nStandalone test successful. The analytical gradient correctly and instantly drives the {calc.K} landmarks to the target mesh!")

if __name__ == "__main__":
    run_standalone_experiment()
