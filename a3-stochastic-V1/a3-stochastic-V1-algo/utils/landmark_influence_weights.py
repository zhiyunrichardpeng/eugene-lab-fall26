"""
Landmark Influence Weight Utility Module

This module provides functionality to compute and apply influence-weighted 
displacement clamping for landmark mesh animation.

The core idea: Landmarks with low influence on the high-resolution mesh 
should have their displacements clamped more aggressively to prevent 
visual artifacts in landmark mesh animations.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Dict, Tuple, Optional
import pickle
import os


class LandmarkInfluenceWeights:
    """
    Computes and manages influence weights for landmarks based on their
    effect on the high-resolution mesh deformation.
    """
    
    def __init__(self, 
                 landmark_positions: np.ndarray,
                 mesh_vertex_positions: np.ndarray,
                 landmark_to_mesh_indices: List[int],
                 base_displacement_limit: float = 0.01,
                 influence_epsilon: float = 0.01,
                 cache_file: Optional[str] = None):
        """
        Initialize the influence weight calculator.
        
        Args:
            landmark_positions: (N, 3) array of landmark positions
            mesh_vertex_positions: (M, 3) array of high-res mesh vertex positions
            landmark_to_mesh_indices: List mapping landmark idx -> mesh vertex idx
            base_displacement_limit: Maximum displacement for highest-influence landmarks
            influence_epsilon: Small value to prevent division by zero
            cache_file: Optional path to cache influence weights
        """
        self.landmark_positions = np.array(landmark_positions)
        self.mesh_vertex_positions = np.array(mesh_vertex_positions)
        self.landmark_to_mesh_indices = landmark_to_mesh_indices
        self.base_displacement_limit = base_displacement_limit
        self.influence_epsilon = influence_epsilon
        self.cache_file = cache_file
        
        self.num_landmarks = len(landmark_positions)
        self.influence_weights = None
        self.displacement_limits = None
        
        # Compute or load weights
        if cache_file and os.path.exists(cache_file):
            self._load_weights()
        else:
            self._compute_influence_weights()
            if cache_file:
                self._save_weights()
    
    def _compute_influence_weights(self):
        """
        Compute influence weight for each landmark based on:
        1. Number of mesh vertices within influence radius
        2. Distance to nearest mesh vertices
        3. Local mesh density around the landmark
        """
        print("[LandmarkInfluenceWeights] Computing influence weights...")
        
        # Build KD-tree for mesh vertices
        mesh_tree = cKDTree(self.mesh_vertex_positions)
        
        # Initialize weights
        self.influence_weights = np.zeros(self.num_landmarks)
        
        # Compute influence radius based on average mesh edge length
        avg_spacing = self._estimate_mesh_spacing()
        influence_radius = avg_spacing * 3  # Influence extends 3x average spacing
        
        for i, lm_pos in enumerate(self.landmark_positions):
            # Count vertices within influence radius
            neighbors = mesh_tree.query_ball_point(lm_pos, influence_radius)
            neighbor_count = len(neighbors)
            
            # Get distance to closest mesh vertices
            distances, _ = mesh_tree.query(lm_pos, k=min(5, len(self.mesh_vertex_positions)))
            avg_distance = np.mean(distances) if len(distances) > 0 else float('inf')
            
            # Influence = neighbor_count / (1 + avg_distance)
            # Higher neighbor count + smaller distance = higher influence
            self.influence_weights[i] = neighbor_count / (1 + avg_distance)
        
        # Normalize weights to [0, 1]
        max_weight = np.max(self.influence_weights)
        if max_weight > 0:
            self.influence_weights /= max_weight
        
        # Compute per-landmark displacement limits
        # Lower influence -> more restrictive limit
        self._compute_displacement_limits()
        
        print(f"[LandmarkInfluenceWeights] Computed {self.num_landmarks} weights")
        print(f"  Min influence: {np.min(self.influence_weights):.4f}")
        print(f"  Max influence: {np.max(self.influence_weights):.4f}")
        print(f"  Mean influence: {np.mean(self.influence_weights):.4f}")
    
    def _estimate_mesh_spacing(self) -> float:
        """Estimate average spacing between mesh vertices."""
        mesh_tree = cKDTree(self.mesh_vertex_positions)
        # Get distance to nearest neighbor for each vertex
        distances, _ = mesh_tree.query(self.mesh_vertex_positions, k=2)
        # Second column is distance to nearest neighbor (first is self=0)
        return np.mean(distances[:, 1])
    
    def _compute_displacement_limits(self):
        """
        Compute per-landmark displacement limits based on influence weights.
        
        Low influence landmarks get smaller displacement limits to prevent
        excessive movement that doesn't affect the mesh.
        """
        # Scale factor: low influence -> small limit, high influence -> base limit
        # Using: limit = base_limit * (influence + epsilon)
        self.displacement_limits = self.base_displacement_limit * (
            self.influence_weights + self.influence_epsilon
        )
        
        print(f"[LandmarkInfluenceWeights] Displacement limits:")
        print(f"  Min limit: {np.min(self.displacement_limits):.6f}")
        print(f"  Max limit: {np.max(self.displacement_limits):.6f}")
    
    def _save_weights(self):
        """Save computed weights to cache file."""
        if self.cache_file:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'influence_weights': self.influence_weights,
                    'displacement_limits': self.displacement_limits,
                }, f)
            print(f"[LandmarkInfluenceWeights] Saved to {self.cache_file}")
    
    def _load_weights(self):
        """Load weights from cache file."""
        with open(self.cache_file, 'rb') as f:
            data = pickle.load(f)
            self.influence_weights = data['influence_weights']
            self.displacement_limits = data['displacement_limits']
        print(f"[LandmarkInfluenceWeights] Loaded from {self.cache_file}")
    
    def clamp_displacement(self, 
                           delta_b0_conca: np.ndarray, 
                           landmark_init_positions: np.ndarray) -> np.ndarray:
        """
        Apply influence-weighted clamping to landmark displacements.
        
        Args:
            delta_b0_conca: Flat array of landmark displacements [x1,y1,z1,x2,y2,z2,...]
            landmark_init_positions: Initial landmark positions (flat or Nx3)
            
        Returns:
            Clamped delta_b0_conca array
        """
        delta_clamped = np.array(delta_b0_conca, dtype=np.float64).copy()
        
        for i in range(self.num_landmarks):
            # Get the 3D displacement for this landmark
            idx_start = i * 3
            idx_end = idx_start + 3
            displacement = delta_clamped[idx_start:idx_end]
            
            # Compute displacement magnitude
            magnitude = np.linalg.norm(displacement)
            
            # Get limit for this landmark
            limit = self.displacement_limits[i]
            
            # Clamp if exceeds limit
            if magnitude > limit:
                scale_factor = limit / magnitude
                delta_clamped[idx_start:idx_end] = displacement * scale_factor
        
        return delta_clamped
    
    def get_low_influence_landmarks(self, threshold: float = 0.1) -> List[int]:
        """
        Get indices of landmarks with influence below threshold.
        
        Args:
            threshold: Influence threshold (0-1)
            
        Returns:
            List of landmark indices with low influence
        """
        return np.where(self.influence_weights < threshold)[0].tolist()
    
    def visualize_influence_weights(self, output_path: str = None):
        """
        Create a visualization of influence weights.
        
        Args:
            output_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D scatter plot with influence as color
        ax1 = fig.add_subplot(121, projection='3d')
        scatter = ax1.scatter(
            self.landmark_positions[:, 0],
            self.landmark_positions[:, 1],
            self.landmark_positions[:, 2],
            c=self.influence_weights,
            cmap='RdYlGn',
            s=50
        )
        plt.colorbar(scatter, ax=ax1, label='Influence Weight')
        ax1.set_title('Landmark Influence Weights (3D)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Histogram of influence weights
        ax2 = fig.add_subplot(122)
        ax2.hist(self.influence_weights, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0.1, color='r', linestyle='--', label='Low influence threshold')
        ax2.set_xlabel('Influence Weight')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Influence Weights')
        ax2.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[LandmarkInfluenceWeights] Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()


def create_landmark_influence_weights(
    vertex_index: List[int],
    mesh_vertices: np.ndarray,
    landmark_positions: np.ndarray,
    base_displacement_limit: float = 0.01,
    cache_dir: Optional[str] = None
) -> LandmarkInfluenceWeights:
    """
    Factory function to create LandmarkInfluenceWeights instance.
    
    Args:
        vertex_index: List of mesh vertex indices corresponding to landmarks
        mesh_vertices: Full mesh vertex positions (Mx3)
        landmark_positions: Current landmark positions (Nx3)
        base_displacement_limit: Maximum displacement for high-influence landmarks
        cache_dir: Directory to cache weights
        
    Returns:
        LandmarkInfluenceWeights instance
    """
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'landmark_influence_weights.pkl')
    
    return LandmarkInfluenceWeights(
        landmark_positions=landmark_positions,
        mesh_vertex_positions=mesh_vertices,
        landmark_to_mesh_indices=vertex_index,
        base_displacement_limit=base_displacement_limit,
        cache_file=cache_file
    )


# Example integration code (to be added to off_sampler.py or evaluator.py)
"""
# At initialization:
from utils.landmark_influence_weights import create_landmark_influence_weights

self.influence_weights = create_landmark_influence_weights(
    vertex_index=self.vertex_index,
    mesh_vertices=mesh_vertices,  # Full mesh vertices
    landmark_positions=x_joint_positions,  # Initial landmark positions
    base_displacement_limit=UNIT_PRIME,
    cache_dir=self.save_folder
)

# When applying displacement:
# Before: x_joint_ = self.x_joint_init + delta_b0_conca[0:self.n]
# After:
clamped_delta = self.influence_weights.clamp_displacement(
    delta_b0_conca[0:self.n], 
    self.x_joint_init
)
x_joint_ = self.x_joint_init + clamped_delta
"""
