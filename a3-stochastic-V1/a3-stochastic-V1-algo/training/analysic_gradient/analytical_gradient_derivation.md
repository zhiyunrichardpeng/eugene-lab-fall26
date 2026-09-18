# Analytical Gradient for Mesh Deformation in Maya (Linear Blend Skinning)

## 1. Understanding $g(\cdot)$
In Autodesk Maya, the primary algorithm used to map the movement of landmarks (joints/bones/controls) to the displacement of mesh vertices is **Linear Blend Skinning (LBS)**. 

When you move or rotate a landmark, Maya calculates the new position of each vertex based on an "influence strength" (skin weights). 

By reproducing the LBS algorithm outside of Maya (e.g., in Python using PyTorch or NumPy), we change $g(\cdot)$ from a "black box" into a fully differentiable mathematical function. This allows us to compute **exact analytical gradients** in milliseconds, completely replacing the slow finite-difference (numerical) method.

---

## 2. The LBS Mathematical Model

Let's define the components:
*   $N$: Number of vertices in the mesh.
*   $K$: Number of landmarks (joints).
*   $\mathbf{v}_i \in \mathbb{R}^3$: The original (bind pose) position of vertex $i$.
*   $w_{i,j}$: The skin weight (influence) of landmark $j$ on vertex $i$.
*   $\mathbf{t}_j \in \mathbb{R}^3$: The translation applied to landmark $j$.
*   $\mathbf{R}_j \in \mathbb{R}^{3 \times 3}$: The rotation matrix applied to landmark $j$.

The new position of vertex $i$, denoted as $\mathbf{y}_i(e)$, is calculated as a weighted sum of the transformations applied by all landmarks:

$$ \mathbf{y}_i(e) = \sum_{j=1}^K w_{i,j} \left( \mathbf{R}_j \mathbf{v}_i + \mathbf{t}_j \right) $$

*(Note: In practice, this is often done using $4 \times 4$ transformation matrices and homogeneous coordinates, but the vector form above is mathematically equivalent and easier to differentiate).*

---

## 3. Deriving the Analytical Gradient

Your current reward function is the negative squared Frobenius norm (Euclidean distance) between the target vertices $\mathbf{x}$ and the current vertices $\mathbf{y}(e)$:

$$ r_{fw}(e) = - \sum_{i=1}^N \parallel \mathbf{x}_i - \mathbf{y}_i(e) \parallel_2^2 $$

### Gradient with respect to Translation
If your optimization parameter $e$ is the translation $\mathbf{t}_k$ of landmark $k$ in the $x, y,$ or $z$ direction, the partial derivative of the vertex position is extremely simple:

$$ \frac{\partial \mathbf{y}_i}{\partial \mathbf{t}_k} = w_{i,k} \mathbf{I}_{3 \times 3} $$

Applying the chain rule to the reward function gives the analytical gradient for the translation of landmark $k$:

$$ \frac{\partial r}{\partial \mathbf{t}_k} = \sum_{i=1}^N 2 w_{i,k} \left( \mathbf{x}_i - \mathbf{y}_i(e) \right) $$

**What this means:** You don't need to perturb the landmarks ($\pm \delta$) and wait for Maya to compute the mesh twice per landmark. You just multiply the error vector $(\mathbf{x}_i - \mathbf{y}_i(e))$ by the skin weight $w_{i,k}$ for each vertex and sum them up.

### Gradient with respect to Rotation
If your parameters involve rotations (e.g., Euler angles $\theta_k$), the derivative is taken with respect to the rotation matrix:

$$ \frac{\partial \mathbf{y}_i}{\partial \theta_k} = w_{i,k} \frac{\partial \mathbf{R}_k}{\partial \theta_k} \mathbf{v}_i $$

$$ \frac{\partial r}{\partial \theta_k} = \sum_{i=1}^N 2 \left( \mathbf{x}_i - \mathbf{y}_i(e) \right)^T \left( w_{i,k} \frac{\partial \mathbf{R}_k}{\partial \theta_k} \mathbf{v}_i \right) $$

---

## 4. How to Realize This in Your Pipeline

To bypass Maya's black box and compute this instantly, you need to follow these steps:

### Step 1: Export Static Data from Maya (One-time setup)
Before optimization begins, run a Maya script to extract the static data and save it (e.g., as JSON, NumPy `.npy`, or HDF5):
1.  **Bind Pose Vertices ($\mathbf{v}$):** The rest position of all vertices (an $N \times 3$ array).
2.  **Skin Weights ($\mathbf{W}$):** The influence of every landmark on every vertex (an $N \times K$ array). You can query this using `maya.cmds.skinPercent`.

### Step 2: Forward and Backward Pass in Python (PyTorch)
Instead of writing the gradient equations manually, you can use a modern automatic differentiation framework like **PyTorch**. 
1. Load the $\mathbf{v}$ and $\mathbf{W}$ arrays into PyTorch tensors.
2. Define the landmark translations and rotations as `requires_grad=True`.
3. Compute the LBS equation (Forward pass).
4. Compute the reward.
5. Call `reward.backward()` (Backward pass). PyTorch will instantly compute the exact analytical gradients using the formulas derived above.

By moving $g(\cdot)$ to PyTorch/NumPy, your optimization step will go from taking seconds/minutes (waiting for Maya API calls and mesh updates) to taking **milliseconds** on the CPU or GPU.
