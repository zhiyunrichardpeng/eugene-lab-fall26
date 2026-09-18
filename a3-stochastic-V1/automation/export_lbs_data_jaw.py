import maya.cmds as cmds
import maya.mel as mel
import json
import os
import time

def export_lbs_data():
    """
    Exports the bind pose and skin weights for a given mesh and landmarks.
    Run this script inside Maya's Script Editor when the character is in its Neutral/Bind Pose.
    """
    mesh_name = "middleres_Jaw"
    output_dir = r"C:\Users\zpa18\a3-stochastic-V1\automation"
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Starting LBS Export for {mesh_name} ---")
    start_time = time.time()

    # 1. Define jaw joints
    # Note: If there are more jaw joints (jaw02, jaw03), add them to this list.
    jaw_joints = ["jaw01"]

    # 2. Find the Skin Cluster
    # findRelatedSkinCluster is a MEL script that reliably gets the skin attached to a mesh
    skin_cluster = mel.eval(f'findRelatedSkinCluster("{mesh_name}")')
    if not skin_cluster:
        cmds.error(f"No skin cluster found on {mesh_name}! Is it bound?")
        return
    print(f"Found Skin Cluster: {skin_cluster}")

    # 3. Map joint names to their internal Maya index
    influences = cmds.skinCluster(skin_cluster, query=True, influence=True)
    
    inf_indices = []
    missing_bones = 0
    for name in jaw_joints:
        if name in influences:
            inf_indices.append(influences.index(name))
        else:
            inf_indices.append(-1)
            missing_bones += 1
            print(f"WARNING: {name} is not influencing the mesh!")
    
    if missing_bones > 0:
        print(f"Warning: {missing_bones}/{len(jaw_joints)} landmarks are missing from the skin cluster.")

    # 4. Extract Vertices and Weights
    vtx_count = cmds.polyEvaluate(mesh_name, vertex=True)
    bind_pose = []
    skin_weights = []
    
    print(f"Extracting data for {vtx_count} vertices... Please wait.")
    
    for i in range(vtx_count):
        vtx = f"{mesh_name}.vtx[{i}]"
        
        # Get World Space Position (Ensure character is in rest pose!)
        pos = cmds.xform(vtx, query=True, worldSpace=True, translation=True)
        bind_pose.append(pos)
        
        # Get all weights for this vertex in one fast API call
        all_weights = cmds.skinPercent(skin_cluster, vtx, query=True, value=True)
        
        # Re-order the weights to strictly match our jaw_joints list
        vertex_weights = []
        for idx in inf_indices:
            if idx != -1:
                vertex_weights.append(all_weights[idx])
            else:
                vertex_weights.append(0.0) # Bone missing, weight is 0
                
        skin_weights.append(vertex_weights)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{vtx_count} vertices...")

    # 5. Extract Joint Pivots
    # This is required for the new PyTorch analytical gradient rotation math
    joint_pivots = []
    for joint in jaw_joints:
        if cmds.objExists(joint):
            # The world space translation of the joint acts as its pivot point
            pivot = cmds.xform(joint, query=True, worldSpace=True, translation=True)
            joint_pivots.append(pivot)
        else:
            print(f"WARNING: Joint {joint} does not exist. Using (0,0,0) as pivot.")
            joint_pivots.append([0.0, 0.0, 0.0])

    # 6. Save to Disk as JSON
    bind_pose_file = os.path.join(output_dir, "jaw_bind_pose.json")
    skin_weights_file = os.path.join(output_dir, "jaw_skin_weights.json")
    joint_pivots_file = os.path.join(output_dir, "jaw_joint_pivots.json")
    
    print("Saving files to disk...")
    with open(bind_pose_file, 'w') as f:
        json.dump(bind_pose, f)
        
    with open(skin_weights_file, 'w') as f:
        json.dump(skin_weights, f)

    with open(joint_pivots_file, 'w') as f:
        json.dump(joint_pivots, f)
        
    end_time = time.time()
    print("=== Export Complete! ===")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Bind Pose Matrix: {vtx_count} x 3 -> {bind_pose_file}")
    print(f"Skin Weights Matrix: {vtx_count} x {len(jaw_joints)} -> {skin_weights_file}")
    print(f"Joint Pivots Matrix: {len(jaw_joints)} x 3 -> {joint_pivots_file}")

if __name__ == "__main__":
    export_lbs_data()
