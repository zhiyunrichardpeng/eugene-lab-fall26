import maya.cmds as cmds
import maya.mel as mel
import json
import os
import time

def generate_landmark_names(prefix, count=300):
    names = []
    for i in range(1, count + 1):
        if i < 10:
            names.append(f"{prefix}_0{i}")
        else:
            names.append(f"{prefix}_{i}")
    return names

def export_component_data(mesh_name, prefix, influences, output_dir, extract_pivots=False):
    print(f"--- Starting LBS Export for {mesh_name} ({prefix}) ---")
    start_time = time.time()

    skin_cluster = mel.eval(f'findRelatedSkinCluster("{mesh_name}")')
    if not skin_cluster:
        print(f"ERROR: No skin cluster found on {mesh_name}! Is it bound? Skipping...")
        return
    print(f"Found Skin Cluster: {skin_cluster}")

    cluster_influences = cmds.skinCluster(skin_cluster, query=True, influence=True)
    
    inf_indices = []
    missing_bones = 0
    for name in influences:
        if name in cluster_influences:
            inf_indices.append(cluster_influences.index(name))
        else:
            inf_indices.append(-1)
            missing_bones += 1
            print(f"WARNING: {name} is not influencing the mesh!")
    
    if missing_bones > 0:
        print(f"Warning: {missing_bones}/{len(influences)} landmarks are missing from the skin cluster.")

    vtx_count = cmds.polyEvaluate(mesh_name, vertex=True)
    if isinstance(vtx_count, list) or isinstance(vtx_count, tuple):
        vtx_count = vtx_count[0]

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
        
        # Re-order the weights to strictly match our list
        vertex_weights = []
        for idx in inf_indices:
            if idx != -1:
                vertex_weights.append(all_weights[idx])
            else:
                vertex_weights.append(0.0) # Bone missing, weight is 0
                
        skin_weights.append(vertex_weights)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{vtx_count} vertices...")

    bind_pose_file = os.path.join(output_dir, f"{prefix}_bind_pose.json")
    skin_weights_file = os.path.join(output_dir, f"{prefix}_skin_weights.json")
    
    print("Saving files to disk...")
    with open(bind_pose_file, 'w') as f:
        json.dump(bind_pose, f)
        
    with open(skin_weights_file, 'w') as f:
        json.dump(skin_weights, f)

    if extract_pivots:
        joint_pivots = []
        for joint in influences:
            if cmds.objExists(joint):
                pivot = cmds.xform(joint, query=True, worldSpace=True, translation=True)
                joint_pivots.append(pivot)
            else:
                print(f"WARNING: Joint {joint} does not exist. Using (0,0,0) as pivot.")
                joint_pivots.append([0.0, 0.0, 0.0])
                
        joint_pivots_file = os.path.join(output_dir, f"{prefix}_joint_pivots.json")
        with open(joint_pivots_file, 'w') as f:
            json.dump(joint_pivots, f)
        print(f"Joint Pivots Matrix: {len(influences)} x 3 -> {joint_pivots_file}")
        
    end_time = time.time()
    print(f"=== Export Complete for {mesh_name}! ===")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Bind Pose Matrix: {vtx_count} x 3 -> {bind_pose_file}")
    print(f"Skin Weights Matrix: {vtx_count} x {len(influences)} -> {skin_weights_file}\n")


def export_all_lbs_data():
    """
    Exports the bind pose, skin weights, and joint pivots for multiple meshes.
    Run this script inside Maya's Script Editor when the character is in its Neutral/Bind Pose.
    """
    output_dir = r"C:\Users\zpa18\a3-stochastic-V1\automation"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    components = [
        {
            "mesh_name": "middleres_OrbicularisOris",
            "prefix": "oor",
            "influences": generate_landmark_names("oor"),
            "extract_pivots": False
        },
        {
            "mesh_name": "middleres_Jaw",
            "prefix": "jaw",
            "influences": ["jaw01"],
            "extract_pivots": True
        },
        {
            "mesh_name": "middleres_OrbicularisOculi_l",
            "prefix": "ooc_l",
            "influences": generate_landmark_names("ooc_l"),
            "extract_pivots": False
        },
        {
            "mesh_name": "middleres_OrbicularisOculi_r",
            "prefix": "ooc_r",
            "influences": generate_landmark_names("ooc_r"),
            "extract_pivots": False
        },
        {
            "mesh_name": "middleres_Frontalis",
            "prefix": "frontalis",
            "influences": generate_landmark_names("frontalis"),
            "extract_pivots": False
        }
    ]

    for comp in components:
        export_component_data(
            mesh_name=comp["mesh_name"],
            prefix=comp["prefix"],
            influences=comp["influences"],
            output_dir=output_dir,
            extract_pivots=comp["extract_pivots"]
        )

if __name__ == "__main__":
    export_all_lbs_data()
