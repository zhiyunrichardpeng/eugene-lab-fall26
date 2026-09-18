import maya.cmds as cmds

def select_head_joint_hierarchy():
    root_joint = "head"
    
    if not cmds.objExists(root_joint):
        cmds.warning(f"'{root_joint}' does not exist in the scene.")
        return
        
    # Get all descendants of 'head' that are specifically of type 'joint'
    descendants = cmds.listRelatives(root_joint, allDescendents=True, type="joint", fullPath=True) or []
    
    # Combine the root with its descendants
    joints_to_select = [root_joint] + descendants
    
    # Select them in the scene (replace=True clears any existing selection)
    cmds.select(joints_to_select, replace=True)
    
    print(f"Successfully selected {len(joints_to_select)} joints in the '{root_joint}' hierarchy.")

def keep_only_frames_0_and_14_on_selected():
    selection = cmds.ls(selection=True)
    
    if not selection:
        cmds.warning("Nothing is selected! Please select the joints first.")
        return
        
    deleted_count = 0
    for node in selection:
        # Check total keys before deletion
        total_keys_before = cmds.keyframe(node, query=True, keyframeCount=True) or 0
        
        if total_keys_before > 0:
            # Query all unique keyframe times on this node
            key_times = cmds.keyframe(node, query=True, timeChange=True) or []
            key_times = list(set(key_times))
            
            for t in key_times:
                # Keep only exactly frame 0 and frame 14
                if t != 0.0 and t != 14.0:
                    # Check how many keys are exactly at this time
                    count_at_t = cmds.keyframe(node, time=(t, t), query=True, keyframeCount=True) or 0
                    if count_at_t > 0:
                        cmds.cutKey(node, time=(t, t), clear=True)
                        deleted_count += count_at_t
        
    print(f"Successfully deleted {deleted_count} keyframes from {len(selection)} selected objects, keeping only frames 0 and 14.")


if __name__ == "__main__":
    # 1. Select the hierarchy
    select_head_joint_hierarchy()
    
    # 2. Keep only frames 0 and 14 on the selected hierarchy
    keep_only_frames_0_and_14_on_selected()
