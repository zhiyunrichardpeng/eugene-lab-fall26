import maya.cmds as cmds

# Predefined list of 45 meshes provided by the user
PREDEFINED_MESHES = [
    "middleres_ZygomaticusMinor_l",
    "middleres_Corrugator_l",
    "middleres_DepressorLabiiInferioris_l",
    "middleres_OrbicularisOculi_r",
    "middleres_AuricularisSuperior",
    "middleres_Mentalis",
    "polySurface5",
    "polySurface6",
    "polySurface4",
    "middleres_OrbicularisOris",
    "middleres_LevatorLabiiSuperioris_r",
    "polySurface10",
    "polySurface11",
    "middleres_DepressorSeptiNasi",
    "middleres_MasseterDeep",
    "middleres_LateralCartige",
    "middleres_NostrilWing",
    "middleres_Frontalis",
    "polySurface13",
    "polySurface9",
    "middleres_ZygomaticusMajor_l",
    "middleres_LevatorAnguliOris",
    "middleres_OrbicularisOculi_l",
    "middleres_AlarCartidge",
    "polySurface14",
    "middleres_Procerus",
    "middleres_LevatorLabiiSuperioris_l",
    "polySurface8",
    "polySurface1",
    "middleres_Temporalis_l",
    "polySurface2",
    "polySurface7",
    "middleres_Corrugator_r",
    "middleres_DepressorLabiiInferioris_r",
    "middleres_ZygomaticusMinor_r",
    "middleres_PlatysmaLabialPortion",
    "middleres_DepressorSupercilii_r",
    "polySurface3",
    "middleres_NasalisTransverse",
    "polySurface12",
    "middleres_ZygomaticusMajor_r",
    "middleres_DepressorSupercilii_l",
    "middleres_Temporalis_r",
    "middleres_skull",
    "middleres_Jaw"
]

def safe_set_attr(node, attr_name, value):
    """
    Attempts to safely set an attribute on a node.
    If the attribute is locked, it unlocks it, sets the value, and locks it back.
    If it is connected, it skips it to avoid breaking dependency graph connections.
    """
    attr_path = "{}.{}".format(node, attr_name)
    if not cmds.attributeQuery(attr_name, node=node, exists=True):
        return False

    # Check if the attribute has an incoming connection
    if cmds.listConnections(attr_path, destination=False, source=True):
        print("Warning: Attribute '{}' has incoming connections and cannot be set directly. Skipping.".format(attr_path))
        return False

    is_locked = cmds.getAttr(attr_path, lock=True)
    try:
        if is_locked:
            cmds.setAttr(attr_path, lock=False)
        
        cmds.setAttr(attr_path, value)
        
        if is_locked:
            cmds.setAttr(attr_path, lock=True)
        return True
    except Exception as e:
        print("Warning: Failed to set '{}': {}".format(attr_path, e))
        # Restore lock if we failed after unlocking
        if is_locked:
            try:
                cmds.setAttr(attr_path, lock=True)
            except:
                pass
        return False

def disable_ziva_collisions(mesh_list=None):
    """
    Disables collisions and collisionVolume on Ziva nodes (zTissue, zBone, zCloth)
    associated with the specified meshes.
    
    If mesh_list is None, it uses the current Maya selection. If nothing is selected,
    it falls back to the predefined list of 45 meshes.
    """
    # 1. Determine target meshes (selection, parameter, or predefined fallback)
    if mesh_list is None:
        selection = cmds.ls(selection=True)
        if selection:
            mesh_list = selection
            print("Using current selection ({} objects).".format(len(mesh_list)))
        else:
            mesh_list = PREDEFINED_MESHES
            print("No meshes selected. Falling back to predefined list of {} meshes.".format(len(PREDEFINED_MESHES)))
    
    processed_nodes = set()
    mesh_count = 0
    
    print("\n--- Starting Ziva Collision Disabling ---")
    
    for mesh_item in mesh_list:
        # Search for the mesh (using exact name first, then partial matching for namespaces/groups)
        found = cmds.ls(mesh_item)
        if not found:
            found = cmds.ls("*" + mesh_item + "*")
            
        if not found:
            print("Warning: Mesh '{}' not found in scene. Skipping.".format(mesh_item))
            continue
            
        # Get shape nodes for the mesh (history is attached to shapes)
        mesh_node = found[0]
        shapes = cmds.listRelatives(mesh_node, shapes=True, fullPath=True) or [mesh_node]
        
        # Gather Ziva nodes connected to shape nodes
        ziva_nodes = []
        for shape in shapes:
            history = cmds.listHistory(shape) or []
            found_ziva = cmds.ls(history, type=['zTissue', 'zBone', 'zCloth'])
            if found_ziva:
                ziva_nodes.extend(found_ziva)
                
        # Make the list of Ziva nodes unique
        ziva_nodes = list(set(ziva_nodes))
        
        if not ziva_nodes:
            print("No Ziva nodes (zTissue, zBone, zCloth) found in the history of '{}'.".format(mesh_node))
            continue
            
        mesh_count += 1
        for z_node in ziva_nodes:
            if z_node in processed_nodes:
                continue
                
            # Disable collisions
            collisions_disabled = safe_set_attr(z_node, 'collisions', 0)
                
            # Disable collisionVolume
            volume_disabled = safe_set_attr(z_node, 'collisionVolume', 0)
                
            if collisions_disabled or volume_disabled:
                processed_nodes.add(z_node)
                print("Disabled collisions/volume on Ziva node: '{}' (associated with '{}')".format(z_node, mesh_node))
                
    print("\n--- Collision Disabling Complete ---")
    print("Processed {} meshes, updated {} Ziva nodes.".format(mesh_count, len(processed_nodes)))

if __name__ == "__main__":
    disable_ziva_collisions()
