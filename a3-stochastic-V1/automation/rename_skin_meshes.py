import maya.cmds as cmds

def cleanup_mesh_names():
    """
    Renames auto-generated mesh names (e.g., polySurfaceXX) to more user-friendly 
    names based on their parent groups.
    """
    # The original list of objects to clean up
    mesh_list = [
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface31",
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface28",
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface30",
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface35",
        "polySurface36",
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface37",
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface32",
        "polySurface23",
        "jack01_MODEL1|jack01_c_loTeeth01_geo",
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface33",
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface29",
        "jack01_MODEL1|jack01_c_head01_high_geo_shp|polySurface19",
        "jack01_MODEL1|jack01_c_eyeOuter01_geo|polySurface25|polySurface25",
        "jack01_MODEL1|jack01_c_head01_high_geo_shp|polySurface20",
        "|eye_r",
        "jack01_MODEL1|jack01_c_tongue01_geo",
        "jack01_MODEL1|jack01_c_tearDuct01_geo|polySurface22",
        "jack01_MODEL1|jack01_c_upTeeth01_geo",
        "jack01_MODEL1|jack01_c_head01_high_geo_shp|polySurface21",
        "jack01_MODEL1|jack01_c_tear01_geo|polySurface34",
        "skin_cloth_copy_v5_pull_back_RENDER"
    ]

    # Counters to give each renamed part a unique number
    counters = {
        "tear": 1,
        "head": 1,
        "eyeOuter": 1,
        "tearDuct": 1,
        "misc": 1
    }

    # Optional: Group them together if they are top level, but we will just rename here to keep hierarchies intact.
    
    print("--- Starting Mesh Housekeeping ---")
    for mesh_path in mesh_list:
        short_name = mesh_path.split('|')[-1]
        parent_name = mesh_path.split('|')[-2] if '|' in mesh_path else ""
        
        # Build a flexible search pattern (e.g. *tear01_geo*|*polySurface31*)
        if parent_name:
            search_pattern = "*{0}*|*{1}*".format(parent_name, short_name)
        else:
            search_pattern = "*{0}*".format(short_name)
            
        found = cmds.ls(search_pattern)
        if not found:
            # Fallback: just look for the short name anywhere
            found = cmds.ls("*{0}*".format(short_name))
            
        if not found:
            cmds.warning("Object not found in scene, skipping: {}".format(mesh_path))
            continue
            
        actual_path = found[0]
        
        # If it's a generated name like 'polySurface', we give it a proper name
        if "polySurface" in short_name:
            if "tear01_geo" in parent_name:
                new_name = "jack01_c_tear01_geo_part_{:02d}".format(counters["tear"])
                counters["tear"] += 1
            elif "head01_high" in parent_name:
                new_name = "jack01_c_head01_high_geo_part_{:02d}".format(counters["head"])
                counters["head"] += 1
            elif "eyeOuter" in parent_name:
                new_name = "jack01_c_eyeOuter01_geo_part_{:02d}".format(counters["eyeOuter"])
                counters["eyeOuter"] += 1
            elif "tearDuct" in parent_name:
                new_name = "jack01_c_tearDuct01_geo_part_{:02d}".format(counters["tearDuct"])
                counters["tearDuct"] += 1
            else:
                new_name = "jack01_render_misc_part_{:02d}".format(counters["misc"])
                counters["misc"] += 1
                
            # Rename the object
            final_name = cmds.rename(actual_path, new_name)
            print("Renamed: {}  ->  {}".format(short_name, final_name))
        else:
            # It already has a somewhat proper name, we can leave it alone
            print("Skipped (already named well): {}".format(short_name))
            
    print("--- Housekeeping Complete ---")

if __name__ == "__main__":
    cleanup_mesh_names()
