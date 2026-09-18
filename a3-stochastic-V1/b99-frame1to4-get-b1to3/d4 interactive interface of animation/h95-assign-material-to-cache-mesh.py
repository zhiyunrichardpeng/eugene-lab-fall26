# h95-assign-material-to-cache-mesh.py
# Run this script in Maya Script Editor (Python tab)
#
# Purpose: Assign the original mesh materials/colors to the cached (new) meshes
#          so that the cache playback has the correct muscle colors.
#
# Naming pattern:
#   - Old hierarchical meshes (parent|child) -> New mesh is root-level: |child
#   - Old standalone meshes (middleres_X) -> New mesh: middleres_X1

import maya.cmds as cmds


# ============================================================
# MESH MAPPING: Old (original) -> New (cached)
# ============================================================
# The mapping is built from the two select commands.
# Pattern: if old has "|", the new mesh is just the leaf name at root.
#          if old has no "|", the new mesh appends "1".

MESH_MAPPING = {
    # --- Hierarchical meshes (parent|child -> |child) ---
    "middleres_Node|polySurface4":                                           "|polySurface4",
    "middleres_Node|polySurface3":                                           "|polySurface3",
    "middleres_DepresorAnguliOris|polySurface12":                            "|polySurface12",
    "middleres_DepresorAnguliOris|polySurface11":                            "|polySurface11",
    "middleres_Buccinator|polySurface2":                                     "|polySurface2",
    "middleres_Buccinator|polySurface1":                                     "|polySurface1",
    "middleres_Risorius|polySurface7":                                       "|polySurface7",
    "middleres_Risorius|polySurface8":                                       "|polySurface8",
    "middleres_MasseterSuperior|polySurface5":                               "|polySurface5",
    "middleres_MasseterSuperior|polySurface6":                               "|polySurface6",
    "middleres_PlatysmaNodularPortion|polySurface9":                         "|polySurface9",
    "middleres_PlatysmaNodularPortion|polySurface10":                        "|polySurface10",
    "middleres_LevatorLabiiSuperiorisAlaequeNasi|polySurface13":             "|polySurface13",
    "middleres_LevatorLabiiSuperiorisAlaequeNasi|polySurface14":             "|polySurface14",
    "middleres_ZygomaticusMinor|middleres_ZygomaticusMinor_r":               "|middleres_ZygomaticusMinor_r",
    "middleres_ZygomaticusMinor|middleres_ZygomaticusMinor_l":               "|middleres_ZygomaticusMinor_l",
    "middleres_ZygomaticusMajor|middleres_ZygomaticusMajor_l":               "|middleres_ZygomaticusMajor_l",
    "middleres_ZygomaticusMajor|middleres_ZygomaticusMajor_r":               "|middleres_ZygomaticusMajor_r",
    "middleres_DepressorSupercilii|middleres_DepressorSupercilii_l":         "|middleres_DepressorSupercilii_l",
    "middleres_DepressorSupercilii|middleres_DepressorSupercilii_r":         "|middleres_DepressorSupercilii_r",
    "middleres_DepressorLabiiInferioris|middleres_DepressorLabiiInferioris_l":"|middleres_DepressorLabiiInferioris_l",
    "middleres_DepressorLabiiInferioris|middleres_DepressorLabiiInferioris_r":"|middleres_DepressorLabiiInferioris_r",
    "middleres_LevatorLabiiSuperioris|middleres_LevatorLabiiSuperioris_l":   "|middleres_LevatorLabiiSuperioris_l",
    "middleres_LevatorLabiiSuperioris|middleres_LevatorLabiiSuperioris_r":   "|middleres_LevatorLabiiSuperioris_r",
    "middleres_OrbicularisOculi|middleres_OrbicularisOculi_r":              "|middleres_OrbicularisOculi_r",
    "middleres_OrbicularisOculi|middleres_OrbicularisOculi_l":              "|middleres_OrbicularisOculi_l",
    "middleres_Temporalis|middleres_Temporalis_l":                           "|middleres_Temporalis_l",
    "middleres_Temporalis|middleres_Temporalis_r":                           "|middleres_Temporalis_r",

    # --- Standalone meshes (name -> name1) ---
    "middleres_Mentalis":              "middleres_Mentalis1",
    "middleres_Procerus":              "middleres_Procerus1",
    "middleres_AuricularisSuperior":   "middleres_AuricularisSuperior1",
    "middleres_Frontalis":             "middleres_Frontalis1",
    "middleres_NasalisTransverse":     "middleres_NasalisTransverse1",
    "middleres_AlarCartidge":          "middleres_AlarCartidge1",
    "middleres_TeethDown":             "middleres_TeethDown1",
    "middleres_TeethUp":               "middleres_TeethUp1",
    "middleres_Eyes_r_simple":         "middleres_Eyes_r_simple1",
    "middleres_Eyes_l_simple":         "middleres_Eyes_l_simple1",
    "middleres_skull":                 "middleres_skull1",
    "middleres_AuricularisPosterior":  "middleres_AuricularisPosterior1",
    "middleres_Jaw":                   "middleres_Jaw1",
    "middleres_LateralCartige":        "middleres_LateralCartige1",
    "middleres_OrbicularisOris":       "middleres_OrbicularisOris1",
    "middleres_NostrilWing":           "middleres_NostrilWing1",
    "middleres_MasseterDeep":          "middleres_MasseterDeep1",
    "middleres_AuricularisAnterior":   "middleres_AuricularisAnterior1",
    "middleres_DepressorSeptiNasi":    "middleres_DepressorSeptiNasi1",
}


def get_shading_engines(mesh_name):
    """
    Get all shading engines (materials) assigned to a mesh.
    Returns a list of shading engine names.
    """
    # Get the shape node(s)
    shapes = cmds.listRelatives(mesh_name, shapes=True, fullPath=True)
    if not shapes:
        return []

    all_sgs = []
    for shape in shapes:
        sgs = cmds.listConnections(shape, type='shadingEngine')
        if sgs:
            all_sgs.extend(sgs)

    # Remove duplicates while preserving order
    seen = set()
    unique_sgs = []
    for sg in all_sgs:
        if sg not in seen:
            seen.add(sg)
            unique_sgs.append(sg)

    return unique_sgs


def transfer_material(old_mesh, new_mesh):
    """
    Transfer material assignments from old_mesh to new_mesh.
    Handles both uniform and per-face material assignments.
    """
    # --- Validate meshes exist ---
    if not cmds.objExists(old_mesh):
        print(f"  [SKIP] Old mesh not found in scene: {old_mesh}")
        return False

    if not cmds.objExists(new_mesh):
        print(f"  [SKIP] New mesh not found in scene: {new_mesh}")
        return False

    # --- Get shading engines from old mesh ---
    shading_engines = get_shading_engines(old_mesh)

    if not shading_engines:
        print(f"  [WARN] No materials found on old mesh: {old_mesh}")
        return False

    if len(shading_engines) == 1:
        # Simple case: one material for the whole mesh
        sg = shading_engines[0]
        cmds.sets(new_mesh, edit=True, forceElement=sg)
        mat = cmds.listConnections(sg + '.surfaceShader')
        mat_name = mat[0] if mat else sg
        print(f"  [OK] {new_mesh} <- {mat_name} (from {old_mesh})")
        return True
    else:
        # Complex case: multiple materials (per-face assignments)
        print(f"  [MULTI-MAT] {old_mesh} has {len(shading_engines)} materials")
        old_leaf = old_mesh.split('|')[-1]
        new_leaf = new_mesh.lstrip('|')

        assigned_any = False
        for sg in shading_engines:
            members = cmds.sets(sg, query=True) or []
            # Find members belonging to the old mesh
            old_members = [m for m in members if old_leaf in m]

            if not old_members:
                continue

            for old_member in old_members:
                if '.f[' in old_member:
                    # Per-face assignment: replace old mesh name with new
                    new_member = old_member.replace(old_leaf, new_leaf)
                    try:
                        cmds.sets(new_member, edit=True, forceElement=sg)
                        assigned_any = True
                    except Exception as e:
                        print(f"    [ERR] Failed per-face assign {new_member}: {e}")
                else:
                    # Whole mesh assignment
                    try:
                        cmds.sets(new_mesh, edit=True, forceElement=sg)
                        assigned_any = True
                    except Exception as e:
                        print(f"    [ERR] Failed assign {new_mesh}: {e}")

            mat = cmds.listConnections(sg + '.surfaceShader')
            mat_name = mat[0] if mat else sg
            print(f"    Assigned {mat_name} to matching faces")

        if not assigned_any:
            # Fallback: just assign the first shading engine to the whole mesh
            cmds.sets(new_mesh, edit=True, forceElement=shading_engines[0])
            print(f"  [FALLBACK] Assigned {shading_engines[0]} to entire {new_mesh}")

        return True


def run():
    """Main function: transfer all materials from old meshes to new cached meshes."""
    print("\n" + "=" * 70)
    print("  MATERIAL TRANSFER: Original Meshes -> Cached Meshes")
    print("=" * 70 + "\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for old_mesh, new_mesh in MESH_MAPPING.items():
        try:
            result = transfer_material(old_mesh, new_mesh)
            if result:
                success_count += 1
            else:
                skip_count += 1
        except Exception as e:
            print(f"  [ERROR] {old_mesh} -> {new_mesh}: {e}")
            fail_count += 1

    print(f"\n{'=' * 70}")
    print(f"  DONE: {success_count} assigned, {skip_count} skipped, {fail_count} failed")
    print(f"  Total mesh pairs: {len(MESH_MAPPING)}")
    print(f"{'=' * 70}\n")


# --- RUN ---
run()
