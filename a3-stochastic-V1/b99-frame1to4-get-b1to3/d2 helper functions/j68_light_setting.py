import maya.cmds as cmds

def create_three_point_rig():
    selection = cmds.ls(sl=True)
    if not selection:
        cmds.warning("Please select your head mesh first so I know where to aim the lights!")
        return

    target = selection[0]
    
    # Calculate the size of your head mesh to scale the lights dynamically
    bbox = cmds.exactWorldBoundingBox(target)
    size = max(bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2])
    center = [(bbox[0]+bbox[3])/2, (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2]

    # Create a locator for the lights to constantly look at
    aim_target = cmds.spaceLocator(name="Light_Aim_Target")[0]
    cmds.xform(aim_target, translation=center, worldSpace=True)
    # Hide the locator so it doesn't render
    cmds.setAttr(f"{aim_target}.visibility", 0) 

    # Create a clean group for the Outliner
    lights_group = cmds.group(empty=True, name="Arnold_ThreePoint_Rig")

    def make_area_light(name, pos_offset, exposure, color):
        # Create an Area Light (Arnold handles these perfectly)
        light_transform = cmds.shadingNode('areaLight', asLight=True, name=name)
        light_shape = cmds.listRelatives(light_transform, shapes=True)[0]
        
        # Position the light relative to the head's size
        cmds.xform(light_transform, translation=[center[0] + pos_offset[0]*size, 
                                                 center[1] + pos_offset[1]*size, 
                                                 center[2] + pos_offset[2]*size], worldSpace=True)
        
        # Automatically aim the light at the center locator
        cmds.aimConstraint(aim_target, light_transform, offset=[0,0,0], weight=1, aimVector=[0,0,-1], upVector=[0,1,0])
        
        # Set physical scale and Arnold Exposure
        cmds.setAttr(f"{light_transform}.scaleX", size * 0.8)
        cmds.setAttr(f"{light_transform}.scaleY", size * 0.8)
        cmds.setAttr(f"{light_shape}.aiExposure", exposure)
        cmds.setAttr(f"{light_shape}.color", color[0], color[1], color[2], type="double3")
        
        cmds.parent(light_transform, lights_group)
        return light_transform

    # Matching the exact diagram you provided:
    # Key Light on the Left (-X), Fill Light on the Right (+X), Back Light on the Back-Left (-X, -Z)
    
    # 1. Key Light (Main light: placed front-left (-X, +Z))
    # Exposure 21.0 is the baseline you found looks good.
    make_area_light("Key_Light", [-1.5, 1.0, 1.5], 21.0, [1.0, 0.95, 0.9])

    # 2. Fill Light (Soft light: placed front-right (+X, +Z))
    # Fill light is usually 1.5 to 2 stops darker than Key to create soft shadows without erasing them.
    make_area_light("Fill_Light", [1.5, 0.2, 1.5], 19.5, [0.9, 0.95, 1.0])

    # 3. Rim Light (Back light: placed back-left (-X, -Z))
    # Rim light is usually 0.5 to 1 stop brighter than Key to create a sharp, glowing edge silhouette.
    make_area_light("Rim_Light", [-1.5, 1.5, -2.0], 22.0, [1.0, 1.0, 1.0])

    cmds.parent(aim_target, lights_group)
    cmds.select(lights_group)
    print("Three-Point Lighting Rig generated successfully!")

create_three_point_rig()