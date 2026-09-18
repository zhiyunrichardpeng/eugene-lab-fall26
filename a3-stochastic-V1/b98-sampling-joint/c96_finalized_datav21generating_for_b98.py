def c96_finalized_datav21generating(keyframe_num):
    import numpy as np
    import maya.cmds as cmds
    import maya.OpenMaya as OpenMaya
    import os
    debug_mode_vsCode = 1   # 1 is running in vs code.  0 is in maya.
    # keyframe_num = 10
    keframe_max = 50
    start_frame = 14  # at frame 15, the gravity effect starts to converge.
    
    save_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub"
    # mayaFilePath = r"C:\keyLabLocal_D\c99-maya-ziva-workshop\e5-start-from-0-adhocV2.mb"
    write_to = "\dataV22_AU7.py"

    with open(save_folder + write_to, 'w') as f:
        f.write("''' \n" + "  \n  \n   \n" + "'''" + "\n"*3 +
                "#start" + '\n \n')

    # import os




    if debug_mode_vsCode == 1:
        import maya.standalone
        maya.standalone.initialize()    
        import maya.cmds as cmds
        import maya.OpenMaya as OpenMaya    

        import sys
        from PyQt5.QtCore import Qt, QCoreApplication
        from PyQt5.QtWidgets import QApplication, QLabel        
        
        # Set the attribute before creating the QApplication
        QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

        app = QApplication(sys.argv)

        # Verify the file path
        maya_file_to_open = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\keyLabCloud\c99-maya-ziva-workshop\experiment_files\g45 AU7 done.mb"
        if not os.path.exists(maya_file_to_open):
            raise RuntimeError(f"File not found: {maya_file_to_open}")

        # Check for unsaved changes and handle them
        if cmds.file(query=True, modified=True):
            # Save or discard changes as needed
            print("Unsaved changes detected.")
            # save_changes = input("You have unsaved changes. Do you want to save them before opening the new file? (yes/no): ").strip().lower()
            if 1==0:  # save_changes in ['yes', 'y']:
                cmds.file(save=True)
                print("Changes saved.")
            else:
                cmds.file(newFile=True, force=True)  # Discard changes

        # Open the Maya file
        try:
            opened_file = cmds.file(maya_file_to_open, open=True, force=True)
            print("Opened file:", opened_file)
        except Exception as e:
            print(f"Failed to open Maya file: {e}")

        
        solver = cmds.ziva(s=True)[0]
        ziva_nodes = cmds.ls(type='zSolverTransform')
        print(f"Ziva solver nodes: {ziva_nodes}")        




    def retrievePosition(object):
        cmds.select( object, visible=True )
        selection = OpenMaya.MSelectionList()
        OpenMaya.MGlobal.getActiveSelectionList( selection )
        iterSel = OpenMaya.MItSelectionList(selection, OpenMaya.MFn.kMesh)
        dagPath = OpenMaya.MDagPath()
        iterSel.getDagPath( dagPath )
        inMeshMPointArray = OpenMaya.MPointArray()
        currentInMeshMFnMesh = OpenMaya.MFnMesh(dagPath)
        currentInMeshMFnMesh.getPoints(inMeshMPointArray, OpenMaya.MSpace.kWorld)
        pointList = []
        vertices = []
        vertex_i = OpenMaya.MItMeshVertex(dagPath)
        for i in range( inMeshMPointArray.length() ) :
            pointList.append( [inMeshMPointArray[i][0], inMeshMPointArray[i][1], inMeshMPointArray[i][2]] )
        
        mesh_name = object
        while not vertex_i.isDone():
            vertices.append(mesh_name + '.vtx[' + str(vertex_i.index()) + ']')
            next(vertex_i)

        # print(vertices)
        # print(pointList)
        return pointList



    def getVertexPosition(i):
        objectList = ["middleres_Jaw", "middleres_OrbicularisOris", "middleres_OrbicularisOculi_l", "middleres_OrbicularisOculi_r", "middleres_Frontalis"]
        # objectList = ["middleres_Jaw", "middleres_OrbicularisOris"]

        # objectList = ["middleres_Jaw", "middleres_OrbicularisOris", "middleres_OrbicularisOculi_l", "middleres_OrbicularisOculi_r"]
        # objectList = ["middleres_Jaw", "middleres_OrbicularisOris"]


        def export_mesh_data(object):
            cmds.select( object, visible=True )
            selection = cmds.ls(selection=True)
            if not selection:
                cmds.error("Please select a mesh object.")
                return
            
            mesh = selection[0]
            
            # Get vertex positions
            vertices = cmds.xform(cmds.polyListComponentConversion(mesh, toVertex=True), query=True, translation=True, worldSpace=True)
            vertices = [vertices[i:i+3] for i in range(0, len(vertices), 3)]
            
            # Get face indices
            faces = cmds.polyEvaluate(mesh, face=True)
            face_indices = []
            for i in range(faces):
                # Get vertices of each face
                vertices_of_face = cmds.polyListComponentConversion(f"{mesh}.f[{i}]", toVertex=True)
                if vertices_of_face:
                    face_vertices = cmds.ls(vertices_of_face, flatten=True)
                    face_indices.append([int(v.split('[')[1].split(']')[0]) for v in face_vertices])
            
            return vertices, face_indices

        for j in range(len(objectList)):
            # pointList = retrievePosition(objectList[0])
            pointList = retrievePosition(objectList[j])
            
            # Remove the first 4 characters from objectList[j]
            objectName = objectList[j][0:]
            
            with open(save_folder + write_to, 'a') as f:
                f.write("x_" + objectName + str(i) + " = " + str(pointList) + '\n'*3)


            # export_mesh_data
            _, faces = export_mesh_data(objectList[j])
            
            with open(save_folder + write_to, 'a') as f:
                f.write("x_" + objectName + "_face_" + str(i) + " = " + str(faces) + '\n'*3)

        return



    def get_N_VertexPosition():
        
        keyframe = np.round(np.linspace(start_frame,keframe_max,keyframe_num)).astype(int)
        # print(keyframe)
        #keyframe[i]
        vertices_list = []
        for i in range(len(keyframe)):
            print("******************", keyframe[i])
            frame_number = keyframe[i]
            cmds.currentTime(str(frame_number), edit=True)
            # temp = getVertexPosition(i)
            getVertexPosition(i)
            # vertices_list.append(np.array(temp))

        # for i in reversed(range(2,4)):
        #     print(vertices_list[i] - vertices_list[i-1])

    get_N_VertexPosition()

    # get the joint position.


    ## print joint position to py file.



    # def getJointPosition(i_temp):

    #     objectList = []

    #     # i = 1
    #     # objectList = objectList + ["jaw0" + str(i)]
    #     i = 7
    #     objectList = objectList + ["lip_lower0" + str(i)]        
        
    #     # for i in range(3,156 + 1):
    #     #     objectList = objectList + ["jaw0" + str(i)]
            
    #     position_list = []
    #     rotation_list = []

    #     with open(save_folder + write_to, 'a') as f:
    #         # for i in range(3-3,156 + 1 - 3):
    #         for i in range(1-1,1 ):  #  + 1
    #             # get the position of the joint
    #             position = cmds.xform(objectList[i], query=True, worldSpace=True, translation=True)
    #             rotation = [0,0,0] # don't need to obtain it so far.
    #             position_list = position_list + [position]
    #             rotation_list = rotation_list + [rotation]
    #             # print the position
    #             # print('Position of {}: {}'.format(objectList[i], position))
    #             # f.write("x_" + objectList[i] + "=" + str(position) + '\n'+ '\n' +'\n')
    #         f.write('\n' + '\n' + "x_joint_cancatenate" + str(i_temp) + " = " + str(position_list) + '\n'+ '\n')
    #         f.write('\n' + '\n' + "x_joint_rotation_cancatenate" + str(i_temp) + " = " + str(rotation_list) + '\n'+ '\n')
            
    #         # print("debug print l236")
        

        

    #     return position_list



    # def get_N_JointsPosition():
    #     keyframe = np.round(np.linspace(start_frame,keframe_max,keyframe_num)).astype(int)
    #     # print(keyframe)
    #     #keyframe[i]
    #     joints_list = []
    #     for i in range(len(keyframe)):
    #         print("******************", keyframe[i])
    #         frame_number = keyframe[i]
    #         cmds.currentTime(str(frame_number), edit=True)
    #         temp = getJointPosition(i)
    #         joints_list.append(np.array(temp))

    #     # for i in reversed(range(2,4)):
    #         # print(joints_list[i] - joints_list[i-1])


    # get_N_JointsPosition()


# # function start

#     import re
#     # Select lower lip
#     cmds.select('middleres_Jaw', replace=True)
#     # File operations
#     # filePath = "C:/Users/zpa18/keyLabLocal/"
#     # exampleFileName = filePath + "example4.txt"

#     # Write to file
#     with open(save_folder + write_to, 'a') as fileId:
#         testString = "jointToVerDict_Jaw = "
#         fileId.write(testString)

#     # Open file for appending
#     with open(save_folder + write_to, 'a') as fileId:
#         testString = "[ "
#         fileId.write(testString)

#     # print("l108")
#     # endpoint = 5
#     # endpoint = cmds.playbackOptions(min=0, max=50, animationStartTime=0, animationEndTime=50)7
#     endpoint = 2
#     # for i in range(3, endpoint):  # for i in range(3, 157):  range(44, 51)
#     for i in range(1, endpoint):  # for i in range(3, 157):  range(44, 51)        
#         with open(save_folder + write_to, 'a') as fileId:
#             testString = "[ "
#             fileId.write(testString)

#         jointName = "jaw0" + str(i)
#         cmds.skinCluster('skinCluster1', e=True, selectInfluenceVerts=jointName)
        

        
#         import re

#         # Dictionary to store the mapping
#         index_to_vertex = {}

#         # Print vertex name
#         selectedVertices = cmds.ls(sl=True, fl=True)
        
#         # debug use 240527, only take one vertices.
#         # selectedVertices = selectedVertices[-2:]

#         for i_2, vertex in enumerate(selectedVertices[:-1]):
#             with open(save_folder + write_to, 'a') as fileId:
#                 # Parse the vertex index
#                 buffer = re.split('[.\[\]]', vertex)
#                 vertexIndices = buffer[2].split('[]')
                
#                 # Add the mapping to the dictionary
#                 # vertexIndices[0] = str(0)
#                 index_to_vertex[i_2] = vertexIndices[0]
                                
#                 if vertex != selectedVertices[-2]:
#                     testString = vertexIndices[0] + ", "
#                 else:
#                     testString = vertexIndices[0]
                    

#                 fileId.write(testString)

            

#         # Add the last vertex mapping
#         # index_to_vertex[len(selectedVertices)-1] = re.split('[.\[\]]', selectedVertices[-1])[2].split('[]')[0]

#         with open(save_folder + write_to, 'a') as fileId:

#             if i != (endpoint-1):
#                 testString = " ], "
#             else:
#                 testString = " ]"
            
#             fileId.write(testString)

#     with open(save_folder + write_to, 'a') as fileId:
#         testString = "]\n\n\n"
#         fileId.write(testString)

# # function end


    import json

    # Your dictionary (index_to_vertex)
    # index_to_vertex = {0: 'vertex1', 1: 'vertex2', 2: 'vertex3'}  # Example dictionary

    # File path where you want to save the dictionary
    
    # with open(save_folder + write_to, 'a') as fileId:
    #     testString = "\n\n\nindex_to_vertex = "
    #     fileId.write(testString)    

    # # Write the dictionary to the text file using json.dump()
    # with open(save_folder + write_to, 'a') as fileId:
    #     json.dump(index_to_vertex, fileId)
    #     fileId.write('\n\n\n')  # Add a newline character after writing the dictionary






    cmds.playbackOptions(min=0, max=50, animationStartTime=0, animationEndTime=50)

    # Function to delete existing cache nodes
    def delete_cache_nodes():
        cache_nodes = cmds.ls(type='cacheFile')
        for node in cache_nodes:
            cmds.delete(node)

    # Delete existing cache nodes
    delete_cache_nodes()

    cmds.cacheFile(createCacheNode=True, singleCache=True, startTime=0, endTime=50, fileName='myCache', dir='cacheDir')

    # Define the data
    x_joint_cancatenate0 = [[-14.761144548876384, -74.25895176498594, -17.87789981704114]]
    x_joint_rotation_cancatenate0 = [[0, 0, 0]]


    # Write the data to the file
    with open(save_folder + write_to, 'a') as file:
        file.write("# dummy data, no use\n\n")
        file.write(f"x_joint_cancatenate0 = {x_joint_cancatenate0}\n")
        file.write(f"x_joint_rotation_cancatenate0 = {x_joint_rotation_cancatenate0}\n")

    print(f"Data written successfully to {save_folder + write_to}")

    # import time

    # # Sleep for 10 seconds
    # time.sleep(5)


    def returnCoordinates(object_name, vertex_indices=None):
        # Select the object
        cmds.select(object_name, visible=True)
        
        # Get the active selection list
        selection = OpenMaya.MSelectionList()
        OpenMaya.MGlobal.getActiveSelectionList(selection)
        
        # Create an iterator for the selection list
        iterSel = OpenMaya.MItSelectionList(selection, OpenMaya.MFn.kMesh)
        
        # Get the DAG path and MFnMesh object
        dagPath = OpenMaya.MDagPath()
        iterSel.getDagPath(dagPath)
        currentInMeshMFnMesh = OpenMaya.MFnMesh(dagPath)
        
        # Get the mesh points in world space
        inMeshMPointArray = OpenMaya.MPointArray()
        currentInMeshMFnMesh.getPoints(inMeshMPointArray, OpenMaya.MSpace.kWorld)
        
        # Determine if we need to get all vertices or specific ones
        if vertex_indices is None:
            # Pre-allocate NumPy array for all points
            num_points = inMeshMPointArray.length()
            pointArray = np.zeros((num_points, 3))
            
            # Copy all points from MPointArray to NumPy array
            for i in range(num_points):
                point = inMeshMPointArray[i]
                pointArray[i, 0] = point.x
                pointArray[i, 1] = point.y
                pointArray[i, 2] = point.z
        else:
            # Extract the coordinates of the specified vertices
            vertex_coordinates = []
            for index in vertex_indices:
                if index < inMeshMPointArray.length():
                    point = inMeshMPointArray[index]
                    vertex_coordinates.append([point.x, point.y, point.z])
                else:
                    raise ValueError(f"Vertex index {index} is out of range for the mesh.")
            pointArray = np.array(vertex_coordinates)
        
        # Flatten the NumPy array to 1D
        one_d_list = pointArray.flatten()
        
        return one_d_list


    # cmds.select(sphere)

    # Change the current time
    # cmds.currentTime(1)

    # object = "middleres_Mentalis"
    # object = "mySphere"
    object = "middleres_OrbicularisOris"

    # new_coords_1 = returnCoordinates(object)


    # Change the current time
    # cmds.currentTime(15)


    # new_coords_15 = returnCoordinates(object)

    # Change the current time
    # cmds.currentTime(50)



    # new_coords_50 = returnCoordinates(object)

    # Set the start and end frames
    start_frame = 0
    # end_frame = 50
    end_frame = 4


    import time


    # cmds.select("mySphere")

    # cmds.currentTime(1)

    # # if frame == 1:
    # print("Performing action at frame 1")
    # new_coords_1 = returnCoordinates(object)
    # # Your action here

    # cmds.currentTime(2)
    # # if frame == 2: # 15:
    # print("Performing action at frame 15")
    # new_coords_15 = returnCoordinates(object)


    cmds.currentTime(1)

    # object = "mySphere"
    new_coords_1 = returnCoordinates(object)




    cmds.currentTime(2)

    # object = "mySphere"
    new_coords_15 = returnCoordinates(object)



    # Your action here
    cmds.currentTime(3)
    # if frame == 3: # 50:
    print("Performing action at frame 50")
    new_coords_50 = returnCoordinates(object)



    # save_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files"

    # i_func = 3
    # with open(save_folder + "/log" + str(i_func) + ".txt", 'a') as f:
    #     f.write("\n\n\n new_coords_1 =\n")
    #     for item in new_coords_1:
    #         f.write(f"{item}, ")


    # save_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files"

    # i_func = 3
    # with open(save_folder + "/log" + str(i_func) + ".txt", 'a') as f:
    #     f.write("\n\n\n new_coords_15 =\n")
    #     for item in new_coords_15:
    #         f.write(f"{item}, ")









    save_folder = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files"



    i_func = 3
    with open(save_folder + "/log" + str(i_func) + ".txt", 'a') as f:
        f.write("\n\n\n new_coords_1 =\n")
        for item in new_coords_1:
            f.write(f"{item}, ")
        
        f.write("\n\n\n new_coords_15 =\n")
        for item in new_coords_15:
            f.write(f"{item}, ")
        
        f.write("\n\n\n new_coords_50 =\n")
        for item in new_coords_50:
            f.write(f"{item}, ")
        
        # Calculate and print the differences
        new_coords_1_array = np.array(new_coords_1)
        new_coords_15_array = np.array(new_coords_15)
        new_coords_50_array = np.array(new_coords_50)

        diff_1_15 = new_coords_15_array - new_coords_1_array
        diff_15_50 = new_coords_50_array - new_coords_15_array

        f.write("\n\n\n Difference between new_coords_1 and new_coords_15 =\n")
        for item in diff_1_15:
            f.write(f"{item}, ")

        f.write("\n\n\n Difference between new_coords_15 and new_coords_50 =\n")
        for item in diff_15_50:
            f.write(f"{item}, ")

        f.write("\n\n\n vertices numbers of oor, =\n")
        f.write(f"{len(new_coords_50)}")
        # for item in diff_15_50:
            







    cmds.select(cl=True)
    cmds.currentCtx()

    # Close the reference
    # cmds.file(opened_file, removeReference=True)    

    return
