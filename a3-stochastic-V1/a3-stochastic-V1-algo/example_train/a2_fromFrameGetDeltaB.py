def a2_fromFrameGetDeltaB(i_func, x_joint_interFunctional, x_ooc_l_joint_interFunctional, x_ooc_r_joint_interFunctional, x_frontalis_joint_interFunctional, 
                          x_jaw_joint_rotation_interFunctional, s_next_interFunctional, s_jaw_next_interFunctional, 
                          s_ooc_l_next_interFunctional, s_ooc_r_next_interFunctional, s_frontalis_next_interFunctional,
                          log_path, job_time_stamp, action_mode, previous_pose_folder, current_threshold, phase_params, 
                          contract_to_neutral_task, this_task, maya_file_to_open, estimated_steps, estimated_completion_steps_ooc_l, estimated_completion_steps_ooc_r, **argsTypeIn):

    j_func = i_func + 1
    print("a2_fromFrameGetDeltaB start")
    # now modify to only use the success action sequence
    # fix maya vs code: 
    # commandPort -name "localhost:7001" -sourceType "mel";

    debug_mode_vsCode = 1   # 1 is running in vs code.  0 is in maya.



    def portal_main():
        pass

    import time
    tic = time.time()

    import argparse
    import os
    from paths import a3_stochastic_relative_path, backup_path
    #import numpy as np

    import sys
    sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochasstic-V1-algo"))
    sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo" / "example_train"))
    sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo" / "env_gym"))
    sys.path.insert(0, str(a3_stochastic_relative_path / "a1-blendshape-guided-animation-pipeline" / "database"))
    sys.path.insert(0, str(a3_stochastic_relative_path / "b98-sampling-joint"))
    sys.path.insert(0, str(a3_stochastic_relative_path / "b98-sampling-joint" / "300joints_suite_good_old"))
    sys.path.insert(0, str(a3_stochastic_relative_path / "b98-sampling-joint" / "joint_positions" / "joint_position_ooc_l"))
    sys.path.insert(0, str(a3_stochastic_relative_path / "b98-sampling-joint" / "joint_positions" / "joint_position_ooc_r"))
    sys.path.insert(0, str(a3_stochastic_relative_path / "b98-sampling-joint" / "joint_positions" / "joint_position_frontalis"))


    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
    from matplotlib.colors import Normalize  # Import Normalize from matplotlib.colors
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation 
    # from IPython import display
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import PathCollection    

    import importlib

    import dataV22
    importlib.reload(dataV22)
    import joint_position
    import joint_position_first_wave
    import joint_position_third_wave
    importlib.reload(joint_position)
    importlib.reload(joint_position_first_wave)
    importlib.reload(joint_position_third_wave)

    import joint_position_ooc_l_base_0
    import joint_position_ooc_l_base_pose11_0
    import joint_position_ooc_l_base_pose11_AU1_0
    import joint_position_ooc_l_base_pose11_AU1_AU2_0

    import joint_position_ooc_r_base_0
    import joint_position_ooc_r_base_pose11_0
    import joint_position_ooc_r_base_pose11_AU1_0
    import joint_position_ooc_r_base_pose11_AU1_AU2_0

    import joint_position_frontalis_base_0
    import joint_position_frontalis_base_AU1_0
    import joint_position_frontalis_base_AU1_AU2_0

    import math
    import random
    from utils.initialization import create_alg,create_buffer,create_env
    from training.evaluator import create_evaluator
    # from training.off_sampler import create_sampler
    from training.trainer import create_trainer
    from utils.init_args import init_args
    # from utils.tensorboard_setup import start_tensorboard, save_tb_to_csv
    from utils.distance_calculator import MeshDistanceCalculator
    distance_calculator = MeshDistanceCalculator()
    
    from utils.phase_manager import PhaseManager

    import numpy as np

    # import importlib
    # my_module = sys.modules['utils']
    # importlib.reload(my_module)
    from importlib import reload
    from scipy.spatial import cKDTree

    utils_initialization = sys.modules['utils.initialization']
    # reload(utils_initialization)

    # from utils.initialization import create_env

    # epoch_number = 500 # also called max_iteration   #800 # 1500  # 800 # 400 + 1500 #1000 # 5 # 1000 # 200 # 1000 # 10000 # 50 # 100  #1000  5000   250
    buffer_warm_size = 200 # 20 # 20 # 100 #20 # 1000 # 100  # 100 # 1000
    print_every = 200 # 200 #  this is the print every in log0.txt. imporYtat. put it smaller if in debug. 100 # 2000 # 2000 # 500  # 500 # 10000  # 5000 #1 # 500  # 1000 # 10000 # 5000 # 1000  # 10000*2 # 10000*1  # 50000
    print_every_forList = 1000  # 1000 # 1 # 1000
    print_continue_num = 1 # 3  # 3 #1 # 3
    # printEvery = 10 # 5 # 1 # 10  # 5
    print_Every_episode = 10  # 10 # 1 # 10 # 5 # 1 # 10  # 5   this control the verbosity in evaluator print reward list.
    # step_counter_internal_max = 15000*2 # 100 # 15000*2 # 15000 # 200 # 1500 #5 #1500 # 50 # 3000 # 1000 # 2500 # 20000 # 3500 # 250 # 2000  #8000 # 15000   # 4000
    # UNIT_PRIME = 0.0005
    # distance_min = 0.035  # 0.035 #  inicial: distance 0.058986877381446264   # distance_min_single should be smaller than the initial.
    idling_punish = round(-1.0/1e6,3)  # -1.0/10000000000.0    # 0.0 or this.  # it is actually 0.0, because I used the round 3.
    reward_win = 500  # 20 # 500
    outsider_punish = -100
    

    # vertex_index = joint_position_first_wave.sampled_points_vertex_idx + joint_position.sampled_points_vertex_idx
    vertex_index = joint_position_first_wave.sampled_points_vertex_idx + joint_position.sampled_points_vertex_idx  + joint_position_third_wave.sampled_points_vertex_idx


    vertex_index_combined = joint_position_first_wave.sampled_points_vertex_idx + joint_position.sampled_points_vertex_idx  + joint_position_third_wave.sampled_points_vertex_idx
    vertex_index = vertex_index_combined    # [-1:] # [-3:]

    print(" vertex_index length is", len(vertex_index))
    print(" vertex_index  ", vertex_index[0:10])


    
    vertex_index_combined = joint_position_ooc_l_base_0.sampled_points_vertex_idx + joint_position_ooc_l_base_pose11_0.sampled_points_vertex_idx  + joint_position_ooc_l_base_pose11_AU1_0.sampled_points_vertex_idx  + joint_position_ooc_l_base_pose11_AU1_AU2_0.sampled_points_vertex_idx
    
    # vertex_index = vertex_index_combined
    vertex_ooc_l_index = vertex_index_combined    # [-1:] # [-3:]
    print(" vertex_index ooc_l length is", len(vertex_ooc_l_index))
    print(" vertex_index ooc_l ", vertex_ooc_l_index[0:10])

    vertex_index_combined = joint_position_ooc_r_base_0.sampled_points_vertex_idx + joint_position_ooc_r_base_pose11_0.sampled_points_vertex_idx  + joint_position_ooc_r_base_pose11_AU1_0.sampled_points_vertex_idx  + joint_position_ooc_r_base_pose11_AU1_AU2_0.sampled_points_vertex_idx
    
    # vertex_index = vertex_index_combined
    vertex_ooc_r_index = vertex_index_combined    # [-1:] # [-3:]
    print(" vertex_index ooc_r length is", len(vertex_ooc_r_index))
    print(" vertex_index ooc_r ", vertex_ooc_r_index[0:10])

    vertex_index_combined = joint_position_frontalis_base_0.sampled_points_vertex_idx + joint_position_frontalis_base_AU1_0.sampled_points_vertex_idx  + joint_position_frontalis_base_AU1_AU2_0.sampled_points_vertex_idx
    
    # vertex_index = vertex_index_combined
    vertex_frontalis_index = vertex_index_combined    # [-1:] # [-3:]
    print(" vertex_index frontalis length is", len(vertex_frontalis_index))
    print(" vertex_index frontalis ", vertex_frontalis_index[0:10])


    vertex_jaw_index = [29, 177, 350]

    # np.random.seed(1)

    os.environ["OMP_NUM_THREADS"] = "4"

    enable_cuda = True

    # np.random.seed(1)
    # tf.set_random_seed(1)

    # parameters controler center
    step_counter = 0



    distance_min_single = 0.00005  # 0.015 # 0.1  initial: 0.007522933990074865  , and distance_min_single should be smaller than the initial.

    # invalid_action_punish = -1.0/1.0

    # increment_parameter = 0.01  # 0.01
    # memory_size_para = 500
    # batch_size_para = 512

    # target_range_u = 5
    # target_range_d = -5




    # avg_windows = 1

    # start_plot_from = 1
    upper_spare = 0  #0.0005 # 0.01
    lower_spare = 0  #0.0005 # 0.01
    #global max and min: 1.4927679687473894, -0.3147046391970497


    # if debug_mode_vsCode == 0:
    #     import maya.cmds as cmds
    #     import maya.OpenMaya as OpenMaya

    if debug_mode_vsCode == 1:
        import maya.standalone
        maya.standalone.initialize()    
        import maya.cmds as cmds
        import maya.OpenMaya as OpenMaya

    try:
        # First time the module imports, loaded is undefined and will trigger the exception.
        # loaded is then defined in the exception block
        # when reloaded, loaded is defined as true, so reload your submodules:
        # print("loaded l51", loaded)
        if loaded:
            reload(utils_initialization)
            from utils.initialization import create_env
    except:
        loaded = True
        # print("loaded l57", loaded)
        reload(utils_initialization)
        from utils.initialization import create_env    



    if 1 == 1:
    # if __name__ == "__main__":        
    
        # Parameters Setup

        # Clear the command line arguments
        # sys.argv = ['']

        # Add your arguments
        # sys.argv.append('--log_save_interval')
        # sys.argv.append('1')



        sys.argv.append('--ini_network_dir')
        sys.argv.append(str(log_path))
        sys.argv.append('--max_iteration')
        sys.argv.append(str(argsTypeIn["max_iteration"]))
        sys.argv.append('--step_counter_internal_max')
        sys.argv.append(str(argsTypeIn["step_counter_internal_max"]))    
        # sys.argv.append('--distance_min')
        # sys.argv.append(str(argsTypeIn["distance_min"]))    
        sys.argv.append('--num_eval_episode')
        sys.argv.append(str(argsTypeIn["num_eval_episode"]))  
        sys.argv.append('--eval_interval')
        sys.argv.append(str(argsTypeIn["eval_interval"]))        
        sys.argv.append('--keyframe_num')
        sys.argv.append(str(argsTypeIn["keyframe_num"]))  
        sys.argv.append('--keyframe_translation_num')
        sys.argv.append(str(argsTypeIn["keyframe_translation_num"]))  


        sys.argv.append('--distance_threshold_percentage')
        sys.argv.append(str(argsTypeIn["distance_threshold_percentage"]))      
        sys.argv.append('--distance_jaw_threshold_percentage')
        sys.argv.append(str(argsTypeIn["distance_jaw_threshold_percentage"]))              
        sys.argv.append('--distance_ooc_l_threshold_percentage')
        sys.argv.append(str(argsTypeIn["distance_ooc_l_threshold_percentage"]))             
        sys.argv.append('--distance_ooc_r_threshold_percentage')
        sys.argv.append(str(argsTypeIn["distance_ooc_r_threshold_percentage"]))     
        sys.argv.append('--distance_frontalis_threshold_percentage')
        sys.argv.append(str(argsTypeIn["distance_frontalis_threshold_percentage"]))                     
        sys.argv.append('--unit_denominator')
        sys.argv.append(str(argsTypeIn["unit_denominator"]))       
        sys.argv.append('--sample_batch_size')
        sys.argv.append(str(argsTypeIn["sample_batch_size"]))               


        log_save_interval = 1
        sys.argv.append('--log_save_interval')
        sys.argv.append(str(log_save_interval))            
        # sys.argv.append('--env_id')
        # sys.argv.append('m1_vertexMove_Va001')    
        # sys.argv.append('--buffer_warm_size')
        # sys.argv.append(str(buffer_warm_size))  
        # sys.argv.append('--enable_cuda')
        # sys.argv.append(str(enable_cuda))            
            # parser.add_argument("--buffer_warm_size", type=int, default=10000)
        
        
        parser = argparse.ArgumentParser()

        ################################################
        # Key Parameters for Richard's
        parser.add_argument("--step_counter_internal_max", type=int, default=100, help="id of environment")
        parser.add_argument("--keyframe_num", type=int, default=10, help="keyframe numbers")
        parser.add_argument("--keyframe_translation_num", type=int, default=10)
        
        parser.add_argument("--distance_threshold_percentage", type=float, default=0.999, help="distance_threshold_percentage")
        parser.add_argument("--distance_jaw_threshold_percentage", type=float, default=0.999, help="distance_jaw_threshold_percentage")
        parser.add_argument("--distance_ooc_l_threshold_percentage", type=float, default=0.999, help="distance_ooc_l_threshold_percentage")
        parser.add_argument("--distance_ooc_r_threshold_percentage", type=float, default=0.999, help="distance_ooc_r_threshold_percentage")
        parser.add_argument("--distance_frontalis_threshold_percentage", type=float, default=0.999, help="distance_frontalis_threshold_percentage")
        
        parser.add_argument("--unit_denominator", type=float, default=100000.0, help="unit_denominator")

        
        # this many steps
        # parser.add_argument("--distance_min", type=float, default=0.05, help="DSAC_V2 or DSAC_V1")
        #set algorithm default to DSAC_V2, but it can be replaced by DSAC_V1 if you want to use the old version of DSAC.

        ################################################
        # Key Parameters for users
        parser.add_argument("--env_id", type=str, default="m1_vertexMove_Va003", help="id of environment")#gym_pendulum, gym_vertexMoveV1   m1_vertexMove_Va002
        #gym_pendulum can be replaced by other envs in the env_gym folder, such as gym_ant, gym_walker2d... but more complex envs need bigger "max iteration" setting. U can refer to "dsac_mlp_humanoid_offserial.py" to set up.
        parser.add_argument("--algorithm", type=str, default="DSAC_V2", help="DSAC_V2 or DSAC_V1")
        #set algorithm default to DSAC_V2, but it can be replaced by DSAC_V1 if you want to use the old version of DSAC.

        parser.add_argument("--enable_cuda", default=enable_cuda, help="Enable CUDA")
        parser.add_argument("--seed", default=None, help="Enable CUDA")
        ################################################
        # 1. Parameters for environment
        parser.add_argument("--reward_scale", type=float, default=1, help="reward scale factor")
        parser.add_argument("--action_type", type=str, default="continu", help="Options: continu/discret")
        parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
        parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")

        ################################################
        # 2.1 Parameters of value approximate function
        parser.add_argument(
            "--value_func_name",
            type=str,
            default="ActionValueDistri",
            help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
        )
        parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
        value_func_type = parser.parse_known_args()[0].value_func_type
        parser.add_argument("--value_hidden_sizes", type=list, default=[256,256,256])
        parser.add_argument(
            "--value_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
        parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")
        parser.add_argument("--value_min_log_std", type=int, default=-8)
        parser.add_argument("--value_max_log_std", type=int, default=8)

        # 2.2 Parameters of policy approximate function
        parser.add_argument(
            "--policy_func_name",
            type=str,
            default="StochaPolicy",
            help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
        )
        parser.add_argument(
            "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
        )
        parser.add_argument(
            "--policy_act_distribution",
            type=str,
            default="TanhGaussDistribution",
            help="Options: default/TanhGaussDistribution/GaussDistribution",
        )
        policy_func_type = parser.parse_known_args()[0].policy_func_type
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256,256,256])
        parser.add_argument(
            "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
        parser.add_argument("--policy_output_activation", type=str, default="linear", help="Options: linear/tanh")
        parser.add_argument("--policy_min_log_std", type=int, default=-20)
        parser.add_argument("--policy_max_log_std", type=int, default=0.5)

        ################################################
        # 3. Parameters for RL algorithm
        parser.add_argument("--value_learning_rate", type=float, default=0.0001)
        parser.add_argument("--policy_learning_rate", type=float, default=0.0001)
        parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
        # special parameter
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--auto_alpha", type=bool, default=False)  #   True
        parser.add_argument("--alpha", type=bool, default=0.2)
        parser.add_argument("--delay_update", type=int, default=2)
        parser.add_argument("--TD_bound", type=float, default=1)
        parser.add_argument("--bound", default=True)

        ################################################
        # 4. Parameters for trainer
        parser.add_argument(
            "--trainer",
            type=str,
            default="off_serial_trainer",
            help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
        )
        # Maximum iteration number
        parser.add_argument("--max_iteration", type=int, default=12000)
        parser.add_argument(
            "--ini_network_dir",
            type=str,
            default=None
        )
        trainer_type = parser.parse_known_args()[0].trainer

        # 4.1. Parameters for off_serial_trainer
        parser.add_argument(
            "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
        )
        # Size of collected samples before training
        parser.add_argument("--buffer_warm_size", type=int, default=buffer_warm_size) # str(buffer_warm_size) #10000
        # Max size of reply buffer
        parser.add_argument("--buffer_max_size", type=int, default=2*500)  #2*500000
        # Batch size of replay samples from buffer
        parser.add_argument("--replay_batch_size", type=int, default=256)
        # Period of sampling
        parser.add_argument("--sample_interval", type=int, default=1)

        ################################################
        # 5. Parameters for sampler
        parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
        # Batch size of sampler for buffer store
        parser.add_argument("--sample_batch_size", type=int, default=20)
        # Add noise to action for better exploration
        parser.add_argument("--noise_params", type=dict, default=None)

        ################################################
        # 6. Parameters for evaluator
        parser.add_argument("--evaluator_name", type=str, default="evaluator")
        parser.add_argument("--num_eval_episode", type=int, default=5)
        parser.add_argument("--eval_interval", type=int, default=100)
        parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")#False

        ################################################
        # 7. Data savings
        #parser.add_argument("--save_folder", type=str, default= "/home/wangwenxuan/gops_idp/gops/results/DSAC2/humanoid_r_0.2_sb_20_si_1_2")
        parser.add_argument("--save_folder", type=str, default= None)  #str(a3_stochastic_relative_path) / "a3-stochastic-V1-algo", "results", "DSAC_V2_m1_vertexMove_Va001", "rewardOutput_001"
        # Save value/policy every N updates
        parser.add_argument("--apprfunc_save_interval", type=int, default=5000)

        # Save key info every N updates
        parser.add_argument("--log_save_interval", type=int, default=100)



        ################################################
        # joint and vertices set up

        # main

        portal_main()


        # x_bar = getattr(dataV22, "x_middleres_Jaw" + str(i_func)). no. should use the end middleres_Jaw position from previous control.
        # by default x_bar is for x_bar with lip lower
        x_bar = s_next_interFunctional
        x_bar_conca = [item for sublist in x_bar for item in sublist]

        # importlib.reload(dataV22)
        # by default x is for x with lip lower
        # important line. this line decides what target pose the face will deform to.

        target_state_temp = getattr(dataV22, "x_middleres_OrbicularisOris" + str(j_func))         # x_middleres_OrbicularisOculi
        target_state_temp_ooc_l = getattr(dataV22, "x_middleres_OrbicularisOculi_l" + str(j_func))         # x_middleres_OrbicularisOculi


        def visualize(vertices, object_name):
            """
            Visualize the vertices, the sampled joint positions, and the surface mesh.

            Args:
            - vertices (list of lists): List of 3D coordinates of vertices.
            - faces (list of lists): List of indices defining the faces of the mesh.
            - sampled_points (list of lists): List of 3D coordinates of sampled joint positions.
            """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the vertices
            vertices_np = np.array(vertices)
            ax.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2], c='b', marker='o', s=1, label='Vertices')

            # Set the view angle
            ax.view_init(elev=(180+30), azim=0, roll=90)  # Change these values to set the desired view

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.legend()
            # plt.show()
            

            # Save the plot to a file
            output_folder = str(backup_path / "d83 initiate rest to AU12 smile")
            output_file = os.path.join(output_folder, f'vertice_{object_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()     

        coords_rest_non_concatenate = target_state_temp_ooc_l
        # coords_rest_non_concatenate = [coords_rest[i:i+3] for i in range(0, len(coords_rest), 3)]
        # visualize(coords_rest_non_concatenate, "target_neutral")

        # print("passed the test, now quit.")
        # import sys
        # sys.exit(0)  # Exit with status code 0  

        # temporary = getattr(dataV22, "x_middleres_Jaw" + str(j_func))
        x_conca = [item for sublist in target_state_temp for item in sublist]

        # x_joint = getattr(dataV22, "x_joint_cancatenate" + str(i_func))
        x_joint = x_joint_interFunctional
        x_joint_conca = [item for sublist in x_joint for item in sublist]

        del x_joint

        # x_joint_init_conca = [item for sublist in x_joint_init for item in sublist]
        x_joint_init_conca = x_joint_conca


        x_ooc_l_joint = x_ooc_l_joint_interFunctional
        x_ooc_l_joint_conca = [item for sublist in x_ooc_l_joint for item in sublist]

        

        x_ooc_r_joint = x_ooc_r_joint_interFunctional
        x_ooc_r_joint_conca = [item for sublist in x_ooc_r_joint for item in sublist]

        # print("x_ooc_r_joint_conca A2 L511", len(x_ooc_r_joint_conca))

        x_frontalis_joint = x_frontalis_joint_interFunctional
        x_frontalis_joint_conca = [item for sublist in x_frontalis_joint for item in sublist]

        x_ooc_l_joint_init_conca = x_ooc_l_joint_conca
        x_ooc_r_joint_init_conca = x_ooc_r_joint_conca
        x_frontalis_joint_init_conca = x_frontalis_joint_conca
        
        # get jaw's rest position
        x_jaw_bar =  s_jaw_next_interFunctional
        x_jaw_bar_conca = [item for sublist in x_jaw_bar for item in sublist]


        # get jaw's target position
        # important line. same with line 464.
        x_jaw = getattr(dataV22, "x_middleres_Jaw" + str(j_func))
        x_jaw_conca = [item for sublist in x_jaw for item in sublist]

        # x_jaw_joint = x_jaw_joint_interFunctional
        # x_jaw_joint_conca = [item for sublist in x_jaw_joint for item in sublist]
        # x_jaw_joint_init_conca = x_jaw_joint_conca

        # get ooc_l's rest position
        x_ooc_l_bar =  s_ooc_l_next_interFunctional
        x_ooc_l_bar_conca = [item for sublist in x_ooc_l_bar for item in sublist]
        print(f"x_ooc_l_bar_conca shape: {np.array(x_ooc_l_bar).shape}")
        # print(f"x_ooc_l_bar_conca sample: {x_ooc_l_bar_conca[:5]}")  # First 5 elements
        print(f"x_ooc_l_conca shape: {np.array(x_ooc_l_bar_conca).shape}")
        # print(f"x_ooc_l_conca sample: {x_ooc_l_conca[:5]}")

        # get ooc_r's rest position
        x_ooc_r_bar =  s_ooc_r_next_interFunctional
        x_ooc_r_bar_conca = [item for sublist in x_ooc_r_bar for item in sublist]

        # get frontalis's rest position
        x_frontalis_bar =  s_frontalis_next_interFunctional
        x_frontalis_bar_conca = [item for sublist in x_frontalis_bar for item in sublist]

        # get ooc's target position
        x_ooc_l = getattr(dataV22, "x_middleres_OrbicularisOculi_l" + str(j_func))
        x_ooc_l_conca = [item for sublist in x_ooc_l for item in sublist]

        # get ooc's target position
        x_ooc_r = getattr(dataV22, "x_middleres_OrbicularisOculi_r" + str(j_func))
        x_ooc_r_conca = [item for sublist in x_ooc_r for item in sublist]

        # get ooc's target position
        x_frontalis = getattr(dataV22, "x_middleres_Frontalis" + str(j_func))
        x_frontalis_conca = [item for sublist in x_frontalis for item in sublist]

        x_jaw_joint_rotation = x_jaw_joint_rotation_interFunctional #x_jaw_joint_rotation_interFunctional
        x_jaw_joint_rotation_conca = [item for sublist in x_jaw_joint_rotation for item in sublist]
        x_jaw_joint_rotation_init_conca = x_jaw_joint_rotation_conca

        x_joint_rotation_init_conca = x_jaw_joint_rotation_init_conca

        N = len(x_bar_conca)
        n = len(x_joint_conca)  # n is (the joint numbers * 3)

        N_jaw = len(x_jaw_conca)
        # n_jaw = len(x_jaw_joint_conca)  # n is (the joint numbers * 3)      
        n_jaw_rotate = len(x_jaw_joint_rotation_conca)

        n_rotate = len(x_jaw_joint_rotation_conca)
        joint_num = int(n/3) #153
        joint_rotate_num = int(n_rotate/3) #1
        vertices_num = int(N/3) #172
        # joint_num = int(n/3) #153

        vertices_jaw_num = int(N_jaw/3) #172
        # joint_jaw_num = int(n_jaw/3) #172        
        joint_jaw_rotate_num = int(n_jaw_rotate/3) #1

        #debug 240527
        #bk2
        N = len(vertex_index) * 3
        
        # n = 3
        # n_rotate = 3
        # joint_num = 1
        joint_rotate_num = 1
        joint_ooc_l_num = len(vertex_ooc_l_index)
        joint_ooc_r_num = len(vertex_ooc_r_index)
        joint_frontalis_num = len(vertex_frontalis_index)

        vertices_num = len(vertex_index)
        vertices_jaw_num = len(vertex_jaw_index)

        vertices_ooc_l_num = len(vertex_ooc_l_index)
        vertices_ooc_r_num = len(vertex_ooc_r_index)
        vertices_frontalis_num = len(vertex_frontalis_index)


        # x_conca
        # x_conca_obs = np.concatenate(
        #     [
        #         x_joint_init_conca,
        #         s_next_interFunctional[joint_num*3:joint_num*3+vertices_num*3],
        #         x_conca,  # x_conca is actually the flattened target_state
        #     ])

        # x_bar_obs = np.concatenate(
        #     [
        #         x_joint_init_conca,
        #         s_next_interFunctional[joint_num*3:joint_num*3+vertices_num*3],
        #         x_conca,  # x_conca is actually the flattened target_state
        #     ])

        # max_array = np.maximum(s_next_interFunctional[joint_num*3:joint_num*3+vertices_num*3], x_conca)
        # min_array = np.minimum(s_next_interFunctional[joint_num*3:joint_num*3+vertices_num*3], x_conca)
        max_array = np.maximum(x_bar_conca, x_conca)
        min_array = np.minimum(x_bar_conca, x_conca)        

        max_array_jaw = np.maximum(x_jaw_bar_conca, x_jaw_conca)
        min_array_jaw = np.minimum(x_jaw_bar_conca, x_jaw_conca)  

        # print(f"x_ooc_l_bar_conca shape: {np.array(x_ooc_l_bar_conca).shape}")
        # print(f"x_ooc_l_bar_conca sample: {x_ooc_l_bar_conca[:5]}")  # First 5 elements
        # print(f"x_ooc_l_conca shape: {np.array(x_ooc_l_conca).shape}")
        # print(f"x_ooc_l_conca sample: {x_ooc_l_conca[:5]}")

        max_array_ooc_l = np.maximum(x_ooc_l_bar_conca, x_ooc_l_conca)
        min_array_ooc_l = np.minimum(x_ooc_l_bar_conca, x_ooc_l_conca)  

        max_array_ooc_r = np.maximum(x_ooc_r_bar_conca, x_ooc_r_conca)
        min_array_ooc_r = np.minimum(x_ooc_r_bar_conca, x_ooc_r_conca)                        

        max_array_frontalis = np.maximum(x_frontalis_bar_conca, x_frontalis_conca)
        min_array_frontalis = np.minimum(x_frontalis_bar_conca, x_frontalis_conca)   

        # max_global = max(max_array)
        # min_global = min(min_array)

        def return_maze_bounary(max_array, min_array):
            import numpy as np

            # Assuming x_bar_conca and x_conca are your input arrays
            # max_array and min_array are already computed as follows:
            # max_array = np.maximum(x_bar_conca, x_conca)
            # min_array = np.minimum(x_bar_conca, x_conca)

            # Initialize minimum and maximum values
            MAZE_x_min = float('inf')
            MAZE_x_max = float('-inf')
            MAZE_y_min = float('inf')
            MAZE_y_max = float('-inf')
            MAZE_z_min = float('inf')
            MAZE_z_max = float('-inf')

            # Iterate through the arrays to find min and max for x, y, z
            for i in range(0, len(max_array), 3):
                x, y, z = max_array[i], max_array[i+1], max_array[i+2]
                
                MAZE_x_min = min(MAZE_x_min, x)
                MAZE_x_max = max(MAZE_x_max, x)
                MAZE_y_min = min(MAZE_y_min, y)
                MAZE_y_max = max(MAZE_y_max, y)
                MAZE_z_min = min(MAZE_z_min, z)
                MAZE_z_max = max(MAZE_z_max, z)

            for i in range(0, len(min_array), 3):
                x, y, z = min_array[i], min_array[i+1], min_array[i+2]
                
                MAZE_x_min = min(MAZE_x_min, x)
                MAZE_x_max = max(MAZE_x_max, x)
                MAZE_y_min = min(MAZE_y_min, y)
                MAZE_y_max = max(MAZE_y_max, y)
                MAZE_z_min = min(MAZE_z_min, z)
                MAZE_z_max = max(MAZE_z_max, z)

            # Print the results
            # print("MAZE_x_min:", MAZE_x_min)
            # print("MAZE_x_max:", MAZE_x_max)
            # print("MAZE_y_min:", MAZE_y_min)
            # print("MAZE_y_max:", MAZE_y_max)
            # print("MAZE_z_min:", MAZE_z_min)
            # print("MAZE_z_max:", MAZE_z_max)

            return MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max

        MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max = return_maze_bounary(max_array, min_array)
        MAZE_BOUNDARY = np.array([MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max])

        MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max = return_maze_bounary(max_array_jaw, min_array_jaw)
        MAZE_BOUNDARY_jaw = np.array([MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max])

        MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max = return_maze_bounary(max_array_ooc_l, min_array_ooc_l)
        MAZE_BOUNDARY_ooc_l = np.array([MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max])

        MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max = return_maze_bounary(max_array_ooc_r, min_array_ooc_r)
        MAZE_BOUNDARY_ooc_r = np.array([MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max])        

        MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max = return_maze_bounary(max_array_frontalis, min_array_frontalis)
        MAZE_BOUNDARY_frontalis = np.array([MAZE_x_min, MAZE_x_max, MAZE_y_min, MAZE_y_max, MAZE_z_min, MAZE_z_max])                
        # max_array_jaw = np.maximum(x_jaw_bar_conca, x_jaw_conca)
        # min_array_jaw = np.minimum(x_jaw_bar_conca, x_jaw_conca)  

        # print("vertices dimention N", N, "", "n  joints dimention n is ", n, "", "n  joint_num is ", joint_num, "", "n  vertices_num is ", vertices_num)


        x_bar_conca = np.asarray(x_bar_conca)
        x_conca = np.asarray(x_conca)
        x_jaw_conca = np.asarray(x_jaw_conca)

        x_ooc_l_bar_conca = np.asarray(x_ooc_l_bar_conca)
        x_ooc_l_conca = np.asarray(x_ooc_l_conca)
        
        x_ooc_r_bar_conca = np.asarray(x_ooc_r_bar_conca)
        x_ooc_r_conca = np.asarray(x_ooc_r_conca)

        x_frontalis_bar_conca = np.asarray(x_frontalis_bar_conca)
        x_frontalis_conca = np.asarray(x_frontalis_conca)

        x_joint_conca = np.asarray(x_joint_conca)
        x_joint_init_conca = np.asarray(x_joint_init_conca)

        x_ooc_l_joint_conca = np.asarray(x_ooc_l_joint_conca)
        x_ooc_l_joint_init_conca = np.asarray(x_ooc_l_joint_init_conca)

        x_ooc_r_joint_conca = np.asarray(x_ooc_r_joint_conca)
        x_ooc_r_joint_init_conca = np.asarray(x_ooc_r_joint_init_conca)

        x_frontalis_joint_conca = np.asarray(x_frontalis_joint_conca)
        x_frontalis_joint_init_conca = np.asarray(x_frontalis_joint_init_conca)

        x_joint_rotation_init_conca = np.asarray(x_joint_rotation_init_conca)

        target_state = x_conca
        target_jaw_state = x_jaw_conca

        target_ooc_l_state = x_ooc_l_conca
        target_ooc_r_state = x_ooc_r_conca
        target_frontalis_state = x_frontalis_conca
        #bk2


        # joint2vertix_dict = dataV22.jointToVerDict_Jaw


        # Elements: mix of triangles and quadrangles (vertex indices)
        elements = dataV22.x_middleres_OrbicularisOris_face_0


        ################################################
        # Get parameter dictionary
        args = vars(parser.parse_args())

        # def euclidean_distance(point1, point2):
        #     distance = np.linalg.norm(np.asarray(point1) - np.asarray(point2))
        #     # vertices_num = 172
        #     vertices_num = int(len(point1)/3) #172
        #     distance_ship = [0] * vertices_num * 3
        #     for i in range(0,vertices_num * 3):
        #         distance_ship[i] = math.sqrt(  (point1[i] - point2[i])**2  )  # + (point1[3*i+1] - point2[3*i+1])**2 + (point1[3*i+2] - point2[3*i+2])**2 

        #     return distance, distance_ship

        # debug use 240527, only take one vertices.
        
        # target_state = list(range((7425+1)*3))

        def control_vertices_slice(target_state, vertex_index):
            combined_slices = []
            
            for index in vertex_index:
                slice_part = target_state[index*3:index*3+3]
                combined_slices.extend(slice_part)
            
            return combined_slices



        def calculate_normal(vertex1, vertex2, vertex3):
            # Create vectors from the vertices
            v1 = np.array(vertex2) - np.array(vertex1)
            v2 = np.array(vertex3) - np.array(vertex1)
            
            # Calculate the normal using the cross product
            normal = np.cross(v1, v2)
            # print("normal", normal)
            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)
            
            return normal

        def compute_direction_reward_normal(s_next, target_state):
            # Extract vertices from the state lists
            A1, B1, C1 = s_next[0:3], s_next[3:6], s_next[6:9]
            A2, B2, C2 = target_state[0:3], target_state[3:6], target_state[6:9]
            
            # Calculate the normal vectors
            real_direction = calculate_normal(A1, B1, C1)
            desired_direction = calculate_normal(A2, B2, C2)
            # print("real_direction:", real_direction)
            # print("desired_direction:", desired_direction)
            # Normalize the vectors (already normalized in calculate_normal)
            real_direction = real_direction / np.linalg.norm(real_direction)
            desired_direction = desired_direction / np.linalg.norm(desired_direction)
            
            # Compute the cosine of the angle between the two vectors
            cos_theta = np.dot(desired_direction, real_direction)
                    
            return cos_theta    
        
        def cos_theta_to_degrees(cos_theta):
            # Compute the arccosine of cos_theta to get the angle in radians
            angle_radians = np.arccos(cos_theta)
            
            # Convert the angle from radians to degrees
            angle_degrees = np.degrees(angle_radians)
            
            return angle_degrees


        # Example usage:
        # target_state = list(range((7425+1)*3))
        # vertex_index = [3940, 5216, 7425]
        # sliced_target_state = control_vertices_slice(target_state, vertex_index)

        # print(sliced_target_state)

        # sliced_target_state = control_vertices_slice(target_state, vertex_index)

        # print(sliced_target_state)        
        initial_distance = 0.0
        # initial_distance, distance_ship = euclidean_distance(control_vertices_slice(target_state, vertex_index), control_vertices_slice(x_bar_conca, vertex_index))      

  

        method = 'euclidean'
        initial_distance = distance_calculator.calculate_distance(method, control_vertices_slice(target_state, vertex_index), control_vertices_slice(x_bar_conca, vertex_index))           
        initial_jaw_distance = distance_calculator.calculate_distance(method, control_vertices_slice(target_jaw_state, vertex_jaw_index), control_vertices_slice(x_jaw_bar_conca, vertex_jaw_index))           

        initial_ooc_l_distance = distance_calculator.calculate_distance(method, control_vertices_slice(target_ooc_l_state, vertex_ooc_l_index), control_vertices_slice(x_ooc_l_bar_conca, vertex_ooc_l_index))
        initial_ooc_r_distance = distance_calculator.calculate_distance(method, control_vertices_slice(target_ooc_r_state, vertex_ooc_r_index), control_vertices_slice(x_ooc_r_bar_conca, vertex_ooc_r_index))
        initial_frontalis_distance = distance_calculator.calculate_distance(method, control_vertices_slice(target_frontalis_state, vertex_frontalis_index), control_vertices_slice(x_frontalis_bar_conca, vertex_frontalis_index))
        # print("initial_distance", initial_distance)
        # print("control_vertices_slice(target_state, vertex_index)", control_vertices_slice(target_state, vertex_index))
        # print("control_vertices_slice(x_bar_conca, vertex_index)", control_vertices_slice(x_bar_conca, vertex_index))

        # import sys
        # sys.exit(0)  # Exit with status code 0      

        cos_theta = compute_direction_reward_normal(control_vertices_slice(x_jaw_bar_conca, vertex_jaw_index), control_vertices_slice(target_jaw_state, vertex_jaw_index))
        #bk1
        initial_angle_degrees = cos_theta_to_degrees(cos_theta)


        # rms_distance_value = rms_distance(target_state, x_bar_conca)        
        # del initial_distance
        # initial_distance = rms_distance_value        
        
        ## get initial distance
        # initial_distance = queryDistance(frame1, frame2)
        distance_min = initial_distance * args['distance_threshold_percentage'] # 0.99 # 0.999 # 0.8
        distance_jaw_min = initial_jaw_distance * args['distance_jaw_threshold_percentage'] # 0.99 # 0.999 # 0.8
        distance_ooc_l_min = initial_ooc_l_distance * args['distance_ooc_l_threshold_percentage'] # 0.99 # 0.999 # 0.8
        distance_ooc_r_min = initial_ooc_r_distance * args['distance_ooc_r_threshold_percentage'] # 0.99 # 0.999 # 0.8
        distance_frontalis_min = initial_frontalis_distance * args['distance_frontalis_threshold_percentage'] # 0.99 # 0.999 # 0.8

        

        expected_total_steps = 5200  # 6000
        
        
        translation_anatomy_limit = 35/30/(expected_total_steps/300)*0.208

        # translation_anatomy_limit = 0.012133
        # old:  0.0208 * (20/args['keyframe_num']) # for 20 frames. if 10 frames, 0.0208 * 2 ---
        rotation_anatomy_limit = 35/30/expected_total_steps*31.24
        synchronizing_adjustment_parameter = 1.0/5 *(1/3) * (1/2) *(0.75) # 1.0/5 # make the law rotates slower 5 times than before.  
        # times 2 because the r limit changed to sigmoid. the intergral is around 1/2 of the original curve.
        # 0.9 is because we want it to scale to slower to further align with translation.  0/9 CAN WORK.  0.6 also passed.  
        # new code has moved the  0.6 part to line 859 in m1: cap_max = self.UNIT_PRIME_jaw_translate * 3000/min((self.expected_total_steps - 500), 3000)

        rotation_anatomy_limit = rotation_anatomy_limit * synchronizing_adjustment_parameter
        
        # expected_total_steps_for_jaw = 33076 # 3525 # 6742   # by our syn extimation fitting function
        # rotation_anatomy_limit =  0.00009017  #  0.00014464 #  0.000238 # 0.0004 * 0.85 * 0.7 # by our syn optimization results.  # degree per second

        # # Call the helper function to get the values
        # target_rotation_deg, estimated_steps, r_limit, this_task = calculate_rotation_parameters()

        # # Initialize config with the calculated values
        # jaw_config = JawRotationConfig(
        #     target_rotation_deg=target_rotation_deg,
        #     estimated_steps=estimated_steps,
        #     r_limit=r_limit,
        #     task=this_task
        # )
        
        

        # Switch to different thresholds as needed
        # current_threshold = '0.9'  # Change this to switch between configurations
        # current_threshold = str(args['distance_jaw_threshold_percentage'])
        # current_phase = 1  # or 2, or whatever phase you're in
        # Get the configuration for current phase
        # phase_config = jaw_config.get_phase_config(current_threshold, current_phase)

        # Load all phases at once
        # phase_params = jaw_config.get_all_phases(current_threshold)
        # Now you have dictionaries like:
        # phase_params['expected_total_steps_for_jaw'][1]  # Phase 1 value
        # phase_params['expected_total_steps_for_jaw'][2]  # Phase 2 value

        # Set the configuration
        # config = jaw_config.set_current_threshold(current_threshold)
        
        # Now you can access the parameters:
        # print(f"Threshold {current_threshold} Configuration:")
        # print(f"Expected total steps for jaw: {config.expected_total_steps_for_jaw}")
        # print(f"Rotation anatomy limit (omega_max): {config.rotation_anatomy_limit}")
        # print(f"Degree: {config.degree}")
        # print(f"Result folder: {config.result_folder}")
        
        # Example of using these parameters in your code:
        # expected_total_steps_for_jaw = config.expected_total_steps_for_jaw
        # rotation_anatomy_limit = config.rotation_anatomy_limit
        # Consolidated variable system
        # expected_total_steps_for_jaw = phase_config['expected_steps']
        # rotation_anatomy_limit = phase_config['rotation_limit']
        # 0.999 case data
        # expected_total_steps_for_jaw_phase2 = 75
        # rotation_anatomy_limit_phase2 = 0.06112688

        # 0.9
        # Degree 1.3210710746775983
        # R limit: 0.00047612
        # 3503 stesps.


        # 0.5 case
        # expected_total_steps_for_jaw_phase2 = 17798 
        # rotation_anatomy_limit_phase2 = 0.00031714

        # if args['distance_jaw_threshold_percentage'] == 0.1:
        #     expected_total_steps_for_jaw_phase2 = 33076 
        #     rotation_anatomy_limit_phase2 = 0.00030677

        # print("", "nParameters ready for use in your simulation:")
        # print(f"expected_total_steps_for_jaw = {expected_total_steps_for_jaw}")
        # print(f"rotation_anatomy_limit = {rotation_anatomy_limit}")

        # sys.exit(0)

        # old: 0.00843 * (20/args['keyframe_num']) # for 20 frames. if 10 frames, 0.0208 * 2 ---
        degrees_min = initial_angle_degrees * args['distance_threshold_percentage']

        UNIT_PRIME = translation_anatomy_limit
        # old distance_min / args['unit_denominator'] # 100000.0 # 70.0 # round(distance_min / 70.0,4)

        if UNIT_PRIME > translation_anatomy_limit:  # 0.1/args['unit_denominator']:# 0.0001:
            UNIT_PRIME = translation_anatomy_limit # 0.1/args['unit_denominator']

        # UNIT_PRIME_jaw_translate = rotation_anatomy_limit

        # UNIT_PRIME_jaw_translate_phase2 = rotation_anatomy_limit_phase2
        # UNIT_PRIME_jaw_translate = phase_config['unit_prime_jaw_translate']
        # old : distance_min / args['unit_denominator'] # 70.0 # round(distance_min / 70.0,4)

        # if UNIT_PRIME_jaw_translate > rotation_anatomy_limit:
        #     UNIT_PRIME_jaw_translate = rotation_anatomy_limit

        
        rest_pose_rotational_position_degree = -5.2205880446952335  # or 5.43429431912 if using 0.1 case in "C:", "keyLabLocal_D", "a3-stochastic-V1-backup-and-results", "results", "DSAC_V2_m1_vertexMove_Va003", "job_250604-025434", "250604-025448"


        max_expected_work = 0.45 # Joule, set by past accumulated work done by muscle 's  plot.

        add_energy_to_jaw_objective_flag = 0  # add: 1, not add: 0.

        plot_distance_figures_flag = 1
        plot_4_figures_flag = 0
        k_tissue = 10000  # Stiffness of soft tissues (N/m)
        scaling_factor = 9.62e-4    # 1 maya unit to 9.62e-4 m.    1 Maya unit = 0.0962 cm   = 0.000962 m
        # # a = 35/30/(6000/300)*0.208
        # a = 35/30/6000*31.24
        # print(a)
        max_strain_energy = 11.1 # unit: J
        # muscle_forces = 70 # N  estimate by experience.
        # landmark_displacements = initial_distance * scaling_factor
        # max_muscle_work = np.sum(muscle_forces * landmark_displacements)  # For lips
        max_muscle_work = 0.000073209
        obj_full_version = 0
        adjustment_factor = 1.731 #  * 3
        if estimated_steps <= 1e-3:
            if estimated_completion_steps_ooc_r!= 0:
                estimated_steps_temp = estimated_completion_steps_ooc_r
            if estimated_completion_steps_ooc_l!= 0:
                estimated_steps_temp = estimated_completion_steps_ooc_l

            else:
                estimated_steps_temp = 2000

        else:
            estimated_steps_temp = estimated_steps
        
        overlapping_length = math.floor(estimated_steps_temp * adjustment_factor * 1 / 35) #make it around 4000  35 is the frames in animation.
        print("overlapping_length", overlapping_length)
        # note that this also affact "self.MAX_ABC_EXECUTIONS ", the multiuple phases deformation running for overlap.

        # Correct way to access the task information
        # phase_number = 1  # The phase you want to access
        # config_key = '0.9999'  # The configuration key

        # First get the config, then the phase, then the task
        # jaw_config = JawRotationConfig()
        # phase_params_tempt = jaw_config.configs[config_key]['phases'][phase_number]
        # this_task = phase_params_tempt['task']  # This will give you the full string "Neutral to IY"



        
        # previous_pose_folder = r"C:", "keyLabLocal_D", "a3-stochastic-V1-backup-and-results", "results", "DSAC_V2_m1_vertexMove_Va003", "job_250428-184837", "250428-184853"
        debug_mode = False
        
        if debug_mode == True:
            log_plot_every = 2 # 30 1
        else:
            log_plot_every = math.floor(1870/100)  # take around 100 points to plot. saving time.
        
        only_run_phase1_flag = 0 #   1: only run phase 1.  0: run phase 1 and 2.

        run_abc = False

        count_diff_log_interval = math.floor(1e7/40) # 100  # 1500  aim to sample only 20 points will be enough. sample another 20 more for extra.  if debug, can set smaller lie 1500.

        fowrad_gradient_log_interval = math.floor(1e7/40) # 4e3   same with above


        def calculate_normal_angle():
            # Get vertex coordinates for current and target states
            current_vertices = control_vertices_slice(x_jaw_bar_conca, vertex_jaw_index)
            target_vertices = control_vertices_slice(target_jaw_state, vertex_jaw_index)
            
            # Reshape into 3x3 arrays (3 vertices with x,y,z coordinates)
            current_vertices = np.array(current_vertices).reshape(3, 3)
            target_vertices = np.array(target_vertices).reshape(3, 3)
            
            # Calculate normal vector n1 from current vertices
            v1_curr, v2_curr, v3_curr = current_vertices
            edge1_curr = v2_curr - v1_curr
            edge2_curr = v3_curr - v1_curr
            n1 = np.cross(edge1_curr, edge2_curr)
            n1 = n1 / np.linalg.norm(n1)  # Normalize
            
            # Calculate normal vector n2 from target vertices
            v1_targ, v2_targ, v3_targ = target_vertices
            edge1_targ = v2_targ - v1_targ
            edge2_targ = v3_targ - v1_targ
            n2 = np.cross(edge1_targ, edge2_targ)
            n2 = n2 / np.linalg.norm(n2)  # Normalize
            
            # Calculate angle between normals (in degrees)
            dot_product = np.dot(n1, n2)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip for numerical stability
            angle_deg = np.degrees(angle_rad)
            
            # Optional: Determine the sign of the angle using a third vector
            # cross_product = np.cross(n1, n2)
            # sign = np.sign(np.dot(cross_product, edge1_curr))  # Or edge1_targ
            signed_angle_deg = abs(angle_deg) # * sign
            
            return signed_angle_deg      

        angle_needed_rotate = calculate_normal_angle()  

        # print("x_ooc_r_joint_conca A2 L1060", len(x_ooc_r_joint_conca))

        args.update({'joint_num': joint_num, 
                'joint_rotate_num': joint_rotate_num,
                'joint_ooc_l_num': joint_ooc_l_num,
                'joint_ooc_r_num': joint_ooc_r_num,
                'joint_frontalis_num': joint_frontalis_num,
                'vertices_num': vertices_num, 
                'vertices_jaw_num': vertices_jaw_num,
                'vertices_ooc_l_num': vertices_ooc_l_num,
                'vertices_ooc_r_num': vertices_ooc_r_num,
                'vertices_frontalis_num': vertices_frontalis_num,
                'action_mode': action_mode,
                'UNIT_PRIME': UNIT_PRIME,
                # 'UNIT_PRIME_jaw_translate': UNIT_PRIME_jaw_translate,
                'phase_params': phase_params,
                'initial_distance': initial_distance,
                'initial_jaw_distance': initial_jaw_distance,
                'initial_ooc_l_distance': initial_ooc_l_distance,
                'initial_ooc_r_distance': initial_ooc_r_distance,
                'initial_frontalis_distance': initial_frontalis_distance,

                'distance_min': distance_min,
                'distance_jaw_min': distance_jaw_min,
                'distance_ooc_l_min': distance_ooc_l_min,
                'distance_ooc_r_min': distance_ooc_r_min,
                'distance_frontalis_min': distance_frontalis_min,
                'reward_win': reward_win, 
                
                
                'vertex_jaw_index': vertex_jaw_index,
                # 'step_counter_internal_max': step_counter_internal_max,
     
                'step_counter': step_counter,
                'n': n,
                'N': N,
                'degrees_min': degrees_min,

                
                'keyframe_num': args['keyframe_num'],
                'print_continue_num': print_continue_num,
                'upper_spare': upper_spare,
                'lower_spare': lower_spare,
                'print_Every_episode': print_Every_episode, 
                'tic': tic,
                'i_func': i_func,
                'j_func': j_func,

                'outsider_punish': outsider_punish,
                # 'joint2vertix_dict': joint2vertix_dict,
                'max_array': max_array, 
                'min_array': min_array,      
                'x_bar_j': x_bar_conca,
                'x_jaw_bar_j': x_jaw_bar_conca,
                'x_ooc_r_bar_j': x_ooc_r_bar_conca,
                'x_ooc_l_bar_j': x_ooc_l_bar_conca,
                'x_frontalis_bar_j': x_frontalis_bar_conca,

                'x_joint': x_joint_conca,
                'x_joint_init': x_joint_init_conca,
                'x_ooc_r_joint': x_ooc_r_joint_conca,
                'x_ooc_r_joint_init': x_ooc_r_joint_init_conca,
                'x_ooc_l_joint': x_ooc_l_joint_conca,
                'x_ooc_l_joint_init': x_ooc_l_joint_init_conca,           
                'x_frontalis_joint': x_frontalis_joint_conca,
                'x_frontalis_joint_init': x_frontalis_joint_init_conca,
                'x_joint_rotation': x_jaw_joint_rotation_conca,
                'x_joint_rotation_init': x_joint_rotation_init_conca,
                'target_state': target_state,
                'target_jaw_state': target_jaw_state,
                'target_ooc_r_state': target_ooc_r_state,   
                'target_frontalis_state': target_frontalis_state,                                
                'target_ooc_l_state': target_ooc_l_state,                
                'MAZE_BOUNDARY': MAZE_BOUNDARY,       
                'MAZE_BOUNDARY_jaw': MAZE_BOUNDARY_jaw,  
                'MAZE_BOUNDARY_ooc_r': MAZE_BOUNDARY_ooc_r,       
                'MAZE_BOUNDARY_frontalis': MAZE_BOUNDARY_frontalis,       
                'MAZE_BOUNDARY_ooc_l': MAZE_BOUNDARY_ooc_l,                         
                # 'x_joint_interFunctional': x_joint_interFunctional,
                # 'x_joint_rotation_interFunctional': x_jaw_joint_rotation_interFunctional,
                'idling_punish': idling_punish, 
                
                'i_count': 0,
                'debug_mode_vsCode': debug_mode_vsCode,          
                'print_every': print_every,
                'print_every_forList': print_every_forList,       
                'distance_min_single': distance_min_single,     
                'vertex_ooc_l_index': vertex_ooc_l_index,                
                'vertex_ooc_r_index': vertex_ooc_r_index,                
                'vertex_frontalis_index': vertex_frontalis_index,                
                'vertex_index': vertex_index,                
                'elements': elements,
                'expected_total_steps': expected_total_steps,
                'rest_pose_rotational_position_degree': rest_pose_rotational_position_degree,
                'max_expected_work': max_expected_work,
                'add_energy_to_jaw_objective_flag': add_energy_to_jaw_objective_flag,
                'plot_distance_figures_flag': plot_distance_figures_flag,
                'plot_4_figures_flag': plot_4_figures_flag,
                'k_tissue': k_tissue,
                'scaling_factor': scaling_factor,
                'max_strain_energy': max_strain_energy,
                'max_muscle_work': max_muscle_work,
                'obj_full_version': obj_full_version,
                'this_task': this_task,
                'contract_to_neutral_task': contract_to_neutral_task,
                'previous_pose_folder': previous_pose_folder,
                'log_plot_every': log_plot_every,
                # 'expected_total_steps_for_jaw': expected_total_steps_for_jaw,
                # 'expected_total_steps_for_jaw_phase2': expected_total_steps_for_jaw_phase2,
                'only_run_phase1_flag': only_run_phase1_flag,
                'angle_needed_rotate': angle_needed_rotate,
                'run_abc': run_abc,
                'count_diff_log_interval': count_diff_log_interval,
                # 'UNIT_PRIME_jaw_translate_phase2': UNIT_PRIME_jaw_translate_phase2,
                'fowrad_gradient_log_interval': fowrad_gradient_log_interval,
                'overlapping_length': overlapping_length,
                'maya_file_to_open': maya_file_to_open,
                })

        print(" ", "n", "n", "n *************************** max_iteration, step_counter_internal_max, initial_distance, distance_min, initial_distance ooc r, distance_min ooc r, initial_distance ooc l, distance_min ooc l,  initial_distance frontalis, distance_min frontalis, UNIT_PRIME", 
              args['max_iteration'], args['step_counter_internal_max'], initial_distance, args['distance_min'], 
              initial_ooc_r_distance, args['distance_ooc_r_min'], 
              initial_ooc_l_distance, args['distance_ooc_l_min'], 
              initial_frontalis_distance, args['distance_frontalis_min'], 
                args['UNIT_PRIME'])
        # initial_angle_degrees, args['degrees_min'],      inidial degree, degrees_min




        env = create_env(**args)
        args = init_args(env, job_time_stamp, **args)

        code_folder = r"modified after folder f26"
        # code source

        with open(args['save_folder'] + "/log" + str(i_func) +".txt", 'w') as f:
            print("log starts ", "n ", "n", file = f)  


        with open(args['save_folder'] + "/log" + str(i_func) +".txt", 'a') as f:
            print(" action mode: ", action_mode, file = f)
            print(" initial distance: ", initial_distance, file = f)
            print(" distance_min: ", args['distance_min'], file = f)
            print(" initial distance jaw: ", initial_jaw_distance, file = f)
            print(" distance_jaw_min: ", args['distance_jaw_min'], file = f)
            print(" initial_ooc_r_distance: ", initial_ooc_r_distance, file = f)
            print(" distance_ooc_r_min: ", args['distance_ooc_r_min'], file = f)

            print(" initial_ooc_l_distance: ", initial_ooc_l_distance, file = f)
            print(" distance_ooc_l_min: ", args['distance_ooc_l_min'], file = f)
            print(" initial_frontalis_distance: ", initial_frontalis_distance, file = f)
            print(" distance_frontalis_min: ", args['distance_frontalis_min'], file = f)
            print(" This result is from code folder: ", code_folder, file = f)
            print(" This task: ", this_task, file = f)
            print(" overlapping_length: ", overlapping_length, file = f)

        #bk8
        # with open(args['save_folder']+ "/log" + str(i_func) +".txt", 'a') as f:
        #     print(" in a2, self.x_bar_j is  ", control_vertices_slice(x_bar_conca, vertex_index), file = f)


        # if initial_distance > 10:
        #start_tensorboard(args["save_folder"])
        # Step 0: Create phase manager
        phase_manager = PhaseManager(max_phases=1)  # Set your actual number of phases        
        # Step 1: create algorithm and approximate function
        # alg = create_alg(**args)
        # Step 2: create sampler in trainer
        # sampler = create_sampler(**args)
        # Step 3: create buffer in trainer
        # buffer = create_buffer(**args)
        # Step 4: create evaluator in trainer
        args['phase_manager'] = phase_manager  # Pass to evaluator
        evaluator = create_evaluator(**args)
        # Step 5: create trainer
        # use_old_policy_flag = 1
        # policy_directory = "folder"
        # , use_old_policy_flag, policy_directory
        
        trainer = create_trainer(evaluator, **args)
        # exit_flag = trainer.exit_flag_determine()
        # exit_flag = 0
        # if exit_flag == 0:

        ################################################
        # Start training ... ...
        print("Start searching trajectory ... ...")
        trainer.train()
        print("searching is finished!")
        # else:
        #     print("initial_distance < 10. skip this keyframe")
        # end of game
        # print('game over')

        # toc = time.time()
        # with open(args['save_folder'] + "/log" + str(i_func) +".txt", 'a') as f:
        #     print("\n ----- Computation time = " + str(  round(   (toc - tic)/60 ,1)    ) + "m", file = f) 
        def log_computation_time():
            toc = time.time()  # End time using time.time()
            elapsed_minutes = round((toc - tic) / 60, 1)
            with open(args['save_folder'] + "/log" + str(i_func) + ".txt", 'a') as f:
                print("\n ----- Computation time = " + str(elapsed_minutes) + "m", file=f)

        log_computation_time()

        with open(args['save_folder'] + "/log" + str(i_func) + ".txt", 'a') as f:
            print("\n vertex_index = ", vertex_index, file=f)


        ################################################
        # Plot and save training figures
        #plot_all(args["save_folder"])

        # controls the save to csv with the RL data to a folder called "data", and contains around 10 files, from "a3-stochastic-V1-algo", "utils", "tensorboard_setup.py"
        # save_tb_to_csv(args["save_folder"])

        # print("Plot & Save are finished!")

        
        import pickle

        try:
            # Try to open and read data from the first file
            with open(args['save_folder'] + "/deltaB_temp.pkl", 'rb') as f:
                data_t = pickle.load(f)
                deltaB = data_t['delta_b0_conca']
                x_joint_ = data_t['x_joint_']
                x_ooc_l_joint_ = data_t['x_ooc_l_joint_']
                x_ooc_r_joint_ = data_t['x_ooc_r_joint_']
                x_frontalis_joint_ = data_t['x_frontalis_joint_']

                x_joint_rotation_ = data_t['x_joint_rotation_']
                s_next = data_t['s_next']
                s_ooc_r_next = data_t['s_ooc_r_next']
                s_ooc_l_next = data_t['s_ooc_l_next']
                s_frontalis_next = data_t['s_frontalis_next']
                s_jaw_next = data_t['s_jaw_next']
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Trying to open and read data from the second file...")
            

        result = {
            "deltaB": deltaB,
            "x_joint_": x_joint_,
            "x_ooc_r_joint_": x_ooc_r_joint_,
            "x_frontalis_joint_": x_frontalis_joint_,
            "x_ooc_l_joint_": x_ooc_l_joint_,                        
            # "x_jaw_joint_": x_jaw_joint_,
            "x_joint_rotation_": x_joint_rotation_,
            "s_next": s_next,
            "s_jaw_next": s_jaw_next,
            "s_ooc_r_next": s_ooc_r_next,    
            "s_frontalis_next": s_frontalis_next,            
            "s_ooc_l_next": s_ooc_l_next,            
            "save_folder": args['save_folder']
        }
        return result


        # return deltaB, x_joint_, x_joint_rotation_, s_next, args['save_folder']



