import time

import numpy as np
import torch

from utils.initialization import create_env
from utils.common_utils import set_seed
from utils.explore_noise import GaussNoise, EpsilonGreedy
from utils.tensorboard_setup import tb_tags
import math
import dataV22
import importlib
importlib.reload(dataV22)
from tqdm import tqdm
from scipy.spatial import cKDTree
from utils.distance_calculator import MeshDistanceCalculator

import importlib
from utils.shared_data import SharedData
from utils.landmark_influence_weights import LandmarkInfluenceWeights, create_landmark_influence_weights

class OffSampler:
    def __init__(self, index=0, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)              
        # initialize necessary hyperparameters
        self.distance_calculator = MeshDistanceCalculator()
        self.env = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 200, self.env)
        self.obs, self.info = self.env.reset(return_info = True)
        self.action_real_list = []
        self.reward_list = []
        self.angle_list = []
        self.reward_scaled_list = []
        self.distance_list = []
        self.distance_jaw_list = []
        self.distance_ooc_l_list = []
        self.distance_ooc_r_list = []
        self.distance_frontalis_list = []
        self.debug_mode_vsCode = kwargs["debug_mode_vsCode"]
        self.tic = kwargs["tic"]
        self.x_joint_init = kwargs["x_joint_init"]
        self.joint_rotate_num = kwargs["joint_rotate_num"]
        self.save_folder = kwargs["save_folder"]
        self.vertex_index = kwargs["vertex_index"]
        self.n = kwargs["n"]
        self.i_func = kwargs["i_func"]
        self.j_func = kwargs["j_func"]
        # self.x_joint_interFunctional = kwargs["x_joint_interFunctional"]
        self.x_joint_rotation_init = kwargs["x_joint_rotation_init"]
        # self.x_joint_rotation_interFunctional = kwargs["x_joint_rotation_interFunctional"]
        self.has_render = hasattr(self.env, "render")
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        self.networks = ApproxContainer(**kwargs)
        self.noise_params = kwargs["noise_params"]
        self.sample_batch_size = kwargs["batch_size_per_sampler"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.action_type = kwargs["action_type"]
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.total_sample_number = 0
        self.reward_scale = 1.0
        self.sampler_counter = 0
        self.action_mode = kwargs["action_mode"]
        self.keyframe_num = kwargs["keyframe_num"]
        self.step_counter = 0
        self.print_allow = 0
        self.print_every_forList = kwargs["print_every_forList"]
        self.degrees_min = kwargs["degrees_min"]
        self.distance_min = kwargs["distance_min"]
        self.distance_threshold_percentage = kwargs["distance_threshold_percentage"]
        self.distance_jaw_min = kwargs["distance_jaw_min"]
        self.distance_jaw_threshold_percentage = kwargs["distance_jaw_threshold_percentage"]
        self.vertices_jaw_num = kwargs["vertices_jaw_num"]
        
        # if "rotation" == self.action_mode:
        #     if self.degrees_min/self.distance_threshold_percentage < SharedData.smallest_angle_distance:
        #         SharedData.smallest_angle_distance = self.degrees_min/self.distance_threshold_percentage
        #         # self.save_best_result(next_info)
        #         # print(f"New smallest distance: {SharedData.smallest_distance}")    
        if "rotation" == self.action_mode:
            if self.distance_jaw_min/self.distance_jaw_threshold_percentage < SharedData.smallest_jaw_distance:
                SharedData.smallest_jaw_distance = self.distance_jaw_min/self.distance_jaw_threshold_percentage
                # self.save_best_result(next_info)
                # print(f"New smallest distance: {SharedData.smallest_distance}")                    
        if "translation" == self.action_mode:
            if self.distance_min/self.distance_threshold_percentage < SharedData.smallest_distance:
                SharedData.smallest_distance = self.distance_min/self.distance_threshold_percentage
                # self.save_best_result(next_info)
                # print(f"New smallest distance: {SharedData.smallest_distance}")    
        if "both" == self.action_mode:
            if self.distance_min/self.distance_threshold_percentage < SharedData.smallest_distance and self.distance_jaw_min/self.distance_jaw_threshold_percentage < SharedData.smallest_jaw_distance:
                SharedData.smallest_distance = self.distance_min/self.distance_threshold_percentage
                SharedData.smallest_jaw_distance = self.distance_jaw_min/self.distance_jaw_threshold_percentage                                       
                # self.save_best_result(next_info)
                # print(f"New smallest distance: {SharedData.smallest_distance}")    
        
        debug_mode_vsCode = 1

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


        self.cmds = cmds
        self.OpenMaya = OpenMaya        



        if self.noise_params is not None:
            if self.action_type == "continu":
                self.noise_processor = GaussNoise(**self.noise_params)
            elif self.action_type == "discret":
                self.noise_processor = EpsilonGreedy(**self.noise_params)

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def sample(self):
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()
        start_time = time.perf_counter()
        batch_data = []
        exit_flag = 0
        for _ in range(self.sample_batch_size):
            # take action using behavior policy

            # # self.obs print
            # with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\debug_printV1.txt", 'w') as f:
            #     print("self.obs in sampler: \n \n", self.obs, file = f)              

            # batch_obs = torch.from_numpy(
            #     np.expand_dims(self.obs, axis=0).astype("float32")
            # )
            # with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\debug_printV1.txt", 'a') as f:
            #     print("batch_obs in sampler: \n \n", batch_obs, file = f)        

            # Adjust numpy print options to display the full array
            np.set_printoptions(threshold=np.inf)

            # Print self.obs
            with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\debug_printV1.txt", 'w') as f:
                print("self.obs in sampler: \n \n", self.obs[0:10], file=f)
                print("self.obs size: \n \n", self.obs.shape, file=f)

            batch_obs = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )

            # Adjust torch print options to display the full tensor
            torch.set_printoptions(threshold=np.inf)

            # Print batch_obs
            with open(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\debug_printV1.txt", 'a') as f:
                # print("batch_obs in sampler: \n \n", batch_obs[0:10], file=f)
                print("batch_obs size: \n \n", batch_obs.size(), file=f)

            logits = self.networks.policy(batch_obs)

            action_distribution = self.networks.create_action_distributions(logits)
            action, logp = action_distribution.sample()
            action = action.detach()[0].numpy()
            logp = logp.detach()[0].numpy()

            if self.noise_params is not None:
                action = self.noise_processor.sample(action)
            # ensure action is array
            action = np.array(action)
            if self.action_type == "continu":
                action_clip = action.clip(
                    self.env.action_space.low, self.env.action_space.high
                )
            else:
                action_clip = action
            # interact with environment
            next_obs, reward, self.done, next_info = self.env.step(action_clip)
            # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:           
            #     print("\n  _, ", _, "\n debug log, sample l82, base_action_prime:", next_info["base_action_prime"], "\n\n ", file = f)
                    
            
       
            
            if "TimeLimit.truncated" not in next_info.keys():
                next_info["TimeLimit.truncated"] = False
            if next_info["TimeLimit.truncated"]:
                self.done = False
            data = [
                self.obs.copy(),
                self.info,
                action,
                self.reward_scale * reward,
                next_obs.copy(),
                self.done,
                logp,
                next_info,
            ]
            batch_data.append(tuple(data))
            self.obs = next_obs
            self.info = next_info

            # if next_info['success_flag'] == 1:
            #     print("in off sampler l92, hit one success. should early stop now. self.info['success_flag']:", next_info['success_flag'])            
                 
            self.action_real_list.append(next_info["base_action_prime"])            
            self.reward_list.append(reward)
            # self.angle_list.append(next_info["angle_degrees"])
            self.distance_list.append(next_info["distance"])
            self.distance_jaw_list.append(next_info["distance_jaw"])
            self.distance_ooc_l_list.append(next_info["distance_ooc_l"])
            self.distance_ooc_r_list.append(next_info["distance_ooc_r"])
            self.distance_frontalis_list.append(next_info["distance_frontalis"])
            
            # self.reward_scaled_list.append(self.reward_scale * reward)
            # allowSamplerExit = 1
            allowSamplerExit = 0
            
            if self.done or next_info["TimeLimit.truncated"]:

                if 1 == allowSamplerExit and next_info["success_flag"] == 1:   #next_info["success_flag"] == 1:
                    print("success in sampler, distance is:", next_info["distance"])



                    ep_rs_sum = sum(self.reward_list)
                    running_reward = ep_rs_sum 
                    # with open(self.save_folder+ "/log_reward" + str(self.i_func) +".txt", 'a') as f:
                    #     print("** sampler episode: ", self.sampler_counter, " \n accumulated reward:  ", running_reward, " \n   reward list : \n ", self.reward_list, " \n self.distance_list: \n ", self.distance_list, " \n self.angle_list: \n ", self.angle_list, file = f) 
                    
                    # Apply the formatting function to each list
                    
                    formatted_reward_list = self.format_list(self.reward_list)
                    formatted_distance_list = self.format_list(self.distance_list)
                    formatted_angle_list = self.format_list(self.angle_list)
                    formatted_distance_jaw_list = self.format_list(self.distance_jaw_list)

                    # Write the formatted lists to the file
                    if "rotation" == self.action_mode:
                        with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
                            print("** sampler episode: ", self.sampler_counter, 
                                " \n accumulated reward:  ", running_reward, 
                                " \n   reward list : \n ", formatted_reward_list, 
                                # " \n self.distance_list: \n ", formatted_distance_list, 
                                " \n self.distance_jaw_list: \n ", formatted_distance_jaw_list, 
                                file = f)   
                            
                    if "translation" == self.action_mode:
                        with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
                            print("** sampler episode: ", self.sampler_counter, 
                                " \n accumulated reward:  ", running_reward, 
                                " \n   reward list : \n ", formatted_reward_list, 
                                " \n self.distance_list: \n ", formatted_distance_list, 
                                # " \n self.angle_list: \n ", formatted_angle_list, 
                                file = f)                                                       
                    
                    if "both" == self.action_mode:
                        with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
                            print("** sampler episode: ", self.sampler_counter, 
                                " \n accumulated reward:  ", running_reward, 
                                " \n   reward list : \n ", formatted_reward_list, 
                                " \n self.distance_list: \n ", formatted_distance_list, 
                                " \n self.distance_jaw_list: \n ", formatted_distance_jaw_list, 
                                file = f)                               

                    final_epoch_action_list = self.action_real_list
                    delta_b0j = final_epoch_action_list
                    delta_b0j_conca = delta_b0j
                    with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                        print("final_epoch_action_list[0] , ... : ", final_epoch_action_list[0], file = f)
                        print("final_epoch_action_list: ", final_epoch_action_list, file = f)
                        print("final_epoch_action_list length: ", len(final_epoch_action_list), file = f)

                    delta_b0_conca = [0] * self.n + [0] * 3 * self.joint_rotate_num
                    # if next_info['success_flag'] == 1:
                    #     print("in off sampler l114, hit one success. should early stop now. next_info['success_flag']:", next_info['success_flag'])            
                    
                    if self.debug_mode_vsCode == 1:
                        # self.error_sum = [self.error_sum[i] + error[i] for i in range(3)]
                        # delta_b0 = [delta_b0 + delta_b0j[i] for i in range(len(delta_b0j))]
                        for i in range(len(delta_b0j_conca)):
                            delta_b0_conca = [delta_b0_conca[j] + delta_b0j_conca[i][j] for j in range(self.n + 3*self.joint_rotate_num)]


                        
                        delta_b0 = [delta_b0_conca[i:i+3] for i in range(0, len(delta_b0_conca), 3)]
                        #     delta_b0[1] = delta_b0[1] + delta_b0j[i][1]
                        #     delta_b0[2] = delta_b0[2] + delta_b0j[i][2]
                        
                        # delta_b0 = np.round(delta_b0, 3)
                        # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                        #     print("delta_b0 is \n", file = f)
                        #     for i in range(len(delta_b0)):            
                        #         print(delta_b0[i:i+1][0], file = f)
                        pass
                            # print("x_joint is \n", x_joint[0:10], "...", file = f)
                        # if next_info['success_flag'] == 1:
                        #     print("in off sampler l153, hit one success. should early stop now. next_info['success_flag']:", next_info['success_flag'])            
                                        

                        # do the move of the maya file.
                        # obs, info_temp = self.env.reset(return_info = True)

                        
                        # x_joint_ = self.x_joint_init + delta_b0_conca[0:self.n]
                        # x_joint_rotation_ = self.x_joint_rotation_init + delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num]
                        # x_joint_rotation_ = self.x_joint_rotation_init
                        
                        # if "rotation" == self.action_mode:
                        #     x_joint_ = self.x_joint_init
                        #     x_joint_rotation_ = self.x_joint_rotation_init + delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num]
                        # else:
                        #     # do translation only
                        #     x_joint_ = self.x_joint_init + delta_b0_conca[0:self.n]
                        #     x_joint_rotation_ = self.x_joint_rotation_init

                        # if "rotation" == self.action_mode:
                        #     x_joint_ = self.x_joint_init
                        #     x_joint_rotation_ = self.x_joint_rotation_init + delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num]
                        # if "translation" == self.action_mode:
                        #     # do translation only
                        #     x_joint_ = self.x_joint_init + delta_b0_conca[0:self.n]
                        #     x_joint_rotation_ = self.x_joint_rotation_init
                        
                        if "both" == self.action_mode:
                            
                            x_joint_ = self.x_joint_init + delta_b0_conca[0:self.n]
                            x_joint_rotation_ = self.x_joint_rotation_init + delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num]
                        # if next_info['success_flag'] == 1:
                        #     print("in off sampler l160, hit one success. should early stop now. next_info['success_flag']:", next_info['success_flag'])            
                                        
                        # self.env.moveJointToFinal(x_joint_, x_joint_rotation_)
                        # for i in range(0,self.joint_num):
                        #     self.cmds.move( x_joint_[3*i], x_joint_[3*i+1], x_joint_[3*i+2], 'joint'+str(i+3), absolute=True )
                        
                        # if next_info['success_flag'] == 1:
                        #     print("in off sampler l165, hit one success. should early stop now. next_info['success_flag']:", next_info['success_flag'])            
                                        
                        # # import time
                        # toc = time.process_time()
                        # with open(self.save_folder + "/log" + str(self.i_func) +".txt", 'a') as f:
                        #     print("\n ----- Computation time = " + str(  round(   (toc - self.tic)/60 ,1)    ) + "m", file = f) 
                        self.log_computation_time()

                        with open(self.save_folder + "/log" + str(self.i_func) +".txt", 'a') as f:
                            print("\n early exit at sampler ", file = f) 



                        # save the maya file.
                        # self.env.mayaSave2file()

                        ## measure the distance. 

                        maya_file_name = "/cp26-a001 after-deform-task b2.mb"
                        maya_file_to_open = self.save_folder + maya_file_name
                        # maya_file_to_open = r"C:\Users\zpa18\keyLabLocal\DSAC-T\results\DSAC_V2_m1_vertexMove_Va001\240419-182813\cp26-a001 after-deform-task b2.mb"
                        distance_in_MayaFile, s_next = self.measureDistanceInMayaFile(maya_file_to_open, next_info['s_next'])
                        print("distance_in_MayaFile: ", distance_in_MayaFile)

                        with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                            print("\n \n distance_in_MayaFile: ", distance_in_MayaFile, "\n \n", file = f) 



                        
                        print("Game over since hit one success action sequence. Exiting the program early.")

                        # torch.save(
                        #     self.networks.state_dict(),
                        #     self.save_folder
                        #     + "/apprfunc/apprfunc_{}.pkl".format("success"),
                        # )

                        # import sys
                        # sys.exit(0)  # Exit with status code 0                 
                        # terminate the whole running program   
                        # with open(self.save_folder+ "/deltaB_temp.pkl", 'wb') as f:
                        #     print("\n \n distance_in_MayaFile: ", distance_in_MayaFile, "\n \n", file = f) 

                        

                        # return x_joint_,  s_next (vertices position after the control done.)
                        # Assuming deltaB is your variable and self.save_folder is the directory where you want to save the file
                        # with open(self.save_folder + "/deltaB_temp.pkl", 'wb') as f:
                        #     pickle.dump(delta_b0_conca, f)

                        x_joint_non_concatenate = [x_joint_[i:i+3] for i in range(0, len(x_joint_), 3)]
                        x_joint_rotation_non_concatenate = [x_joint_rotation_[i:i+3] for i in range(0, len(x_joint_rotation_), 3)]
                        s_next_non_concatenate = [s_next[i:i+3] for i in range(0, len(s_next), 3)]
                        s_jaw_next_non_concatenate = [next_info["s_jaw_next"][i:i+3] for i in range(0, len(s_next), 3)]
                        import pickle

                        with open(self.save_folder + "/deltaB_temp.pkl", 'wb') as f:
                            pickle.dump({
                                'delta_b0_conca': delta_b0_conca, 
                                'x_joint_': x_joint_non_concatenate, 
                                's_next': s_next_non_concatenate,
                                's_jaw_next': s_jaw_next_non_concatenate,
                                'x_joint_rotation_': x_joint_rotation_non_concatenate,                                
                                }, f)

                        exit_flag = 1
                        end_time = time.perf_counter()
                        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000                        
                        tb_info["exit_flag"] = exit_flag
                        return batch_data, tb_info

                self.obs, self.info = self.env.reset(return_info = True)
                self.action_real_list = []
                self.distance_list = []
                self.reward_list = []

            self.update_smallest_distance(next_info)
            #bk1


        end_time = time.perf_counter()
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000
        tb_info["exit_flag"] = exit_flag


        ep_rs_sum = sum(self.reward_list)
        running_reward = ep_rs_sum    
        self.sampler_counter += 1
        # with open(self.save_folder+ "/log_reward" + str(self.i_func) +".txt", 'a') as f:
        #     print("** sampler episode: ", self.sampler_counter, " accumulated reward: \n ", running_reward, " \n \n  reward list : \n ", self.reward_list, "\n \n self.distance_list: \n ", self.distance_list, file = f) 



        if self.step_counter % self.print_every_forList == 0:                 
            
            self.print_allow = 1
            self.i_count = 0
        # print("self.print_allow", self.print_allow)

        # Function to format each element in a list to keep only digits after the decimal
  
        
        if(self.print_allow == 1):           
            
            # Apply the formatting function to each list
            formatted_reward_list = self.format_list(self.reward_list)
            formatted_distance_list = self.format_list(self.distance_list)
            # formatted_angle_list = self.format_list(self.angle_list)

            # Write the formatted lists to the file
            # with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
            #     print("** sampler episode: ", self.sampler_counter, 
            #         " \n accumulated reward:  ", running_reward, 
            #         " \n   reward list : \n ", formatted_reward_list, 
            #         " \n self.distance_list: \n ", formatted_distance_list, 
            #         " \n self.angle_list: \n ", formatted_angle_list, 
            #         file = f)       


            # Write the formatted lists to the file
            if "rotation" == self.action_mode:
                with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
                    print("** sampler episode: ", self.sampler_counter, 
                        " \n accumulated reward:  ", running_reward, 
                        " \n   reward list : \n ", formatted_reward_list, 
                        # " \n self.distance_list: \n ", formatted_distance_list, 
                        " \n self.angle_list: \n ", formatted_angle_list, 
                        file = f)   
                    
            if "translation" == self.action_mode:
                with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
                    print("** sampler episode: ", self.sampler_counter, 
                        " \n accumulated reward:  ", running_reward, 
                        " \n   reward list : \n ", formatted_reward_list, 
                        " \n self.distance_list: \n ", formatted_distance_list, 
                        # " \n self.angle_list: \n ", formatted_angle_list, 
                        file = f)   

            if "both" == self.action_mode:
                with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
                    print("** sampler episode: ", self.sampler_counter, 
                        " \n accumulated reward:  ", running_reward, 
                        " \n   reward list : \n ", formatted_reward_list, 
                        " \n self.distance_list: \n ", formatted_distance_list, 
                        # " \n self.angle_list: \n ", formatted_angle_list, 
                        file = f)   

            #bk1  
                                    
            # self.i_count += 1
            # print("self.i_count", self.i_count)
            # if self.i_count >=self.print_continue_num:
            self.print_allow = 0
        
        self.step_counter = self.step_counter + 1       
        



        # with open(self.save_folder+ "/log_reward" + str(self.i_func) +".txt", 'a') as f:
        #     print("** sampler episode: ", self.sampler_counter, " \n accumulated reward:  ", running_reward, " \n   reward list : \n ", self.reward_list, " \n self.distance_list: \n ", self.distance_list, " \n self.angle_list: \n ", self.angle_list, file = f) 


     

        return batch_data, tb_info

    def get_total_sample_number(self):
        return self.total_sample_number




    # def euclidean_distance(self, point1, point2):

    #     distance = np.linalg.norm(np.asarray(point1) - np.asarray(point2))

    #     # vertices_num = 172
    #     vertices_num = int(len(point1)/3) #172
    #     distance_ship = [0] * vertices_num * 3
    #     for i in range(0,vertices_num * 3):
    #         distance_ship[i] = math.sqrt(  (point1[i] - point2[i])**2  )  # + (point1[3*i+1] - point2[3*i+1])**2 + (point1[3*i+2] - point2[3*i+2])**2 

    #     return distance, distance_ship


    def returnCoordinates(self):
        self.cmds.select( 'middleres_Jaw', visible=True )
        selection = self.OpenMaya.MSelectionList()
        self.OpenMaya.MGlobal.getActiveSelectionList( selection )
        iterSel = self.OpenMaya.MItSelectionList(selection, self.OpenMaya.MFn.kMesh)

        # while not iterSel.isDone():
        dagPath = self.OpenMaya.MDagPath()
        iterSel.getDagPath( dagPath )
        inMeshMPointArray = self.OpenMaya.MPointArray()
        currentInMeshMFnMesh = self.OpenMaya.MFnMesh(dagPath)
        currentInMeshMFnMesh.getPoints(inMeshMPointArray, self.OpenMaya.MSpace.kWorld)
        pointList = []

        # vertices = []
        # vertex_i = self.OpenMaya.MItMeshVertex(dagPath)
        for i in range( inMeshMPointArray.length() ) :
            pointList.append( [inMeshMPointArray[i][0], inMeshMPointArray[i][1], inMeshMPointArray[i][2]] )

        adHoc=pointList
        one_d_list = []      
        one_d_list = [item for sublist in adHoc for item in sublist]
        one_d_list = np.asarray(one_d_list)

        return one_d_list



    ## measure the distance. 
    def measureDistanceInMayaFile(self,maya_file_to_open, next_obs):


        # run:
        # 
        # "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" "C:\Users\zpa18\keyLabLocal\DSAC-T\example_train\draft1.py"
        # verify distance.


        # maya_file_to_open = r"C:\Users\zpa18\keyLabLocal\DSAC-T\results\DSAC_V2_m1_vertexMove_Va001\240419-182813\cp26-a001 after-deform-task b2.mb"

        # Have a namespace if you want (recommended)
        namespace = "ref" 

        # cmds.file(maya_file_to_open, r=True )
        # Open your file
        # opened_file = self.cmds.file(maya_file_to_open, r=True, ns=namespace)

        importlib.reload(dataV22)

        # target_state = dataV13.x_middleres_Jaw
        # target_state = getattr(dataV22, "x_middleres_Jaw" + str(self.j_func))

        # x_bar_conca = [item for sublist in getattr(dataV22, "x_middleres_Jaw" + str(self.i_func)) for item in sublist]
        # x_conca = [item for sublist in getattr(dataV22, "x_middleres_Jaw" + str(self.j_func)) for item in sublist]
        target_state_temp = getattr(dataV22, "x_middleres_OrbicularisOris" + str(self.j_func))        
        # temporary = getattr(dataV22, "x_middleres_Jaw" + str(j_func))
        x_conca = [item for sublist in target_state_temp for item in sublist]                

        # # x_joint = getattr(dataV22, "x_joint_cancatenate" + str(self.i_func))
        # x_joint = self.x_joint_interFunctional
        # x_joint_rotation = self.x_joint_rotation_interFunctional

        # x_joint_conca = [item for sublist in x_joint for item in sublist]
        # x_joint_rotation_conca = [item for sublist in x_joint_rotation for item in sublist]

        # del x_joint
        # del x_joint_rotation

        # x_joint_init_conca = [item for sublist in x_joint_init for item in sublist]
        # x_joint_init_conca = x_joint_conca
        # x_joint_init_rotation_conca = x_joint_rotation_conca

        # N = len(x_bar_conca)
        # n = len(x_joint_conca)  # n is (the joint numbers * 3)
        # n_rotate = len(x_joint_rotation_conca)
        # vertices N 516 
        #   joints n is  516
        # but need to  do /3 to get real num  vertices : 172
        # real joints #: 153

        # joint_num = int(n/3) #153
        # joint_rotate_num = int(n_rotate/3) #1
        # vertices_num = int(N/3) #172

        # print("vertices dimention N", N, "\n  joints dimention n is ", n, "\n  joint_num is ", joint_num, "\n  vertices_num is ", vertices_num)


        # x_bar_conca = np.asarray(x_bar_conca)
        x_conca = np.asarray(x_conca)
        # x_joint_conca = np.asarray(x_joint_conca)
        # x_joint_init_conca = np.asarray(x_joint_init_conca)

        # x_joint_rotation_conca = np.asarray(x_joint_rotation_conca)
        # x_joint_init_rotation_conca = np.asarray(x_joint_init_rotation_conca)

        target_state = x_conca

        # s_next = self.returnCoordinates()
        s_next = next_obs
        # debug use 240527, only take one vertices.
        # s_next = s_next

        distance = 0.0
        # distance, distance_ship = self.euclidean_distance(self.control_vertices_slice(target_state, self.vertex_index), s_next)
        # rms_distance_value = self.distance_calculator.rms_distance(target_state, s_next)
        method = 'euclidean'
        distance = self.distance_calculator.calculate_distance(method, self.control_vertices_slice(target_state, self.vertex_index), s_next)         
        # del distance
        # distance = rms_distance_value

        # print("distance", distance)
        return distance, s_next

    def control_vertices_slice(self, target_state, vertex_index):
        combined_slices = []
        
        for index in vertex_index:
            slice_part = target_state[index*3:index*3+3]
            combined_slices.extend(slice_part)
        
        return combined_slices    
    
    def update_smallest_distance(self, next_info):
        if "rotation" == self.action_mode:
            if next_info["distance_jaw"] < SharedData.smallest_jaw_distance:
                SharedData.smallest_jaw_distance = next_info["distance_jaw"]    
                self.save_best_result(next_info)
                # print(f"New smallest distance: {SharedData.smallest_distance}")    
        if "translation" == self.action_mode:
            if next_info["distance"] < SharedData.smallest_distance:
                SharedData.smallest_distance = next_info["distance"]
                self.save_best_result(next_info)
                # print(f"New smallest distance: {SharedData.smallest_distance}")    
        
        if "both" == self.action_mode:
            if next_info["distance"] < SharedData.smallest_distance and next_info["distance_jaw"] < SharedData.smallest_jaw_distance:
                SharedData.smallest_jaw_distance = next_info["distance_jaw"]           
                SharedData.smallest_distance = next_info["distance"]
                # self.save_best_result(next_info)                
                # self.save_best_result(next_info)
                # print(f"New smallest distance: {SharedData.smallest_distance}")                    
                
            
                

    def save_best_result(self, next_info):
        # Code to save the best result
        #bk2
        ep_rs_sum = sum(self.reward_list)
        running_reward = ep_rs_sum 
        
        # Apply the formatting function to each list
        
        formatted_reward_list = self.format_list(self.reward_list)
        formatted_distance_list = self.format_list(self.distance_list)
        formatted_distance_jaw_list = self.format_list(self.distance_jaw_list)

        # Write the formatted lists to the file
        with open(self.save_folder + "/log_reward" + str(self.i_func) + ".txt", 'a') as f:
            print("** sampler episode: ", self.sampler_counter, 
                " \n accumulated reward:  ", running_reward, 
                " \n   reward list : \n ", formatted_reward_list, 
                " \n self.distance_list: \n ", formatted_distance_list, 
                " \n self.distance_jaw_list: \n ", formatted_distance_jaw_list, 
                file = f)   
                    
        # with open(self.save_folder+ "/log_reward" + str(self.i_func) +".txt", 'a') as f:
        #     print("** sampler episode: ", self.sampler_counter, " \n accumulated reward:  ", running_reward, " \n   reward list : \n ", self.reward_list, " \n self.distance_list: \n ", self.distance_list, " \n self.distance_jaw_list: \n ", self.distance_jaw_list, file = f) 


        final_epoch_action_list = self.action_real_list
        delta_b0j = final_epoch_action_list
        delta_b0j_conca = delta_b0j
        # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
        #     print("final_epoch_action_list[0] , ... : ", final_epoch_action_list[0], file = f)
        #     print("final_epoch_action_list: ", final_epoch_action_list, file = f)
        #     print("final_epoch_action_list length: ", len(final_epoch_action_list), file = f)

        delta_b0_conca = [0] * self.n + [0] * 3 * self.joint_rotate_num
        # if next_info['success_flag'] == 1:
        #     print("in off sampler l114, hit one success. should early stop now. next_info['success_flag']:", next_info['success_flag'])            
        
        if self.debug_mode_vsCode == 1:
            # self.error_sum = [self.error_sum[i] + error[i] for i in range(3)]
            # delta_b0 = [delta_b0 + delta_b0j[i] for i in range(len(delta_b0j))]
            for i in range(len(delta_b0j_conca)):
                delta_b0_conca = [delta_b0_conca[j] + delta_b0j_conca[i][j] for j in range(self.n + 3*self.joint_rotate_num)]


            
            delta_b0 = [delta_b0_conca[i:i+3] for i in range(0, len(delta_b0_conca), 3)]
            #     delta_b0[1] = delta_b0[1] + delta_b0j[i][1]
            #     delta_b0[2] = delta_b0[2] + delta_b0j[i][2]
            
            # delta_b0 = np.round(delta_b0, 3)
            # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
            #     print("delta_b0 is \n", file = f)
            #     for i in range(len(delta_b0)):            
            #         print(delta_b0[i:i+1][0], file = f)
             
            if "rotation" == self.action_mode:
                x_joint_ = self.x_joint_init
                x_joint_rotation_ = self.x_joint_rotation_init + delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num]
            if "translation" == self.action_mode:
                # do translation only
                x_joint_ = self.x_joint_init + delta_b0_conca[0:self.n]
                x_joint_rotation_ = self.x_joint_rotation_init
            
            if "both" == self.action_mode:
                # do translation only
                x_joint_ = self.x_joint_init + delta_b0_conca[0:self.n]
                x_joint_rotation_ = self.x_joint_rotation_init + delta_b0_conca[self.n : self.n + 3*self.joint_rotate_num]

            # import time
            # toc = time.process_time()
            # with open(self.save_folder + "/log" + str(self.i_func) +".txt", 'a') as f:
            #     print("\n ----- Computation time = " + str(  round(   (toc - self.tic)/60 ,1)    ) + "m", file = f) 
            self.log_computation_time()


            # save the maya file.
            # self.env.mayaSave2file()

            ## measure the distance. 

            maya_file_name = "/cp26-a001 after-deform-task b2.mb"
            maya_file_to_open = self.save_folder + maya_file_name
            # maya_file_to_open = r"C:\Users\zpa18\keyLabLocal\DSAC-T\results\DSAC_V2_m1_vertexMove_Va001\240419-182813\cp26-a001 after-deform-task b2.mb"
            distance_in_MayaFile, s_next = self.measureDistanceInMayaFile(maya_file_to_open, next_info['s_next'])
            # print("distance_in_MayaFile: ", distance_in_MayaFile)

            if "rotation" == self.action_mode:
                # with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                #     print("\n \n smallest so far, distance_angle_in_MayaFile: ", next_info["angle_degrees"], "\n \n", file = f) 
                with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                    print("\n \n smallest so far, distance_jaw: ", next_info["distance_jaw"], "\n \n", file = f) 
            if "translation" == self.action_mode:
                with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                    print("\n \n smallest so far, distance_in_MayaFile: ", distance_in_MayaFile, "\n \n", file = f) 

            if "both" == self.action_mode:
                with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                    print("\n \n smallest so far, distance_in_MayaFile: ", distance_in_MayaFile, "\n \n", file = f) 
                with open(self.save_folder+ "/log" + str(self.i_func) +".txt", 'a') as f:
                    print("\n \n smallest so far, distance_jaw: ", next_info["distance_jaw"], "\n \n", file = f) 


            x_joint_non_concatenate = [x_joint_[i:i+3] for i in range(0, len(x_joint_), 3)]
            x_joint_rotation_non_concatenate = [x_joint_rotation_[i:i+3] for i in range(0, len(x_joint_rotation_), 3)]
            s_next_non_concatenate = [s_next[i:i+3] for i in range(0, len(s_next), 3)]
            s_jaw_next_non_concatenate = [next_info["s_jaw_next"][i:i+3] for i in range(0, len(s_next), 3)]
            import pickle

            with open(self.save_folder + "/deltaB_temp.pkl", 'wb') as f:
                pickle.dump({
                    'delta_b0_conca': delta_b0_conca, 
                    'x_joint_': x_joint_non_concatenate, 
                    's_next': s_next_non_concatenate,
                    's_jaw_next': s_jaw_next_non_concatenate,
                    'x_joint_rotation_': x_joint_rotation_non_concatenate,                                
                    }, f)        
                
            # warning: this saving "deltaB_temp.pkl", may damage the multiple keyframes system. if one game failed, the best option is use the previous rest pose. but here,
            # it will use the best in this game, which may be worse than the rest pose.

        pass
        
    # def format_list(self, lst):
    #     return [f"{x:.3f}" for x in lst]   
    def format_list(self, lst):
        # Format each number to 3 decimal places and join them with ', '
        return ", ".join([f"{x:.3f}" for x in lst])           

    def log_computation_time(self):
        toc = time.time()  # End time using time.time()
        elapsed_minutes = round((toc - self.tic) / 60, 1)
        with open(self.save_folder + "/log" + str(self.i_func) + ".txt", 'a') as f:
            print("\n ----- Computation time = " + str(elapsed_minutes) + "m", file=f)


def create_sampler(**kwargs):
    sampler = OffSampler(**kwargs)
    return sampler
