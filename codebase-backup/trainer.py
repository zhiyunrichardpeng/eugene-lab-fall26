__all__ = ["OffSerialTrainer"]

from cmath import inf
import os
import time

import torch


class PhaseManager:
    def __init__(self, max_phases):
        self.current_phase = 1
        self.max_phases = max_phases
        self.phase_data = {
            phase: {
                'gradient_magnitudes': {},
                'term_history': {
                    'steps': [],
                    'distance': [],
                    'strain': [],
                    'muscle': []
                },
                'logs': {
                    'distance': [],
                    'step_counter': [],
                    'energy': [],
                    # ... other logs
                }
            }
            for phase in range(1, max_phases+1)
        }
        self.phase_params = {}  # Stores phase-specific parameters

class OffSerialTrainer:
    def __init__(self, evaluator, **kwargs):  # , use_old_policy_flag, policy_directory
        # self.alg = alg
        # self.sampler = sampler
        # self.buffer = buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.evaluator = evaluator
        self.phase_manager = kwargs["phase_manager"]

        # initialize center network
        if kwargs.get("ini_network_dir", "").lower() == "none":
            kwargs["ini_network_dir"] = None        
        if kwargs["ini_network_dir"] is not None:
            # self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))
            pass

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.only_run_phase1_flag = kwargs["only_run_phase1_flag"]
        # self.iteration = 0
        self.exit_flag = 0

        # pre sampling
        from tqdm import tqdm
        import time
        step_counter_internal_max = kwargs["buffer_warm_size"]
        pbar = tqdm(total=step_counter_internal_max) #, disable=True

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            print("************** USING CUDA")
            # self.networks.cuda()

        self.start_time = time.time()
    
    def exit_flag_determine(self):
        return self.exit_flag

    def step(self):
        # print("trainer l61")
        # sampling
        # sampler_tb_dict = {}
        """Main execution flow through all phases"""
        payload = None

        # payload_phase1to2 = self.evaluator.run_evaluation()
        #         # print("traner l98")
        #     # print("l120, exit_flag is", exit_flag)

        # # the second phase, controller 2 turned on.
        # if self.only_run_phase1_flag == 1:
        #     pass
        # else:
        #     self.evaluator.run_evaluation_phase2(payload_phase1to2)        

        # # if exit_flag == 1:
        # #     print("l121, should return, exit_flag is", exit_flag)
        # #     return exit_flag

        # return   # sampler_tb_dict["exit_flag"]

        for phase in range(1, self.phase_manager.max_phases+1):
            current_phase = phase
            print(f"Running phase {current_phase}")            
            # if phase == 1 and self.only_run_phase1_flag:
            #     continue
                
            payload = self.evaluator.run_evaluation(phase=current_phase, payload=payload)
            
            # Update phase manager
            # self.phase_manager.update(payload)

            # Early exit if needed
            # if payload:  #  and payload.get('exit_flag')
            #     break
        
        return          #  payload

    def train(self):
        import time
            
        self.step()  # exit_flag = 
        # if exit_flag == 1:
        #     print("trainer l184, exit_flag is", exit_flag)
        
        return


def create_trainer(evaluator, **kwargs):
    trainer = OffSerialTrainer(evaluator, **kwargs)
    print("Create trainer successfully!")
    return trainer
