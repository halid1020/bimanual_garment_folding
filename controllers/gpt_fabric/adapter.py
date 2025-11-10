from .utils import METHODS

class GPTFabricAdapter(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.method_name = config.get("method_name", 'RGBD_simple')
        self.direction_seg = config.get("direction_seg", 8)
        self.distance_seg = config.get("distance_seg", 4)
        self.method_config = METHODS[self.method_name]


        self.manual = self.method_config['manual'] if "manual" in method else False
        self.need_box=method['need_box'] if 'need_box' in method else False
        self.depth_reasoning=method['depth_reasoning'] if "depth_reasoning" in method else False
        self.memory=method['memory'] if "memory" in method else False
        self.in_context_learning=method['in_context_learning'] if "in_context_learning" in method else False
        self.goal_config=method['goal_config'] if "goal_config" in method else False
        self.system_prompt_path=method['system_prompt_path'] if "system_prompt_path" in method else "system_prompts/RGBD_prompt.txt"
        self.demo_dir=method['demo_dir'] if "demo_dir" in method else None
        self.img_size=method['img_size'] if "img_size" in method else 720
        self.fine_tuning=method["fine_tuning"] if "fine_tuning" in method else False
        self.fine_tuning_model_path=method["fine_tuning_model_path"] if "fine_tuning_model_path" in method else None
        self.corner_limit=method['corner_limit'] if 'corner_limit' in method else 10

        self.method = RGBD_manipulation_part_obs()
            # env=env,
            # env_name=method["env_name"],
            # obs_dir=save_obs_dir,
            # goal_image=goal_image,
            # goal_config=goal_config,
            # goal_depth=goal_depth,
            # img_size=img_size,
            # in_context_learning=in_context_learning,
            # demo_dir=demo_dir,
            

    def single_act(self, info, update=False):
        goal_image = info['goal']['rgb']
        goal_depth = info['goal']['depth']

        # I need to figure what is following
        # goal_depth=np.round(goal_depth[:,:,3:].squeeze(),3)
        messages = self.internal_states[aid]['messages']

        messages,last_step_info, pick_point, place_point = self.method.gpt_single_step(
            rgb, depth,
            headers=headers,
            messages=messages,
            system_prompt_path=system_prompt_path,
            memory=memory,
            need_box=need_box,
            corner_limit=corner_limit,
            last_step_info=last_step_info,
            depth_reasoning=depth_reasoning,
            direction_seg=self.direction_seg,
            distance_seg=self.distance_seg,
            specifier="step"+str(i))

        self.internal_states[aid]['messages'] = messages
        

        return {
            'norm-pixel-pick-and-place': {
                'pick_0': pick_point,
                'place_0': place_point
            }
        }