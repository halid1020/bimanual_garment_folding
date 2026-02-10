from .utils import METHODS
from actoris_harena import Agent
from .rgbd_manipulation_part_obs import RGBD_manipulation_part_obs
from openai import OpenAI

class GPTFabricAdapter(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.method_name = config.get("method_name", 'RGBD_simple')
        self.direction_seg = config.get("direction_seg", 8)
        self.distance_seg = config.get("distance_seg", 4)
        self.method_config = METHODS[self.method_name]


        self.manual = self.method_config['manual'] if "manual" in self.method_config else False
        self.need_box=self.method_config['need_box'] if 'need_box' in self.method_config else False
        self.depth_reasoning=self.method_config['depth_reasoning'] if "depth_reasoning" in self.method_config else False
        self.memory=self.method_config['memory'] if "memory" in self.method_config else False
        self.in_context_learning=self.method_config['in_context_learning'] if "in_context_learning" in self.method_config else False
        self.goal_config=self.method_config['goal_config'] if "goal_config" in self.method_config else False
        self.system_prompt_path=self.method_config['system_prompt_path'] if "system_prompt_path" in self.method_config else "system_prompts/RGBD_prompt.txt"
        self.demo_dir=self.method_config['demo_dir'] if "demo_dir" in self.method_config else None
        self.img_size=self.method_config['img_size'] if "img_size" in self.method_config else 720
        self.fine_tuning=self.method_config["fine_tuning"] if "fine_tuning" in self.method_config else False
        self.fine_tuning_model_path=self.method_config["fine_tuning_model_path"] if "fine_tuning_model_path" in self.method_config else None
        self.corner_limit=self.method_config['corner_limit'] if 'corner_limit' in self.method_config else 10

        self.method = RGBD_manipulation_part_obs()
        # Set up the API key for GPT-4 communication
        with open("./assets/GPT-API-KEY.txt", "r") as f:
            api_key = f.read().strip()

        self.client = OpenAI(api_key=api_key)

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

            # env=env,
            # env_name=method["env_name"],
            # obs_dir=save_obs_dir,
            # goal_image=goal_image,
            # goal_config=goal_config,
            # goal_depth=goal_depth,
            # img_size=img_size,
            # in_context_learning=in_context_learning,
            # demo_dir=demo_dir,
        
    def reset(self, arena_ids):
        for aid in arena_ids:
            self.internal_states[aid] = {}
            self.internal_states[aid]['messages'] = []
            self.internal_states[aid]['last_step_info'] = None

    def single_act(self, info, update=False):
        goal_image = info['goal']['rgb']
        goal_depth = info['goal']['depth']

        aid = info['arena_id']
        messages = self.internal_states[aid]['messages']

        messages,last_step_info, pick_point, place_point = self.method.gpt_single_step(
            info,
            self.headers, info['observation']['rgb'], 
            info['observation']['depth'], 
            info['observation']['mask'],
            cloth_size=info['cloth_size'],
            cloth_particle_radius=info['arena']._env.cloth_particle_radius,
            goal_mask=info['arena'].flatten_obs['mask'],
            messages=messages,
            goal_config=self.goal_config,
            system_prompt_path=self.system_prompt_path,
            memory=self.memory,
            need_box=self.need_box,
            corner_limit=self.corner_limit,
            last_step_info=self.internal_states[aid]['last_step_info'],
            depth_reasoning=self.depth_reasoning,
            direction_seg=self.direction_seg,
            distance_seg=self.distance_seg,
            specifier="step"+str(info['observation']['action_step'])
        )

        self.internal_states[aid]['last_step_info'] = last_step_info
        self.internal_states[aid]['messages'] = messages

        H, W, _ = info['observation']['rgb'].shape
        print('H, W', H, W)
        print('result pick pixel', pick_point)
        print('result place pixel', place_point)

        # Normalize after swapping axes
        def normalize_pixel(px, H, W):
            # Swap: (x, y) -> (y, x)
            y, x = px

            x_norm = (1.0*x / W ) * 2 - 1
            y_norm = (1.0*y / H) * 2 - 1
            return [x_norm, y_norm]

        pick_point_norm = normalize_pixel(pick_point, H, W)
        place_point_norm = normalize_pixel(place_point, H, W)
        print('norm pick', pick_point_norm)
        print('norm place', place_point_norm)

        return {
            'norm-pixel-pick-and-place': {
                'pick_0': pick_point_norm,
                'place_0': place_point_norm
            },
            'pick_0': pick_point_norm,
            'place_0': place_point_norm
        }