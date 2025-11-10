METHODS={
        # Maunal generate the pick-and-place action
        "Manual":{
            "env_name":"ClothFlattenGPTRGB",

            "manual":True,
            "need_box":True,
            
        },
        
        # Used in the paper with GPT generating the pick-and-place action  
        "RGBD_simple":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":True,
            "goal_config":False,
            "system_prompt_path":"system_prompts/RGBD_prompt.txt",
            "img_size":720,
            "corner_limit":15,            
        },
        
        # Let GPT be guided with the goal image. Didn't improve performance in our tests, but you can try it.
        "RGBD_goal_config":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":True,
            "goal_config":True,
            "system_prompt_path":"system_prompts/RGBD_prompt_goal_config.txt",
            "img_size":720,
            "corner_limit":15,            
        },
        
        # Let GPT be guided with the previous steps' information. Didn't improve performance in our tests, but you can try it.
        "RGBD_memory":{
        "env_name":"ClothFlattenGPTRGB",
        "need_box":True,
        "goal_config":False,
        "memory":True,
        "system_prompt_path":"system_prompts/RGBD_prompt.txt",
        "img_size":720,
        "corner_limit":15,            
        },
        
        # Let GPT be guided with the previous steps' information and goal config. Didn't improve performance in our tests, but you can try it.
        "RGBD_goal_config_memory":{
        "env_name":"ClothFlattenGPTRGB",
        "need_box":True,
        "goal_config":True,
        "memory":True,
        "system_prompt_path":"system_prompts/RGBD_prompt_goal_config.txt",
        "img_size":720,
        "corner_limit":15,            
        },
        
        # Depth-reasoning, deprecated.
        "RGBD_depth_reasoning":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":True,
            "goal_config":False,
            "depth_reasoning":True,
            "system_prompt_path":"system_prompts/RGBD_prompt_depth_reasoning.txt",
            "img_size":720,
            "corner_limit":15, 
        },
        
        # In-context learning method. You can change your demo_dir to your own directory. 
        "RGBD_ICL":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":True,
            "goal_config":False,
            "system_prompt_path":"system_prompts/RGBD_prompt.txt",
            "img_size":720,
            "corner_limit":15,
            "in_context_learning":True,
            "demo_dir":"./demo/Manual_test_14",            
        },
        
        # Naive method, without both evalution module and image preprocessing module. Ablation use.
        "RGBD_naive":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":False,
            "goal_config":False,
            "system_prompt_path":"system_prompts/RGBD_naive_prompt.txt",
            "img_size":720,
            "corner_limit":15,
            
        }
            
    }