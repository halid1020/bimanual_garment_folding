from actoris_harena import Agent
import os
import cv2
import torch
from hydra import compose
from .garment_phase_classifier import GarmentPhaseClassifier
from .online_garment_phase_classifier import OnlineGarmentPhaseClassifier

class VLMBasedStitchingPolicy(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'vlm_based_stitching_policy'
        self.config = config # Store config for use_reasoning, etc.

        # ... (Your existing policy loading code) ...
        import actoris_harena.api as ag_ar
        
        # Load sub-policies
        flattening_policy_config = compose(config_name=config.flattening_policy)
        folding_policy_config = compose(config_name=config.folding_policy)

        self.flattening_policy = ag_ar.build_agent(
            flattening_policy_config.agent.name,
            flattening_policy_config.agent,
            project_name=flattening_policy_config.project_name,
            exp_name=config.flattening_policy,
            save_dir=os.path.join(flattening_policy_config.save_root, config.flattening_policy)
        )

        self.folding_policy = ag_ar.build_agent(
            folding_policy_config.agent.name,
            folding_policy_config.agent,
            project_name=folding_policy_config.project_name,
            exp_name=config.folding_policy,
            save_dir=os.path.join(folding_policy_config.save_root, config.folding_policy)
        )

        self.flattening_policy.load_best()
        self.folding_policy.load_best()

        # 🔹 VLM phase classifier initialized with config flags
        if config.use_online_classifier:
            print('[VLMBasedStitchingPolicy] Using OnlineGarmentPhaseClassifier')
            self.phase_classifier = OnlineGarmentPhaseClassifier(config)
        else:
            self.phase_classifier = GarmentPhaseClassifier(config)


        self.config.use_human_reasoning_skill = config.get("use_human_reasoning_skill", True)
        
        # Buffers to store context for the VLM
        self.history_buffer = [] 
        self.demo_images = []
        self.human_reasoning_buffer_images, self.human_reasoning_buffer_phases, self.human_reasoning_buffer_reasoning = self._fetch_human_reasoning("controllers/human/human_reasoning_skill") #if self.config.use_human_reasoning_skill else None
        self.max_history_len = config.get("max_history_len", 3)

    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}
        self.history_buffer = [] # Clear history on reset
        self.flattening_policy.reset(arena_ids)
        self.folding_policy.reset(arena_ids)

    def init(self, infos):
        """
        Extract demo images from the environment info if available.
        """
        # Assuming the arena provides demo images in the 'info' during init
        if self.config.use_demo and "goals" in infos[0]:
            self.demo_images = [goal['observation']['rgb'] for goal in infos[0]['goals']]
            print('[VLMBasedStitchingPolicy] len demo images', len(self.demo_images))
            print('[VLMBasedStitchingPolicy] shape demo image', self.demo_images[0].shape)
            
        self.flattening_policy.init(infos)
        self.folding_policy.init(infos)

    def update(self, infos, actions):
        # Update history buffer with the previous observation
        if self.config.use_history:
            for info in infos:
                # Store the RGB image from the observation
                rgb = info["observation"]["rgb"]
                self.history_buffer.append(rgb)
                
                # Keep buffer size manageable for VLM context window
                if len(self.history_buffer) > self.max_history_len:
                    self.history_buffer.pop(0)

        self.flattening_policy.update(infos, actions)
        self.folding_policy.update(infos, actions)

    def act(self, info_list, update=False):
        return [self.single_act(info) for info in info_list]
    

    def _load_json_metadata(self, data_dir):
        """
        Load JSON metadata files from the specified directory.
        Returns a list of dictionaries containing the metadata.
        """
        metadata = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                with open(os.path.join(data_dir, file_name), 'r') as f:
                    data = json.load(f)
                    metadata.append(data)
        return metadata
    

    def _fetch_human_reasoning(self, data_dir="human/human_reasoning_skill"):
        """
        Placeholder for fetching human reasoning input.
        In a real implementation, this could be a UI prompt or an API call.
        For now, it returns None or a dummy reasoning string.
        """
        if self.config.use_human_reasoning_skill:
            # In practice, replace this with actual human input mechanism
            ref_images = [cv2.imread(os.path.join(data_dir, img_name)) for img_name in os.listdir(data_dir) if img_name.endswith(('.png', '.jpg', '.jpeg'))]
            
            metadata = self._load_json_metadata(data_dir)
            ref_phases = [x["phase"] for x in metadata]
            ref_reasoning = [x['reasoning'] for x in metadata]


            print(f'[VLMBasedStitchingPolicy] Loaded {len(ref_images)} reference images and their reasoning from {data_dir}')
            print(f'[VLMBasedStitchingPolicy] Loaded {len(ref_phases)} reference images and their reasoning from {data_dir}')
            print(f'[VLMBasedStitchingPolicy] Loaded {len(ref_reasoning)} reference images and their reasoning from {data_dir}')



            return ref_images, ref_phases, ref_reasoning
        else:
            return None

    def _should_folding(self, state):
        """
        Use VLM to decide phase using Current RGB + History + Demo
        """
        rgb = state["observation"]["rgb"]
        
        # Call the multimodal predict_phase
        # Note: If reasoning is enabled, it returns (phase, reasoning)
        result = self.phase_classifier.predict_phase(
            current_rgb=rgb,
            history_images=self.history_buffer if self.config.use_history else None,
            demo_images=self.demo_images if self.config.use_demo else None,
            human_reasoning_images=self.human_reasoning_buffer_images if self.config.use_human_reasoning_skill else None,
            human_reasoning_phases=self.human_reasoning_buffer_phases if self.config.use_human_reasoning_skill else None,
            human_reasoning_reasoning=self.human_reasoning_buffer_reasoning if self.config.use_human_reasoning_skill else None
        )

        if self.config.use_reasoning:
            phase, reasoning = result
            print(f"[VLMBasedStitchingPolicy] VLM Reason: {reasoning}")
        else:
            phase = result

        print(f"[VLMBasedStitchingPolicy] Predicted Phase: {phase}")
        return phase == "folding"

    def single_act(self, state, update=False):
        if self._should_folding(state):
            return self.folding_policy.single_act(state)
        else:
            return self.flattening_policy.single_act(state)