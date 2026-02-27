import datetime
import json
from agent_arena import Agent
import os
import torch
from hydra import compose
from .garment_phase_classifier import GarmentPhaseClassifier
from PIL import ImageTk, Image
from pathlib import Path


class HumanBasedStitchingPolicy(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'human_based_stitching_policy'
        self.config = config # Store config for use_reasoning, etc.

        # ... (Your existing policy loading code) ...
        import agent_arena.api as ag_ar
        
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
        self.phase_classifier = GarmentPhaseClassifier(config)
        
        # Buffers to store context for the VLM
        self.history_buffer = [] 
        self.demo_images = []
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
            print('[HumanBasedStitchingPolicy] len demo images', len(self.demo_images))
            print('[HumanBasedStitchingPolicy] shape demo image', self.demo_images[0].shape)
            
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

    def _should_folding(self, state):
        """
        Use VLM to decide phase using Current RGB + History + Demo
        """
        rgb = state["observation"]["rgb"]
        
        # Call the multimodal predict_phase
        # Note: If reasoning is enabled, it returns (phase, reasoning)
        result = self.phase_classifier.let_human_reason_and_decide(
            current_rgb=rgb,
            history_images=self.history_buffer if self.config.use_history else None,
            demo_images=self.demo_images if self.config.use_demo else None
        )

        reasoning = None
        if self.config.use_reasoning:
            phase, reasoning = result
            print(f"[HumanBasedStitchingPolicy] Human Reason: {reasoning}")
        else:
            phase = result


        self.save_image_data(
            save_dir=self.config.save_data,
            image=rgb,
            phase=phase,
            reasoning=reasoning,
        )

        print(f"[HumanBasedStitchingPolicy] Human decided Phase: {phase}")
        return phase == "folding"

    def single_act(self, state, update=False):
        if self._should_folding(state):
            return self.folding_policy.single_act(state)
        else:
            return self.flattening_policy.single_act(state)
        
    

    def save_image_data(
        self,
        save_dir: str,
        image,
        phase: str,
        reasoning: str = None,
    ) -> None:
        """
        Saves an image and a linked JSON file containing text metadata.

        Files produced:
            base_name.png
            base_name.json
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        basename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # --- Save image ---
        image_path = save_dir / f"image_{basename}.png"
        Image.fromarray(image).save(image_path)

        # --- Save linked metadata ---
        metadata = {
            "image_file": image_path.name,
            "phase": phase,
            "reasoning": reasoning,
        }


        metadata_path = save_dir / f"metadata_{basename}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)