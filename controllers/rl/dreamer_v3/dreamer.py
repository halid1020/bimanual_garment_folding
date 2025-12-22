import argparse
import functools
import os
import pathlib
import numpy as np
import ruamel.yaml as yaml

from .exploration import *
from .models import *
from .tools import *

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()

def draw_text_with_bg(
    img,
    text,
    org,
    font,
    font_scale,
    text_color,
    bg_color=(0, 0, 0),
    thickness=1,
    padding=4
):
    (tw, th), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = org

    # Background rectangle
    cv2.rectangle(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        thickness=-1
    )

    # Text
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


class Dreamer(nn.Module):
    def __init__(self, obs_space, rnd_act_space, config, logger, dataset, vis_dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self.train_every = batch_steps / config.train_ratio
        #print('train every', train_every)
        self._should_train = Every(self.train_every)
        self._should_pretrain = Once()
        self._should_reset = Every(config.reset_every)
        self._should_expl = Until(int(config.expl_until / config.action_repeat))
        self._updates_per_step = self._config.get('updates_per_step', 1)
        self._metrics = {}
        # this is update step
        #self._action_step = logger.step // config.action_repeat
        # if self._action_step > 0:
        #     self._should_pretrain()
        #print('init step', self._action_step)
        self._update_count = 0
        self._dataset = dataset
        self._vis_dataset = vis_dataset
        self._wm = WorldModel(obs_space, config)
        self._task_behavior = ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: Random(config, rnd_act_space),
            plan2explore=lambda: Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
    
    def set_data_augmenter(self, data_augmenter):
        self._wm.data_augmenter = data_augmenter

    def log(self):
        if self._should_log(self._logger.update_step):
            print(f'[dreamerV3] Log training at update step {self._logger.update_step}')
            for name, values in self._metrics.items():
                self._logger.scalar(name, float(np.mean(values)))
                self._metrics[name] = []
            if self._config.train_video_pred_log:
                openl = self._wm.video_pred(
                    next(self._vis_dataset),
                    episodes=6,
                    post_steps=self._config.vis_post_steps,
                    prior_steps=self._config.vis_prior_steps)  # B, T, H, W, C

                # Convert openl to T, H, B*W, C
                B, T, H, W, C = openl.shape

                # permute to (T, H, B, W, C)
                x = openl.permute(1, 2, 0, 3, 4)

                # reshape to (T, H, B*W, C)
                convert_openl = x.reshape(T, H, B * W, C)
                self._logger.video("train_openl", to_np(convert_openl))

                # First sample
                single_openl = openl[0]  # (T, H, W, C)

                # Convert to (H, T*W, C)
                convert_single_openl = single_openl.permute(1, 0, 2, 3).reshape(H, T * W, C)

                img = to_np(convert_single_openl)  # (H, T*W, C)
                img = (img.clip(0, 1) * 255).astype(np.uint8)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1

                # Mild colors (BGR)
                posterior_color = (120, 200, 120)  # soft green
                prior_color = (100, 160, 200)      # soft blue
                row_text_color = (255, 255, 255)   # white

                

                img = cv2.resize(img, (128*T, 128*3))

                H, TW, C = img.shape
                W = TW // T

                post_steps = self._config.vis_post_steps
                prior_steps = self._config.vis_prior_steps

                # -------------------------------
                # Draw red vertical separator
                # -------------------------------
                sep_x = post_steps * W
                cv2.line(
                    img,
                    (sep_x, 0),
                    (sep_x, H),
                    color=(255, 0, 0),
                    thickness=3
                )

                # -------------------------------
                # Top-right captions (Posterior / Prior)
                # -------------------------------

                y_text = 18
                margin = W - W//2
                for i in range(post_steps):
                    x = i * W + margin
                    draw_text_with_bg(
                        img,
                        f"Post {i+1}",
                        (x, y_text),
                        font,
                        font_scale,
                        posterior_color,
                        thickness=thickness
                    )

                # Prior captions
                for i in range(prior_steps):
                    x = (post_steps + i) * W + margin
                    draw_text_with_bg(
                        img,
                        f"Prior {i+1}",
                        (x, y_text),
                        font,
                        font_scale,
                        prior_color,
                        thickness=thickness
                    )

                # -------------------------------
                # Left-side row labels
                # -------------------------------
                row_names = ["Truth", "Model", "Error"]
                row_height = H // len(row_names)
                margin = row_height - 20
                for i, name in enumerate(row_names):
                    y = i * row_height + margin
                    draw_text_with_bg(
                        img,
                        name,
                        (6, y),
                        font,
                        font_scale,
                        row_text_color,
                        thickness=thickness
                    )

                # Convert back to float for logger
                img = img.astype(np.float32) / 255.0

                self._logger.image("train_recon", img)

                #self._logger.image("train_recon", to_np(convert_single_openl))
                    
                
            self._logger.write(fps=True)

    def __call__(self, obs, reset, state=None, training=True):
        # step = self._action_step
        #print('training', training, 'step', step)
        if training:
            steps_to_update = self._should_train(self._action_step) * self._updates_per_step
            #print('update steps', steps)
            for _ in range(steps_to_update):
                self._train(next(self._dataset))
                #print('train!!!')
                #self._update_count += 1
                self._logger.update_step += 1
                self._metrics["action_step"] = self._action_step
                self.log()
            

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._action_step += len(reset)
        return policy_output, state

    def _policy(self, obs_in, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs_in)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._action_step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        # print('train data keys', data.keys())
        # print('train data image states min and max', data['image'].min(),  data['image'].max())
        #data = self.data_augmenter(data)
        # print('augment data keys', data.keys())
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = sample_episodes(episodes, config.batch_length)
    dataset = from_generator(generator, config.batch_size)
    return dataset