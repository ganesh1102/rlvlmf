#!/usr/bin/env python3
import numpy as np
import torch
import os
import time
import pickle as pkl

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from reward_model_score import RewardModelScore
from collections import deque
from prompt import clip_env_prompts

import utils
import hydra
from PIL import Image

from vlms.blip_infer_2 import blip2_image_text_matching
from vlms.clip_infer import clip_infer_score as clip_image_text_matching
import cv2


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.cfg.prompt = clip_env_prompts[cfg.env]
        self.cfg.clip_prompt = clip_env_prompts[cfg.env]
        self.reward = self.cfg.reward  # what type of reward to use
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        current_file_path = os.path.dirname(os.path.realpath(__file__))
        os.system(f"cp {current_file_path}/prompt.py {self.logger._log_dir}")

        # --------------------------------------------------------------------
        # CHANGED: Environment creation
        # --------------------------------------------------------------------
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            self.env = utils.make_classic_control_env(cfg)
        else:
            # Everything else (including Genesis) goes through make_env
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)

        # image sizing logic (unchanged)
        image_height = image_width = cfg.image_size
        self.resize_factor = 1
        if "sweep" in cfg.env or 'drawer' in cfg.env or "soccer" in cfg.env:
            image_height = image_width = 300 
        if "Rope" in cfg.env:
            image_height = image_width = 240
            self.resize_factor = 3
        elif "Water" in cfg.env:
            image_height = image_width = 360
            self.resize_factor = 2
        if "CartPole" in cfg.env:
            image_height = image_width = 200
        if "Cloth" in cfg.env:
            image_height = image_width = 360

        self.image_height = image_height
        self.image_width = image_width

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity) if not self.cfg.image_reward else 200000,
            self.device,
            store_image=self.cfg.image_reward,
            image_size=image_height)

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiate the reward model (unchanged)
        reward_model_class = RewardModel
        if self.reward == 'learn_from_preference':
            reward_model_class = RewardModel
        elif self.reward == 'learn_from_score':
            reward_model_class = RewardModelScore

        self.reward_model = reward_model_class(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
            capacity=cfg.max_feedback * 2,
            vlm_label=cfg.vlm_label,
            vlm=cfg.vlm,
            env_name=cfg.env,
            clip_prompt=clip_env_prompts[cfg.env],
            log_dir=self.logger._log_dir,
            flip_vlm_label=cfg.flip_vlm_label,
            cached_label_path=cfg.cached_label_path,
            image_reward=cfg.image_reward,
            image_height=image_height,
            image_width=image_width,
            resize_factor=self.resize_factor,
            resnet=cfg.resnet,
            conv_kernel_sizes=cfg.conv_kernel_sizes,
            conv_strides=cfg.conv_strides,
            conv_n_channels=cfg.conv_n_channels,
        )

        if self.cfg.reward_model_load_dir != "None":
            print(f"loading reward model at {self.cfg.reward_model_load_dir}")
            self.reward_model.load(self.cfg.reward_model_load_dir, 1000000)

        if self.cfg.agent_model_load_dir != "None":
            print(f"loading agent model at {self.cfg.agent_model_load_dir}")
            self.agent.load(self.cfg.agent_model_load_dir, 1000000)

    def evaluate(self, save_additional=False):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        all_ep_infos = []
        for episode in range(self.cfg.num_eval_episodes):
            print(f"evaluating episode {episode}")
            images = []
            obs = self.env.reset()
            if "metaworld" in self.cfg.env:
                obs = obs[0]

            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            ep_info = []
            rewards = []
            t_idx = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                try:
                    obs, reward, done, extra = self.env.step(action)
                except:
                    obs, reward, terminated, truncated, extra = self.env.step(action)
                    done = terminated or truncated
                ep_info.append(extra)

                rewards.append(reward)
                if "metaworld" in self.cfg.env:
                    rgb_image = self.env.render()
                    if self.cfg.mode != 'eval':
                        rgb_image = rgb_image[::-1, :, :]
                        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                            rgb_image = rgb_image[100:400, 100:400, :]
                    else:
                        rgb_image = rgb_image[::-1, :, :]
                elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                    rgb_image = self.env.render(mode='rgb_array')
                else:
                    # CHANGED: unified rendering for all other envs
                    rgb_image = self.env.render(mode='rgb_array')

                # CHANGED: always append frames
                images.append(rgb_image)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra.get('success', 0))

                t_idx += 1
                if self.cfg.mode == 'eval' and t_idx > 50:
                    break

            all_ep_infos.append(ep_info)

            save_gif_path = os.path.join(
                save_gif_dir,
                f'step{self.step:07}_episode{episode:02}_{round(true_episode_reward, 2)}.gif'
            )
            utils.save_numpy_as_gif(np.array(images), save_gif_path)

            if save_additional:
                save_image_dir = os.path.join(self.logger._log_dir, 'eval_images')
                os.makedirs(save_image_dir, exist_ok=True)
                for i, image in enumerate(images):
                    Image.fromarray(image).save(
                        os.path.join(
                            save_image_dir,
                            f'step{self.step:07}_episode{episode:02}_{i}.png'
                        )
                    )
                save_reward_path = os.path.join(self.logger._log_dir, "eval_reward")
                os.makedirs(save_reward_path, exist_ok=True)
                with open(
                    os.path.join(
                        save_reward_path,
                        f"step{self.step:07}_episode{episode:02}.pkl"
                    ), "wb"
                ) as f:
                    pkl.dump(rewards, f)

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate, self.step)
        self.logger.dump(self.step)

    def learn_reward(self, first_flag=0):
        # … your existing code, unchanged …
        pass

    def run(self):
        model_save_dir = os.path.join(self.work_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)

        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()

        interact_count = 0
        reward_learning_acc = 0
        vlm_acc = 0
        eval_cnt = 0

        while self.step < self.cfg.num_train_steps:
            if done:
                # … logging unchanged …

                # CHANGED: unified reset
                obs = self.env.reset()
                if "metaworld" in self.cfg.env:
                    obs = obs[0]
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                traj_images = []
                ep_info = []

            # action selection (unchanged) …
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # … reward learning / agent update logic unchanged …

            # Take a step
            try:
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated
            ep_info.append(extra)

            # … image collection for VLM (only drop 'softgym' branch) …
            if self.cfg.vlm_label or \
               self.reward in ['blip2_image_text_matching', 'clip_image_text_matching'] or \
               (self.cfg.image_reward and self.reward not in ["gt_task_reward", "sparse_task_reward"]):

                if "metaworld" in self.cfg.env:
                    rgb_image = self.env.render()[::-1, :, :]
                    if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                        rgb_image = rgb_image[100:400, 100:400, :]
                elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                    rgb_image = self.env.render(mode='rgb_array')
                else:
                    # CHANGED: unified rendering
                    rgb_image = self.env.render(mode='rgb_array')

                if self.cfg.image_reward and \
                   'Water' not in self.cfg.env and \
                   'Rope' not in self.cfg.env:
                    rgb_image = cv2.resize(
                        rgb_image, (self.image_height, self.image_width)
                    )
                traj_images.append(rgb_image)
            else:
                rgb_image = None

            # … reward_hat computation unchanged …

            # CHANGED: unified done_no_max
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done

            # … buffer adds, logging, saving unchanged …

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            if self.step % self.cfg.save_interval == 0 and self.step > 0:
                self.agent.save(model_save_dir, self.step)
                self.reward_model.save(model_save_dir, self.step)

        # final save (unchanged)
        self.agent.save(model_save_dir, self.step)
        self.reward_model.save(model_save_dir, self.step)


@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    if cfg.mode == 'eval':
        workspace.evaluate(save_additional=cfg.save_images)
        exit()
    workspace.run()


if __name__ == '__main__':
    main()
