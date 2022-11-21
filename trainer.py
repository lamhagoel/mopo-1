import time
import os

import numpy as np
import torch

from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        algo,
        eval_env,
        epoch,
        step_per_epoch,
        rollout_freq,
        logger,
        log_freq,
        eval_episodes=10,
        model_save_dir=None,
        dynamics_model_save_dir=None,
        save_freq=100
    ):
        self.algo = algo
        self.eval_env = eval_env

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._rollout_freq = rollout_freq

        self.logger = logger
        self._log_freq = log_freq
        self._eval_episodes = eval_episodes
        self.model_save_dir = model_save_dir
        self.dynamics_model_save_dir = dynamics_model_save_dir
        self.save_freq = save_freq

    def train_dynamics(self, saved_dir=None):
        if (saved_dir is not None):
            self.algo.dynamics_model = torch.load(saved_dir)
            return
        start_time = time.time()
#         self.algo.save_dynamics_model(
#             save_path=os.path.join(self.logger.writer.get_logdir(), "dynamics_model")
#         )
        self.algo.learn_dynamics()
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))
        torch.save(self.algo.dynamics_model,os.path.join(self.logger.writer.get_logdir(), "policyDynamics.pth"))
        if self.dynamics_model_save_dir is not None:
            torch.save(self.algo.dynamics_model,self.dynamics_model_save_dir + ".pth")

    def train_policy(self, saved_model_path=None, start_epoch=1):
        if saved_model_path is not None:
            self.algo.policy.load_state_dict(torch.load(saved_model_path))
        start_time = time.time()
        num_timesteps = 0
        # train loop
        for e in range(start_epoch, self._epoch + 1):
            self.algo.policy.train()
            with tqdm(total=self._step_per_epoch, desc=f"Epoch #{e}/{self._epoch}") as t:
                while t.n < t.total:
                    if num_timesteps % self._rollout_freq == 0:
                        self.algo.rollout_transitions()
                    # update policy by sac
                    loss = self.algo.learn_policy()
                    t.set_postfix(**loss)
                    # log
                    if num_timesteps % self._log_freq == 0:
                        for k, v in loss.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                    num_timesteps += 1
                    t.update(1)
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            self.logger.record("eval/episode_reward", ep_reward_mean, num_timesteps, printed=False)
            self.logger.record("eval/episode_length", ep_length_mean, num_timesteps, printed=False)
            self.logger.print(f"Epoch #{e}: episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}, episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
        
            # save policy
            torch.save(self.algo.policy.state_dict(), os.path.join(self.logger.writer.get_logdir(), "policy.pth"))
            if e % self.save_freq == 0 and self.model_save_dir is not None:
                torch.save(self.algo.policy.state_dict(), self.model_save_dir + str(e) + ".pth")
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))

    def _evaluate(self):
        self.algo.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
