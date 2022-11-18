import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy


class CQLPolicy(nn.Module):
    def __init__(
        self, 
        actor, 
        critic1, 
        critic2,
        actor_optim, 
        critic_optim, 
        # critic2_optim,
        action_space,
        dist, 
        tau=0.005, 
        gamma=0.99, 
        alpha=0.2,
        beta=1.0,
        device="cpu"
    ):
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        # self.critic2_optim = critic2_optim

        self.action_space = action_space
        self.dist = dist

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._is_auto_beta = False
        if isinstance(beta, tuple):
            self._is_auto_beta = True
            self._log_beta, self._beta_optim = beta
            self._beta = self._log_beta.detach().exp()
        else:
            self._beta = beta
        
        self.__eps = np.finfo(np.float32).eps.item()

        self._device = device
    
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
    
    def _sync_weight(self):
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def forward(self, obs, deterministic=False):
        dist = self.actor.get_dist(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        return squashed_action, log_prob

    def get_sampled_actions(self, obs, num_samples):
        dist = self.actor.get_dist(obs)
        sampled_actions = torch.stack([dist.rsample() for _ in range(num_samples_for_estimation)], dim=0)
        log_prob = dist.log_prob(sampled_actions)

        action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        squashed_actions = torch.tanh(sampled_actions)
        log_prob = log_prob - torch.log(action_scale * (1 - sampled_actions.pow(2)) + self.__eps).sum(-1, keepdim=True)
        return sampled_actions, log_prob

    def sample_action(self, obs, deterministic=False):
        action, _ = self(obs, deterministic)
        return action.cpu().detach().numpy()

    def learn(self, data):
        obs, actions, next_obs, terminals, rewards = data["observations"], \
            data["actions"], data["next_observations"], data["terminals"], data["rewards"]
        
        rewards = torch.as_tensor(rewards).to(self._device)
        terminals = torch.as_tensor(terminals).to(self._device)
        
        # update critic
        q1, q2 = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, next_log_probs = self(next_obs)
            next_q = torch.min(
                self.critic1_old(next_obs, next_actions), self.critic2_old(next_obs, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        # self.critic1_optim.zero_grad()
        # critic1_loss.backward()
        # self.critic1_optim.step()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # Add the penalty term for conservative estimate
        num_samples_for_estimation = 10
        random_actions_shape = list(torch.as_tensor(actions).to(self._device).unsqueeze(0).shape)
        random_actions_shape[0] = num_samples_for_estimation

        random_actions = torch.rand(random_actions_shape).to(torch.as_tensor(actions).to(self._device)) * 2 - 1
        # action_dist1, sampled_log_prob = self(obs)
        sampled_actions, sampled_log_prob = self.get_sampled_actions(obs, num_samples_for_estimation)
        # sampled_actions = torch.stack([action_dist.rsample() for _ in range(num_samples_for_estimation)], dim=0)
        # print()
        # print(str(action_dist1.shape) + " " + str(sampled_log_prob.shape) + " " + str(action_dist.shape) + " " + str(sampled_actions.shape))

        random_next_actions = torch.rand(random_actions_shape).to(torch.as_tensor(actions).to(self._device)) * 2 - 1
        # next_action_dist, sampled_next_log_prob = self(next_obs)
        sampled_next_actions, sampled_next_log_prob = self.get_sampled_actions(obs, num_samples_for_estimation)

        # next_action_dist = self.get_actions_dist(next_obs)
        # sampled_next_actions = torch.stack([next_action_dist.rsample() for _ in range(num_samples_for_estimation)], dim=0)

        sampled_actions = torch.cat([random_actions, sampled_actions], dim=0)
        repeated_obs = torch.repeat_interleave(torch.as_tensor(obs).to(self._device).unsqueeze(0), sampled_actions.shape[0], 0)
        sampled_actions = sampled_actions.reshape((-1,sampled_actions.shape[-1]))
        repeated_obs = repeated_obs.reshape((-1,repeated_obs.shape[-1]))
        print()
        print(str(obs.shape) + " " + str(actions.shape))
        print(str(random_actions_shape) + " " + str(random_actions.shape) + " " + str(sampled_actions.shape) + " " + str(random_next_actions.shape) + " " + str(sampled_next_actions.shape) + " " + str(sampled_actions.shape) + " " + str(repeated_obs.shape))
        sampled_q1 = self.critic1(repeated_obs, sampled_actions)
        sampled_q2 = self.critic2(repeated_obs, sampled_actions)

        sampled_next_actions = torch.cat([random_next_actions, sampled_next_actions], dim=0)
        repeated_next_obs = torch.repeat_interleave(torch.as_tensor(next_obs).to(self._device).unsqueeze(0), sampled_next_actions.shape[0], 0)
        sampled_next_actions = sampled_next_actions.reshape((-1,sampled_next_actions.shape[-1]))
        repeated_next_obs = repeated_next_obs.reshape((-1,repeated_next_obs.shape[-1]))
        sampled_next_q1 = self.critic1(repeated_next_obs, sampled_next_actions)
        sampled_next_q2 = self.critic2(repeated_next_obs, sampled_next_actions)

        sampled_q1 = torch.cat([sampled_q1, sampled_next_q1], dim=0)
        sampled_q2 = torch.cat([sampled_q2, sampled_next_q2], dim=0) 

        #importance sampling
        _random_log_prob = torch.ones(num_samples_for_estimation*actions.shape[0], 1).to(sampled_q1) * actions.shape[-1] * np.log(0.5)
        sampling_weight = torch.cat([_random_log_prob, sampled_log_prob, _random_log_prob, sampled_next_log_prob], dim=0)
        sampled_q1 = sampled_q1 - sampling_weight
        sampled_q2 = sampled_q2 - sampling_weight

        q1_penalty = (torch.logsumexp(sampled_q1, dim=0) - q1) #* self.args['base_beta']
        q2_penalty = (torch.logsumexp(sampled_q2, dim=0) - q2) #* self.args['base_beta']

        # update beta
        lagrange_threshold = 10.0
        beta_loss = - torch.mean(torch.exp(self.log_beta) * (q1_penalty - lagrange_threshold).detach()) - torch.mean(torch.exp(self.log_beta) * (q2_penalty - lagrange_threshold).detach())
        self.log_beta_optim.zero_grad()
        beta_loss.backward()
        self.log_beta_optim.step()

        q1_penalty = q1_penalty * torch.exp(self.log_beta)
        q2_penalty = q2_penalty * torch.exp(self.log_beta)

        critic_loss = critic1_loss + critic2_loss + torch.mean(q1_penalty) + torch.mean(q2_penalty)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        a, log_probs = self(obs)
        q1a, q2a = self.critic1(obs, a).flatten(), self.critic2(obs, a).flatten()
        actor_loss = (self._alpha * log_probs.flatten() - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        if self._is_auto_beta:
            result["loss/beta"] = beta_loss.item()
            result["beta"] = self._beta.item()
        
        return result

