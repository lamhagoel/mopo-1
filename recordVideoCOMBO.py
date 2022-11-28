import argparse
import datetime
import os
import random
import importlib

import gym
import d4rl

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.cql import CQLPolicy
from algo.combo import COMBO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer

from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--auto-beta', default=True)
    parser.add_argument('--target-entropy', type=int, default=-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--beta-lr', type=float, default=3e-4)
    parser.add_argument('--lagrange-threshold', type=float, default=0.0)
    parser.add_argument('--sample-from-previous', type=bool, default=False)
    parser.add_argument('--model-save-dir', type=str, default=None)
    parser.add_argument('--dynamics-model-save-dir', type=str, default=None)
    parser.add_argument('--saved-model-dir', type=str, default=None)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--save-freq', type=int, default=50)
    parser.add_argument('--output-log-file', type=str, default=None)
    parser.add_argument('--increase-exp-data', type=bool, default=True)
    parser.add_argument('--video-folder', type=str, default="recordings")
    parser.add_argument('--deterministic-sampling', type=bool, default=True)

    # dynamics model's arguments
    parser.add_argument("--dynamics-lr", type=float, default=0.001)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)
    parser.add_argument("--saved-dynamics-model-dir", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def load_model(algo):
    algo.policy.load_state_dict(torch.load(saved_model_path))

def eval_record_video(algo, video_folder, envName, deterministic):
    prefix = envName
    eval_env = DummyVecEnv([lambda: gym.make(envName)])
    num_steps = 1000

    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=num_steps,
                                name_prefix=prefix)

    obs = eval_env.reset()
    # env.render()
    algo.policy.eval()

    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs) 
        # action = eval_env.action_space.sample()
        action = algo.policy.sample_action(obs, deterministic=deterministic)
        
        # apply the action
        obs, reward, done, info = eval_env.step(action)
        
        # Render the env
        # env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        # time.sleep(0.001)
        
        # If the epsiode is up, then start another one
        if done:
            obs = eval_env.reset()

    # Close the env
    eval_env.close()

def train(args=get_args()):
    # print()
    # print("1 Beta: " + str(args.beta) + " Auto beta: " + str(args.auto_beta) + " " + str(type(args.beta)))
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env.seed(args.seed)

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer, args.output_log_file)

    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam([*critic1.parameters(), *critic2.parameters()], lr=args.critic_lr)
    # critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha is True:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # print("2 Beta: " + str(args.beta) + " Auto beta: " + str(args.auto_beta) + " " + str(type(args.beta)))
    if args.auto_beta is True:
        log_beta = torch.zeros(1, requires_grad=True, device=args.device)
        beta_optim = torch.optim.Adam([log_beta], lr=args.beta_lr)
        args.beta = (log_beta, beta_optim)

    # print("3 Beta: " + str(args.beta) + " Auto beta: " + str(args.auto_beta) + " " + str(type(args.beta)))

    # create policy
    cql_policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        lagrange_threshold=args.lagrange_threshold,
        sample_from_previous = args.sample_from_previous,
        device=args.device
    )

    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     lr=args.dynamics_lr,
                                     **config["transition_params"]
                                     )

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )

    # create COMBO algo
    algo = COMBO(
        cql_policy,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **config["combo_params"]
    )

    # create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        eval_episodes=args.eval_episodes,
        model_save_dir = args.model_save_dir,
        dynamics_model_save_dir = args.dynamics_model_save_dir,
        save_freq = args.save_freq
    )

    # pretrain dynamics model on the whole dataset
    trainer.train_dynamics(args.saved_dynamics_model_dir)

    # begin train
    load_model(algo, args.saved_model_dir)

    video_folder = args.video_folder
    envName = args.task

    eval_record_video(algo, video_folder, envName, args.deterministic_sampling)


if __name__ == "__main__":
    train()
