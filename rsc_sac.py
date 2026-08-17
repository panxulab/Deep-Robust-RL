#!/usr/bin/env python3
"""Single-file RSC-SAC reproduction.

A compact implementation of the core method from:
"Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation".

Examples:
    python rsc_sac_single.py --steps 50000
    python rsc_sac_single.py --augment-ratio 0       # plain SAC baseline
    python rsc_sac_single.py --test                  # lightweight self-tests
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Toy environment with a spurious train/test correlation
# -----------------------------------------------------------------------------


class SpuriousGoalEnv:
	"""Small continuous-control task for testing distribution-shift robustness.

	Observation is ``[position, 0.25 * goal, shortcut]``. During training,
	``shortcut == goal``; under test shift, ``shortcut == -goal``. The shortcut
	affects neither the dynamics nor the reward.
	"""

	obs_dim=3
	action_dim=1

	def __init__(self,shifted: bool = False,horizon: int = 100,seed: int = 0) -> None:
		self.shifted=shifted
		self.horizon=horizon
		self.rng=np.random.default_rng(seed)
		self.position=0.0
		self.goal=1.0
		self.step_count=0

	def reset(self) -> np.ndarray:
		self.position=float(self.rng.uniform(-0.2,0.2))
		self.goal=float(self.rng.choice([-1.0,1.0]))
		self.step_count=0
		return self._observation()

	def step(self,action: np.ndarray) -> tuple[np.ndarray,float,bool]:
		control=float(np.clip(action[0],-1.0,1.0))
		self.position=float(np.clip(self.position+0.12*control,-1.5,1.5))
		self.step_count+=1

		reward=1.0-abs(self.position-self.goal)
		done=self.step_count>=self.horizon
		return self._observation(),reward,done

	def _observation(self) -> np.ndarray:
		shortcut=-self.goal if self.shifted else self.goal
		return np.array([self.position,0.25*self.goal,shortcut],dtype=np.float32)

# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------


@dataclass
class Batch:
	obs: torch.Tensor
	action: torch.Tensor
	reward: torch.Tensor
	next_obs: torch.Tensor
	done: torch.Tensor

class ReplayBuffer:
	def __init__(self,capacity: int,obs_dim: int,action_dim: int) -> None:
		self.capacity=capacity
		self.obs=np.empty((capacity,obs_dim),dtype=np.float32)
		self.action=np.empty((capacity,action_dim),dtype=np.float32)
		self.reward=np.empty((capacity,1),dtype=np.float32)
		self.next_obs=np.empty((capacity,obs_dim),dtype=np.float32)
		self.done=np.empty((capacity,1),dtype=np.float32)
		self.size=0
		self.index=0

	def add(
			self,
			obs: np.ndarray,
			action: np.ndarray,
			reward: float,
			next_obs: np.ndarray,
			done: bool,
	) -> None:
		i=self.index
		self.obs[i]=obs
		self.action[i]=action
		self.reward[i]=reward
		self.next_obs[i]=next_obs
		self.done[i]=done
		self.index=(i+1)%self.capacity
		self.size=min(self.size+1,self.capacity)

	def sample(self,batch_size: int,device: torch.device) -> Batch:
		if self.size<batch_size:
			raise ValueError(f"buffer has {self.size} samples, need {batch_size}")

		indices=np.random.randint(0,self.size,size=batch_size)

		def tensor(array: np.ndarray) -> torch.Tensor:
			return torch.as_tensor(array[indices],device=device)

		return Batch(
			obs=tensor(self.obs),
			action=tensor(self.action),
			reward=tensor(self.reward),
			next_obs=tensor(self.next_obs),
			done=tensor(self.done),
		)

# -----------------------------------------------------------------------------
# RSC state perturbation: paper Eq. (7)
# -----------------------------------------------------------------------------


def perturb_state_dimensions(
		states: torch.Tensor,
		ratio: float,
		eps: float = 1e-6,
) -> torch.Tensor:
	"""Replace one dimension in selected states using a semantically close donor.

	For each selected row, a random state dimension is chosen. The donor is the
	sample with a large difference in that dimension and small differences in all
	remaining dimensions.
	"""
	if states.ndim!=2:
		raise ValueError("states must have shape [batch, state_dim]")
	if not 0.0<=ratio<=1.0:
		raise ValueError("ratio must lie in [0, 1]")

	batch_size,state_dim=states.shape
	if batch_size<2 or ratio==0.0:
		return states.clone()

	result=states.clone()
	selected_rows=(torch.rand(batch_size,device=states.device)<ratio).nonzero().flatten()

	for row in selected_rows.tolist():
		dim=int(torch.randint(state_dim,(),device=states.device))
		difference=states-states[row]
		target_difference=difference[:,dim].square()
		other_difference=difference.square().sum(dim=1)-target_difference
		score=target_difference/(other_difference+eps)
		score[row]=-torch.inf
		donor=int(score.argmax())
		result[row,dim]=states[donor,dim]

	return result

# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------


def mlp(sizes: Iterable[int],activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
	sizes=list(sizes)
	layers: list[nn.Module]=[]
	for input_dim,output_dim in zip(sizes[:-2],sizes[1:-1]):
		layers.extend((nn.Linear(input_dim,output_dim),activation()))
	layers.append(nn.Linear(sizes[-2],sizes[-1]))
	return nn.Sequential(*layers)

class GaussianActor(nn.Module):
	def __init__(self,obs_dim: int,action_dim: int,hidden_dim: int = 256) -> None:
		super().__init__()
		self.network=mlp([obs_dim,hidden_dim,hidden_dim,2*action_dim])

	def forward(self,obs: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
		mean,log_std=self.network(obs).chunk(2,dim=-1)
		return mean,log_std.clamp(-5.0,2.0)

	def sample(self,obs: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
		mean,log_std=self(obs)
		distribution=torch.distributions.Normal(mean,log_std.exp())
		raw_action=distribution.rsample()
		action=raw_action.tanh()

		# Numerically stable tanh change-of-variables correction.
		correction=2.0*(
				math.log(2.0)-raw_action-F.softplus(-2.0*raw_action)
		)
		log_prob=(distribution.log_prob(raw_action)-correction).sum(-1,keepdim=True)
		return action,log_prob

	@torch.no_grad()
	def deterministic(self,obs: torch.Tensor) -> torch.Tensor:
		return self(obs)[0].tanh()

class TwinCritic(nn.Module):
	def __init__(self,obs_dim: int,action_dim: int,hidden_dim: int = 256) -> None:
		super().__init__()
		sizes=[obs_dim+action_dim,hidden_dim,hidden_dim,1]
		self.q1=mlp(sizes)
		self.q2=mlp(sizes)

	def forward(self,obs: torch.Tensor,action: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
		inputs=torch.cat((obs,action),dim=-1)
		return self.q1(inputs),self.q2(inputs)

class CausalTransitionModel(nn.Module):
	"""Learnable sparse causal graph for next-state and reward prediction.

	Each scalar in ``[state, action]`` is independently encoded by one shared
	encoder. A Gumbel-Softmax graph connects input dimensions to the dimensions
	of ``[next_state, reward]``. One shared decoder predicts each output scalar.
	"""

	def __init__(
			self,
			state_dim: int,
			action_dim: int,
			feature_dim: int = 64,
			position_dim: int = 16,
			hidden_dim: int = 128,
			temperature: float = 1.0,
	) -> None:
		super().__init__()
		self.input_dim=state_dim+action_dim
		self.output_dim=state_dim+1
		self.temperature=temperature

		self.input_position=nn.Parameter(torch.randn(self.input_dim,position_dim)*0.02)
		self.output_position=nn.Parameter(torch.randn(self.output_dim,position_dim)*0.02)
		self.encoder=mlp([1+position_dim,hidden_dim,feature_dim])
		self.decoder=mlp([feature_dim+position_dim,hidden_dim,1])
		self.graph_logits=nn.Parameter(torch.zeros(self.input_dim,self.output_dim,2))

	def graph(self,hard: bool|None = None) -> torch.Tensor:
		hard=self.training if hard is None else hard
		if self.training:
			return F.gumbel_softmax(
				self.graph_logits,
				tau=self.temperature,
				hard=hard,
				dim=-1,
			)[...,1]
		return self.graph_logits.softmax(dim=-1)[...,1]

	def forward(
			self,
			state: torch.Tensor,
			action: torch.Tensor,
			hard_graph: bool|None = None,
	) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
		inputs=torch.cat((state,action),dim=-1)
		batch_size=inputs.shape[0]

		input_position=self.input_position.unsqueeze(0).expand(batch_size,-1,-1)
		encoded=self.encoder(torch.cat((inputs.unsqueeze(-1),input_position),dim=-1))
		graph=self.graph(hard_graph)
		aggregated=torch.einsum("bif,io->bof",encoded,graph)

		output_position=self.output_position.unsqueeze(0).expand(batch_size,-1,-1)
		prediction=self.decoder(torch.cat((aggregated,output_position),dim=-1)).squeeze(-1)
		return prediction[:,:-1],prediction[:,-1:],graph

	def loss(
			self,
			state: torch.Tensor,
			action: torch.Tensor,
			next_state: torch.Tensor,
			reward: torch.Tensor,
			sparsity_weight: float,
			sparsity_p: float = 0.1,
	) -> tuple[torch.Tensor,dict[str,float]]:
		predicted_state,predicted_reward,graph=self(state,action)
		prediction_loss=F.mse_loss(predicted_state,next_state)+F.mse_loss(
			predicted_reward,reward
		)
		sparsity_loss=(graph.abs()+1e-8).pow(sparsity_p).mean()
		total_loss=prediction_loss+sparsity_weight*sparsity_loss
		return total_loss,{
			"model_loss":float(prediction_loss.detach()),
			"graph_density":float(graph.detach().mean()),
		}

# -----------------------------------------------------------------------------
# RSC-SAC agent
# -----------------------------------------------------------------------------


@dataclass
class SacConfig:
	gamma: float=0.99
	tau: float=0.005
	alpha: float=0.1
	actor_lr: float=3e-4
	critic_lr: float=1e-3
	model_lr: float=3e-4
	hidden_dim: int=256
	augment_ratio: float=0.5
	graph_sparsity: float=0.1
	graph_p: float=0.1

class RscSacAgent:
	def __init__(
			self,
			obs_dim: int,
			action_dim: int,
			config: SacConfig,
			device: str|torch.device = "cpu",
	) -> None:
		self.config=config
		self.device=torch.device(device)

		self.actor=GaussianActor(obs_dim,action_dim,config.hidden_dim).to(self.device)
		self.critic=TwinCritic(obs_dim,action_dim,config.hidden_dim).to(self.device)
		self.target_critic=TwinCritic(obs_dim,action_dim,config.hidden_dim).to(self.device)
		self.target_critic.load_state_dict(self.critic.state_dict())
		self.transition_model=CausalTransitionModel(obs_dim,action_dim).to(self.device)

		self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=config.actor_lr)
		self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=config.critic_lr)
		self.model_optimizer=torch.optim.Adam(
			self.transition_model.parameters(),lr=config.model_lr
		)

	@torch.no_grad()
	def act(self,obs: np.ndarray,deterministic: bool = False) -> np.ndarray:
		obs_tensor=torch.as_tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
		action=(
			self.actor.deterministic(obs_tensor)
			if deterministic
			else self.actor.sample(obs_tensor)[0]
		)
		return action.squeeze(0).cpu().numpy()

	def update(self,batch: Batch) -> dict[str,float]:
		model_loss,metrics=self.transition_model.loss(
			batch.obs,
			batch.action,
			batch.next_obs,
			batch.reward,
			self.config.graph_sparsity,
			self.config.graph_p,
		)
		self.model_optimizer.zero_grad(set_to_none=True)
		model_loss.backward()
		self.model_optimizer.step()

		augmented_batch=self._augment(batch)
		metrics["critic_loss"]=self._update_critic(augmented_batch)
		metrics["actor_loss"]=self._update_actor(augmented_batch.obs)
		self._soft_update_target()
		return metrics

	@torch.no_grad()
	def _augment(self,batch: Batch) -> Batch:
		modified_obs=perturb_state_dimensions(batch.obs,self.config.augment_ratio)
		changed=(modified_obs!=batch.obs).any(dim=1,keepdim=True)
		predicted_next_obs,predicted_reward,_=self.transition_model(
			modified_obs,batch.action,hard_graph=False
		)
		return Batch(
			obs=torch.where(changed,modified_obs,batch.obs),
			action=batch.action,
			reward=torch.where(changed,predicted_reward,batch.reward),
			next_obs=torch.where(changed,predicted_next_obs,batch.next_obs),
			done=batch.done,
		)

	def _update_critic(self,batch: Batch) -> float:
		with torch.no_grad():
			next_action,next_log_prob=self.actor.sample(batch.next_obs)
			target_q1,target_q2=self.target_critic(batch.next_obs,next_action)
			target_q=torch.min(target_q1,target_q2)-self.config.alpha*next_log_prob
			backup=batch.reward+self.config.gamma*(1.0-batch.done)*target_q

		q1,q2=self.critic(batch.obs,batch.action)
		loss=F.mse_loss(q1,backup)+F.mse_loss(q2,backup)
		self.critic_optimizer.zero_grad(set_to_none=True)
		loss.backward()
		self.critic_optimizer.step()
		return float(loss.detach())

	def _update_actor(self,obs: torch.Tensor) -> float:
		action,log_prob=self.actor.sample(obs)
		q1,q2=self.critic(obs,action)
		loss=(self.config.alpha*log_prob-torch.min(q1,q2)).mean()
		self.actor_optimizer.zero_grad(set_to_none=True)
		loss.backward()
		self.actor_optimizer.step()
		return float(loss.detach())

	@torch.no_grad()
	def _soft_update_target(self) -> None:
		for target,source in zip(self.target_critic.parameters(),self.critic.parameters()):
			target.mul_(1.0-self.config.tau).add_(source,alpha=self.config.tau)

# -----------------------------------------------------------------------------
# Training, evaluation
# -----------------------------------------------------------------------------

def train(args: argparse.Namespace) -> RscSacAgent:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	env=SpuriousGoalEnv(seed=args.seed)
	config=SacConfig(
		augment_ratio=args.augment_ratio,
		graph_sparsity=args.graph_sparsity,
	)
	agent=RscSacAgent(env.obs_dim,env.action_dim,config,args.device)
	replay=ReplayBuffer(args.buffer_size,env.obs_dim,env.action_dim)

	obs=env.reset()
	latest_metrics: dict[str,float]={}

	for step in range(1,args.steps+1):
		if step<=args.warmup:
			action=np.random.uniform(-1.0,1.0,env.action_dim).astype(np.float32)
		else:
			action=agent.act(obs)

		next_obs,reward,done=env.step(action)
		replay.add(obs,action,reward,next_obs,done)
		obs=env.reset() if done else next_obs

		if step>args.warmup and replay.size>=args.batch_size:
			for _ in range(args.updates_per_step):
				latest_metrics=agent.update(replay.sample(args.batch_size,agent.device))

		if step%args.log_every==0 or step==args.steps:
			nominal_return=evaluate(agent,shifted=False,episodes=args.eval_episodes)
			shifted_return=evaluate(agent,shifted=True,episodes=args.eval_episodes)
			graph_density=latest_metrics.get("graph_density",float("nan"))
			print(
				f"step={step:>7} "
				f"nominal={nominal_return:7.2f} "
				f"shifted={shifted_return:7.2f} "
				f"graph_density={graph_density:.3f}"
			)

	return agent

def evaluate(agent: RscSacAgent,shifted: bool,episodes: int = 20) -> float:
	env=SpuriousGoalEnv(shifted=shifted,seed=10_000+int(shifted))
	returns: list[float]=[]

	for _ in range(episodes):
		obs,done,episode_return=env.reset(),False,0.0
		while not done:
			obs,reward,done=env.step(agent.act(obs,deterministic=True))
			episode_return+=reward
		returns.append(episode_return)

	return float(np.mean(returns))

def parse_args() -> argparse.Namespace:
	parser=argparse.ArgumentParser(description="Single-file minimal RSC-SAC reproduction")
	parser.add_argument("--steps",type=int,default=100_000)
	parser.add_argument("--warmup",type=int,default=1_000)
	parser.add_argument("--batch-size",type=int,default=256)
	parser.add_argument("--buffer-size",type=int,default=100_000)
	parser.add_argument("--updates-per-step",type=int,default=10)
	parser.add_argument("--augment-ratio",type=float,default=0.5)
	parser.add_argument("--graph-sparsity",type=float,default=0.1)
	parser.add_argument("--log-every",type=int,default=5_000)
	parser.add_argument("--eval-episodes",type=int,default=20)
	parser.add_argument("--seed",type=int,default=0)
	parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--test",action="store_true",help="run lightweight self-tests")
	return parser.parse_args()

def main() -> None:
	args=parse_args()
	train(args)

if __name__=="__main__":
	main()
