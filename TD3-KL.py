import copy
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

server=os.path.isdir("/code")
if server: sys.path.append("/code/")
path="/output/" if server else "./"

from algorithm.util import ReplayBuffer,unpack_batch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(layer,std=np.sqrt(2),bias=0.0):
	nn.init.orthogonal_(layer.weight,std)
	nn.init.constant_(layer.bias,bias)

class Actor_con(nn.Module):

	def __init__(self,state_dim,action_dim,hidden_dim):
		super().__init__()

		self.l1=nn.Linear(state_dim,hidden_dim)
		self.l2=nn.Linear(hidden_dim,hidden_dim)
		self.l3=nn.Linear(hidden_dim,action_dim)

		init_weights(self.l1)
		init_weights(self.l2)
		init_weights(self.l3,0.01)

	def forward(self,state):
		x=F.relu(self.l1(state))
		x=F.relu(self.l2(x))
		mu=torch.tanh(self.l3(x))
		return mu

class Critic(nn.Module):

	def __init__(self,state_dim,action_dim,hidden_dim):
		super().__init__()

		input_dim=state_dim+action_dim
		self.l11=nn.Linear(input_dim,hidden_dim)
		self.l12=nn.Linear(hidden_dim,hidden_dim)
		self.l13=nn.Linear(hidden_dim,1)

		self.l21=nn.Linear(input_dim,hidden_dim)
		self.l22=nn.Linear(hidden_dim,hidden_dim)
		self.l23=nn.Linear(hidden_dim,1)

		init_weights(self.l11)
		init_weights(self.l12)
		init_weights(self.l13,0.01)

		init_weights(self.l21)
		init_weights(self.l22)
		init_weights(self.l23,0.01)

	def forward(self,state,action):
		x=torch.cat([state,action],dim=-1)
		q1=F.relu(self.l11(x))
		q1=F.relu(self.l12(q1))
		q1=self.l13(q1).squeeze(-1)

		q2=F.relu(self.l21(x))
		q2=F.relu(self.l22(q2))
		q2=self.l23(q2).squeeze(-1)

		return q1,q2

class TD3(nn.Module):

	def __init__(self,state_dim,action_dim,
			action_space,
			hidden_dim=256,
			batch_size=256,
			gamma=0.99,  # discount factor
			noise_actor=0.2,
			noise_explore=0.1,
			noise_clip=0.5,
			update_policy=2,
			tau=0.005,  # param update rate
			lr=3e-4,  # learning rate
			rho=None  # robust parameter
	):
		super().__init__()

		self.state_dim=state_dim
		self.action_dim=action_dim
		self.action_space=action_space
		self.hidden_dim=hidden_dim
		self.batch_size=batch_size
		self.buffer=ReplayBuffer(state_dim=state_dim,action_dim=action_dim)

		self.noise_actor=noise_actor
		self.noise_explore=torch.tensor(noise_explore).to(device)
		self.noise_clip=noise_clip
		self.update_policy=update_policy

		self.gamma=gamma
		self.tau=tau
		self.H=1/(1-gamma)
		self.rho=rho

		nlow=torch.from_numpy(self.action_space.low).to(device)
		nhigh=torch.from_numpy(self.action_space.high).to(device)
		self.trans_action=lambda action:(1-action)/2*nlow+(1+action)/2*nhigh

		self.actor=Actor_con(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
		self.actor_target=copy.deepcopy(self.actor)
		self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)
		for param in self.actor_target.parameters(): param.requires_grad=False

		self.critic=Critic(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
		self.critic_target=copy.deepcopy(self.critic)
		self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)
		for param in self.critic_target.parameters(): param.requires_grad=False

	def get_action(self,state,explore=True):
		state=torch.from_numpy(state.astype(np.float32)).to(device)  # (state_dim)
		action=self.actor(state).detach()
		if explore:
			noise=torch.randn_like(action)*self.noise_explore
			action=(action+noise).clamp(-1,1)
		action=self.trans_action(action).cpu().numpy()
		return action

	def actor_step(self,batch):
		state=unpack_batch(batch)[0]
		action=self.trans_action(self.actor(state))

		q1,q2=self.critic(state,action)
		q_max=torch.clamp(torch.max(q1,q2),min=1e-6)

		actor_loss=(self.rho*self.gamma*torch.log(q_max)).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		return {"actor_loss":actor_loss.item()}

	def critic_step(self,batch):
		state,action,state_next,reward,done=unpack_batch(batch)
		reward,done=reward.squeeze(dim=1),done.squeeze(dim=1)
		q1,q2=self.critic(state,action)

		with torch.no_grad():
			action_next=self.actor_target(state_next)
			noise=torch.randn_like(action_next)*self.noise_actor
			noise=noise.clamp(-self.noise_clip,self.noise_clip)
			action_next=self.trans_action((action_next+noise).clamp(-1,1))

			q1_next,q2_next=self.critic_target(state_next,action_next)
			q_max=torch.clamp(torch.max(q1_next,q2_next),min=1e-6)

			v_next=self.rho*self.gamma*torch.log(q_max)
			q_target=torch.exp(-reward/self.rho/self.gamma)*torch.exp((1-done)/self.rho*v_next)

		q1_loss=F.mse_loss(q1,q_target)
		q2_loss=F.mse_loss(q2,q_target)

		loss=q1_loss+q2_loss
		self.critic_optimizer.zero_grad()
		loss.backward()
		self.critic_optimizer.step()

		return {"q1":q1.mean().item(),"q2":q2.mean().item(),"q1_loss":q1_loss.item(),"q2_loss":q2_loss.item()}

	def update_target(self,model,model_target,tau):
		with torch.no_grad():
			for param,param_target in zip(model.parameters(),model_target.parameters()):
				param_target.copy_(tau*param+(1-tau)*param_target)

	def train_batch(self,update_actor=False):
		batch=self.buffer.sample(self.batch_size)
		critic_info=self.critic_step(batch)
		actor_info={}
		if update_actor:
			actor_info=self.actor_step(batch)
			self.update_target(self.actor,self.actor_target,self.tau)
			self.update_target(self.critic,self.critic_target,self.tau)

		return {**critic_info,**actor_info}
