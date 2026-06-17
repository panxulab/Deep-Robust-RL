import copy
import math
import os
import sys

import numpy as np
import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F

server=os.path.isdir("/code")
if server: sys.path.append("/code/")
path="/output/" if server else "./"

from algorithm.util import ReplayBuffer,SquashedNormal,unpack_batch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor_con(nn.Module):

	def __init__(self,state_dim,action_dim,hidden_dim,init_w=3e-3):
		super().__init__()

		self.l1=nn.Linear(state_dim,hidden_dim)
		self.l2=nn.Linear(hidden_dim,hidden_dim)
		# self.l3=nn.Linear(hidden_dim,hidden_dim)
		self.l4=nn.Linear(hidden_dim,2*action_dim)

		self.l4.weight.data.uniform_(-init_w,init_w)
		self.l4.bias.data.uniform_(-init_w,init_w)

	def forward(self,state):
		LOG_STD_MAX=2
		LOG_STD_MIN=-20

		x=F.relu(self.l1(state))
		x=F.relu(self.l2(x))
		# x=F.relu(self.l3(x))
		mu,log_std=self.l4(x).chunk(2,dim=-1)

		log_std=torch.tanh(log_std)
		log_std=LOG_STD_MIN+0.5*(LOG_STD_MAX-LOG_STD_MIN)*(log_std+1)
		std=log_std.exp()

		dist=SquashedNormal(mu,std)
		return dist

class Actor_dis(nn.Module):

	def __init__(self,state_dim,action_num,hidden_dim,init_w=3e-3):
		super().__init__()

		self.l1=nn.Linear(state_dim,hidden_dim)
		self.l2=nn.Linear(hidden_dim,hidden_dim)
		# self.l3=nn.Linear(hidden_dim,hidden_dim)
		self.l4=nn.Linear(hidden_dim,action_num)

		self.l4.weight.data.uniform_(-init_w,init_w)
		self.l4.bias.data.uniform_(-init_w,init_w)

	def forward(self,state):
		z=F.relu(self.l1(state))
		z=F.relu(self.l2(z))
		# z=F.relu(self.l3(z))
		logits=self.l4(z)
		logits=torch.clamp(logits,min=-20,max=20)

		dist=td.Categorical(logits=logits)
		return dist

class Critic(nn.Module):

	def __init__(self,state_dim,action_dim,hidden_dim,init_w=3e-3):
		super().__init__()

		input_dim=state_dim+action_dim
		self.l11=nn.Linear(input_dim,hidden_dim)
		self.l12=nn.Linear(hidden_dim,hidden_dim)
		# self.l13=nn.Linear(hidden_dim,hidden_dim)
		self.l14=nn.Linear(hidden_dim,1)

		self.l14.weight.data.uniform_(-init_w,init_w)
		self.l14.bias.data.uniform_(-init_w,init_w)

		self.l21=nn.Linear(input_dim,hidden_dim)
		self.l22=nn.Linear(hidden_dim,hidden_dim)
		# self.l23=nn.Linear(hidden_dim,hidden_dim)
		self.l24=nn.Linear(hidden_dim,1)

		self.l24.weight.data.uniform_(-init_w,init_w)
		self.l24.bias.data.uniform_(-init_w,init_w)

	def forward(self,state,action):
		x=torch.cat([state,action],dim=-1)
		q1=F.relu(self.l11(x))
		q1=F.relu(self.l12(q1))
		# q1=F.relu(self.l13(q1))
		q1=self.l14(q1).squeeze(-1)

		q2=F.relu(self.l21(x))
		q2=F.relu(self.l22(q2))
		# q2=F.relu(self.l23(q2))
		q2=self.l24(q2).squeeze(-1)

		return q1,q2

class V_net(nn.Module):

	def __init__(self,state_dim,hidden_dim,init_w=3e-3):
		super().__init__()

		self.l1=nn.Linear(state_dim,hidden_dim)
		self.l2=nn.Linear(hidden_dim,hidden_dim)
		# self.l3=nn.Linear(hidden_dim,hidden_dim)
		self.l4=nn.Linear(hidden_dim,1)

		self.l4.weight.data.uniform_(-init_w,init_w)
		self.l4.bias.data.uniform_(-init_w,init_w)

	def forward(self,state):
		v=F.relu(self.l1(state))
		v=F.relu(self.l2(v))
		# v=F.relu(self.l3(v))
		v=self.l4(v).squeeze(-1)

		return v

class LSVI_SAC(nn.Module):

	def __init__(self,state_dim,action_dim,
			action_space,
			action_num=None,
			hidden_dim=256,
			batch_size=256,
			alpha=0.1,  # initial temperature
			gamma=0.99,  # discount factor
			tau=0.005,  # param update rate
			lr=3e-4,  # learning rate
			rho=None  # robust parameter
	):
		super().__init__()

		self.state_dim=state_dim
		self.action_dim=action_dim
		self.action_num=action_num
		self.action_space=action_space
		self.hidden_dim=hidden_dim
		self.batch_size=batch_size
		self.buffer=ReplayBuffer(state_dim=state_dim,action_dim=action_dim)
		self.entropy_target=-action_dim if action_num is None else math.log(action_num)*0.4

		assert rho

		self.gamma=gamma
		self.tau=tau
		self.H=1/(1-gamma)
		self.rho=rho

		if action_num is None:
			nlow=torch.from_numpy(self.action_space.low).to(device)
			nhigh=torch.from_numpy(self.action_space.high).to(device)
			self.trans_action=lambda action:(1-action)/2*nlow+(1+action)/2*nhigh

		if action_num is None:
			self.actor=Actor_con(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
		else:
			self.actor=Actor_dis(state_dim=state_dim,action_num=action_num,hidden_dim=hidden_dim).to(device)
		self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)

		self.critic=Critic(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
		self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

		self.V_net=V_net(state_dim=state_dim,hidden_dim=hidden_dim).to(device)
		self.V_net_optimizer=torch.optim.Adam(self.V_net.parameters(),lr=lr)
		self.V_net_target=copy.deepcopy(self.V_net).to(device)
		for param in self.V_net_target.parameters(): param.requires_grad=False

		self.log_alpha=torch.tensor(np.log(alpha),dtype=torch.float32,requires_grad=True,device=device)
		self.alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=lr)

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def get_action(self,state,explore=True):
		state=torch.from_numpy(state.astype(np.float32)).to(device)  # (state_dim)
		dist=self.actor(state)
		if self.action_num is None:
			action=dist.sample() if explore else dist.mean.detach()
			action=self.trans_action(action).cpu().numpy()
		else:
			action=dist.sample() if explore else torch.argmax(dist.probs).detach()
			action=self.action_space[action.item()].squeeze()
		return action

	def actor_step(self,batch):
		state=unpack_batch(batch)[0]
		dist=self.actor(state)

		if self.action_num is None:
			action=dist.rsample()
			log_prob=dist.log_prob(action).sum(dim=1)
			action=self.trans_action(action)

			q1,q2=self.critic(state,action)
			q_min=torch.clamp(torch.max(q1,q2),min=1e-6)

			actor_loss=(self.alpha.detach()*log_prob+self.rho*self.gamma*torch.log(q_min)).mean()
			alpha_loss=-(self.alpha*(log_prob.detach()+self.entropy_target)).mean()
		else:
			state=state.repeat_interleave(self.action_num,dim=0)
			action=torch.from_numpy(self.action_space).to(device).repeat(self.batch_size,1)

			q1,q2=self.critic(state,action)
			q_min=torch.clamp(torch.max(q1,q2),min=1e-6).reshape(self.batch_size,self.action_num)

			actor_loss=(dist.probs*(self.alpha.detach()*torch.log(dist.probs)+self.rho*self.gamma*torch.log(q_min.detach()))).sum(dim=-1).mean()
			alpha_loss=-(self.alpha*((dist.probs*torch.log(dist.probs)).sum(dim=-1).detach()+self.entropy_target)).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		return {"actor_loss":actor_loss.item(),"alpha":self.alpha.item(),"alpha_loss":alpha_loss.item()}

	def critic_step(self,batch):
		state,action,state_next,reward,done=unpack_batch(batch)
		reward,done=reward.squeeze(dim=1),done.squeeze(dim=1)

		with torch.no_grad():
			dist_new=self.actor(state)

			if self.action_num is None:
				action_new=dist_new.sample()
				log_prob=dist_new.log_prob(action_new).sum(dim=1)
				action_new=self.trans_action(action_new)

				q1_new,q2_new=self.critic(state,action_new)
				q_min=torch.clamp(torch.max(q1_new,q2_new),min=1e-6)
				v_target=self.rho*self.gamma*torch.log(q_min)+self.alpha*log_prob
			else:
				state_new=state.repeat_interleave(self.action_num,dim=0)
				action_new=torch.from_numpy(self.action_space).to(device).repeat(self.batch_size,1)

				q1_new,q2_new=self.critic(state_new,action_new)
				q_min=torch.clamp(torch.max(q1_new,q2_new),min=1e-6).reshape(self.batch_size,self.action_num)
				v_target=self.rho*self.gamma*torch.log(q_min)+self.alpha*torch.log(dist_new.probs)
				v_target=(dist_new.probs*v_target).sum(dim=-1)

		v0=self.V_net(state)
		v_loss=F.mse_loss(v0,v_target)
		self.V_net_optimizer.zero_grad()
		v_loss.backward()
		self.V_net_optimizer.step()

		q1,q2=self.critic(state,action)

		with torch.no_grad():
			v_next=self.V_net_target(state_next)
			q_target=torch.exp(-reward/self.rho/self.gamma)*torch.exp((1-done)/self.rho*v_next)

		q1_loss=F.mse_loss(q1,q_target)
		q2_loss=F.mse_loss(q2,q_target)

		loss=q1_loss+q2_loss
		self.critic_optimizer.zero_grad()
		loss.backward()
		self.critic_optimizer.step()

		return {"q1_loss":q1_loss.item(),"q2_loss":q2_loss.item(),"v":v0.mean().item(),"v_loss":v_loss.item()}

	def update_target(self,model,model_target,tau):
		with torch.no_grad():
			for param,param_target in zip(model.parameters(),model_target.parameters()):
				param_target.copy_(tau*param+(1-tau)*param_target)

	def train_batch(self):
		batch=self.buffer.sample(self.batch_size)
		critic_info=self.critic_step(batch)
		actor_info=self.actor_step(batch)
		self.update_target(self.V_net,self.V_net_target,self.tau)

		return {**critic_info,**actor_info}
