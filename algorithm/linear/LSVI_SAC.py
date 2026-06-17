import copy
import math
import os
import sys
from itertools import chain

import numpy as np
import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F

server=os.path.isdir("/code")
if server: sys.path.append("/code/")
path="/output/" if server else "./"

from algorithm.util import ReplayBuffer,SquashedNormal,unpack_batch,weight_init
from VAE.dis_VAE import Decoder,Encoder,Feature

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_state(state):
	return state/np.array([4.8,5.0,0.4,5.0])

class Actor_con(nn.Module):

	def __init__(self,state_dim,action_dim,hidden_dim):
		super().__init__()

		self.l1=nn.Linear(state_dim,hidden_dim)
		self.l2=nn.Linear(hidden_dim,hidden_dim)
		self.l3=nn.Linear(hidden_dim,2*action_dim)

		self.apply(weight_init)

	def forward(self,state):
		LOG_STD_MAX=2.0
		LOG_STD_MIN=-5.0

		x=F.relu(self.l1(state))
		x=F.relu(self.l2(x))
		mu,log_std=self.l3(x).chunk(2,dim=-1)

		log_std=torch.tanh(log_std)
		log_std=LOG_STD_MIN+0.5*(LOG_STD_MAX-LOG_STD_MIN)*(log_std+1)
		std=log_std.exp()

		dist=SquashedNormal(mu,std)
		return dist

class Actor_dis(nn.Module):

	def __init__(self,state_dim,action_dim,hidden_dim):
		super().__init__()

		self.l1=nn.Linear(state_dim,hidden_dim)
		self.l2=nn.Linear(hidden_dim,hidden_dim)
		self.l3=nn.Linear(hidden_dim,action_dim)

		self.apply(weight_init)

	def forward(self,state):
		z=F.relu(self.l1(state))
		z=F.relu(self.l2(z))
		logits=self.l3(z)
		logits=torch.clamp(logits,min=-20,max=20)

		dist=td.Categorical(logits=logits)
		return dist

class Critic(nn.Module):

	def __init__(self,feature_dim,hidden_dim):
		super().__init__()

		self.l11=nn.Linear(feature_dim,hidden_dim)
		self.l12=nn.Linear(hidden_dim,hidden_dim)
		self.l13=nn.Linear(hidden_dim,1)

		self.l21=nn.Linear(feature_dim,hidden_dim)
		self.l22=nn.Linear(hidden_dim,hidden_dim)
		self.l23=nn.Linear(hidden_dim,1)

		self.apply(weight_init)

	def forward(self,feature):
		q1=F.relu(self.l11(feature))
		q1=F.relu(self.l12(q1))
		q1=self.l13(q1).squeeze(-1)

		q2=F.relu(self.l21(feature))
		q2=F.relu(self.l22(q2))
		q2=self.l23(q2).squeeze(-1)

		return q1,q2

class LSVI_VAE(nn.Module):

	def __init__(self,state_dim,action_dim,feature_dim,
			action_space,
			action_num=None,
			hidden_dim=256,
			batch_size=512,
			eta=3e-3,  # entropy parameter
			alpha=0.1,  # initial temperature
			gamma=0.99,  # discount factor
			tau_critic=0.01,  # param update rate
			tau_feature=0.005,  # param update rate
			feature_steps=1,  # feature update steps
			lr=3e-4,  # learning rate
			rho=None  # robust parameter
	):
		super().__init__()

		self.state_dim=state_dim
		self.action_dim=action_dim
		self.feature_dim=feature_dim
		self.action_num=action_num
		self.action_space=action_space
		self.hidden_dim=hidden_dim
		self.batch_size=batch_size
		self.buffer=ReplayBuffer(state_dim=state_dim,action_dim=action_dim)
		self.entropy_target=-action_dim if action_num is None else math.log(action_num)*0.4

		self.eta=eta
		self.gamma=gamma
		self.rho=rho
		self.H=1/(1-gamma)
		self.tau_critic=tau_critic
		self.tau_feature=tau_feature
		self.feature_steps=feature_steps

		if action_num is None:
			self.actor=Actor_con(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
		else:
			self.actor=Actor_dis(state_dim=state_dim,action_dim=action_num,hidden_dim=hidden_dim).to(device)
		self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)
		self.critic=Critic(feature_dim=feature_dim,hidden_dim=hidden_dim).to(device)
		self.critic_target=copy.deepcopy(self.critic)
		self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

		self.log_alpha=torch.tensor(np.log(alpha),requires_grad=True,device=device)
		self.alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=lr)

		self.category_size=1
		self.class_size=feature_dim

		self.encoder=Encoder(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim,category_size=1,class_size=feature_dim).to(device)
		self.decoder=Decoder(state_dim=state_dim,hidden_dim=hidden_dim,category_size=1,class_size=feature_dim).to(device)
		self.feature=Feature(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim,category_size=1,class_size=feature_dim).to(device)
		self.feature_target=copy.deepcopy(self.feature)
		feature_params=chain(self.encoder.parameters(),self.decoder.parameters(),self.feature.parameters())
		self.feature_optimizer=torch.optim.Adam(feature_params,lr=lr)

		for param in self.critic_target.parameters(): param.requires_grad=False
		for param in self.feature_target.parameters(): param.requires_grad=False

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def get_action(self,state,explore=True):
		state=norm_state(state).astype(np.float32)
		state=torch.from_numpy(state).to(device)  # (state_dim)
		dist=self.actor(state)
		if self.action_num is None:
			action=dist.sample() if explore else dist.mean.detach()
			action=action.item()
		else:
			action=dist.sample() if explore else torch.argmax(dist.probs).detach()
			action=self.action_space[action.item()][0]
		return action

	def get_feature(self,state,action):
		feature=self.feature_target.get_feature(state,action)
		return feature

	def actor_step(self,batch):
		state=unpack_batch(batch)[0]
		dist=self.actor(state)

		if self.action_num is None:
			action=dist.rsample()
			log_prob=dist.log_prob(action).sum(dim=1)
			phi=self.get_feature(state,action)
			q1,q2=self.critic(phi)
			q=torch.min(q1,q2)

			actor_loss=(self.alpha.detach()*log_prob-q).mean()
			alpha_loss=(-self.alpha*(log_prob.detach()+self.entropy_target)).mean()
		else:
			state=state.repeat_interleave(self.action_num,dim=0)
			action=torch.from_numpy(self.action_space).to(device).repeat(self.batch_size,1)
			phi=self.get_feature(state,action).reshape(self.batch_size,self.action_num,-1)
			q1,q2=self.critic(phi)
			q=torch.min(q1,q2)

			actor_loss=(dist.probs*(self.alpha.detach()*torch.log(dist.probs)-q.detach())).sum(dim=-1).mean()
			alpha_loss=(-self.alpha*((dist.probs*torch.log(dist.probs)).sum(dim=-1).detach()+self.entropy_target)).mean()

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

		phi=self.get_feature(state,action).detach()
		q1,q2=self.critic(phi)

		with torch.no_grad():
			dist=self.actor(state_next)

			if self.action_num is None:
				action_next=dist.sample()
				log_prob=dist.log_prob(action_next).sum(dim=1)
				phi_next=self.get_feature(state_next,action_next)

				q1_next,q2_next=self.critic_target(phi_next)
				q_next=torch.min(q1_next,q2_next)-self.alpha*log_prob
			else:
				state_next=state_next.repeat_interleave(self.action_num,dim=0)
				action_next=torch.from_numpy(self.action_space).to(device).repeat(self.batch_size,1)
				phi_next=self.get_feature(state_next,action_next).reshape(self.batch_size,self.action_num,-1)

				q1_next,q2_next=self.critic_target(phi_next)
				q_next=(dist.probs*(torch.min(q1_next,q2_next)-self.alpha*torch.log(dist.probs))).sum(dim=-1)

			q_target=reward+(1-done)*self.gamma*torch.clamp(q_next,min=0,max=self.H)

		q1_loss=F.mse_loss(q1,q_target)
		q2_loss=F.mse_loss(q2,q_target)

		loss=q1_loss+q2_loss
		self.critic_optimizer.zero_grad()
		loss.backward()
		self.critic_optimizer.step()

		return {"q1":q1.mean().item(),"q2":q2.mean().item(),"q1_loss":q1_loss.item(),"q2_loss":q2_loss.item()}

	def feature_step(self,batch):
		state,action,state_next,reward=unpack_batch(batch)[:4]
		dist_encoder=self.encoder.get_dist(state,action,state_next)
		dist_feature=self.feature.get_dist(state,action)
		z=dist_encoder.rsample()
		x,r=self.decoder(z)
		recon_loss=F.mse_loss(x,state_next)+F.mse_loss(r,reward)

		KL_loss=torch.distributions.kl_divergence(dist_encoder,dist_feature).mean()
		entropy_loss=self.category_size*math.log(self.class_size)-dist_feature.entropy().mean()

		loss=recon_loss+KL_loss+self.eta*entropy_loss
		self.feature_optimizer.zero_grad()
		loss.backward()
		self.feature_optimizer.step()

		return {"recon_loss":recon_loss.item(),"KL_loss":KL_loss.item(),"entropy_loss":entropy_loss.item(),"VAE_loss":loss.item()}

	def update_target(self,model,model_target,tau):
		with torch.no_grad():
			for param,param_target in zip(model.parameters(),model_target.parameters()):
				param_target.copy_(tau*param+(1-tau)*param_target)

	def train_batch(self):
		feature_info={}
		for _ in range(self.feature_steps):
			batch=self.buffer.sample(self.batch_size)
			feature_info=self.feature_step(batch)
			self.update_target(self.feature,self.feature_target,self.tau_feature)

		batch=self.buffer.sample(self.batch_size)
		critic_info=self.critic_step(batch)
		actor_info=self.actor_step(batch)
		self.update_target(self.critic,self.critic_target,self.tau_critic)

		return {**feature_info,**critic_info,**actor_info}
