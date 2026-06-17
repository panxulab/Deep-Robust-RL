import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

server=os.path.isdir("/code")
if server: sys.path.append("/code/")
path="/output/" if server else "./"

from algorithm.util import SquashedNormal

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

		self.log_std=nn.Parameter(torch.zeros(action_dim))

	def forward(self,state):
		x=F.relu(self.l1(state))
		x=F.relu(self.l2(x))
		mu=self.l3(x)
		std=self.log_std.clamp(-20,2).exp().expand_as(mu)

		dist=SquashedNormal(mu,std)
		return dist

class Critic(nn.Module):

	def __init__(self,state_dim,hidden_dim):
		super().__init__()

		self.l11=nn.Linear(state_dim,hidden_dim)
		self.l12=nn.Linear(hidden_dim,hidden_dim)
		self.l13=nn.Linear(hidden_dim,1)

		init_weights(self.l11)
		init_weights(self.l12)
		init_weights(self.l13,0.01)

	def forward(self,state):
		q1=F.relu(self.l11(state))
		q1=F.relu(self.l12(q1))
		q1=self.l13(q1).squeeze(-1)
		return q1

class PPO(nn.Module):

	def __init__(self,state_dim,action_dim,
			action_space,
			hidden_dim=128,
			num_epochs=10,
			num_steps=2048,
			num_batch=32,
			epsilon=0.2,  # clip parameter
			gamma=0.99,  # discount factor
			lamda=0.95,  # GAE parameter
			clip_grad=0.5,
			lr=1e-4,  # learning rate
			rho=None  # robust parameter
	):
		super().__init__()

		self.state_dim=state_dim
		self.action_dim=action_dim
		self.action_space=action_space
		self.hidden_dim=hidden_dim

		self.num_epochs=num_epochs
		self.num_steps=num_steps
		self.num_batch=num_batch
		self.batch_size=self.num_steps//self.num_batch

		self.epsilon=epsilon
		self.lamda=lamda
		self.clip_grad=clip_grad

		self.gamma=gamma
		self.H=1/(1-gamma)
		self.rho=rho

		self.reset_buffer()
		self.state_next=None

		nlow=torch.from_numpy(self.action_space.low).to(device)
		nhigh=torch.from_numpy(self.action_space.high).to(device)
		self.trans_action=lambda action:(1-action)/2*nlow+(1+action)/2*nhigh
		self.untrans_action=lambda action:2*action/(nhigh-nlow)-(nhigh+nlow)/(nhigh-nlow)

		self.actor=Actor_con(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
		self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)

		self.critic=Critic(state_dim=state_dim,hidden_dim=hidden_dim).to(device)
		self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

	def reset_buffer(self):
		self.step=0
		self.buffer=[
			np.zeros((self.num_steps,self.state_dim)),  # state
			np.zeros((self.num_steps,self.action_dim)),  # action
			np.zeros((self.num_steps)),  # log_prob
			np.zeros((self.num_steps)),  # reward
			np.zeros((self.num_steps))  # done
		]

	def insert_data(self,state,action,log_prob,state_next,reward,done):
		self.buffer[0][self.step]=state
		self.buffer[1][self.step]=action
		self.buffer[2][self.step]=log_prob
		self.buffer[3][self.step]=reward
		self.buffer[4][self.step]=done
		self.state_next=state_next
		self.step+=1

	def get_action(self,state,explore=True):
		state=torch.from_numpy(state.astype(np.float32)).to(device)  # (state_dim)
		dist=self.actor(state)

		action=dist.sample() if explore else dist.mean.detach()
		action=torch.clamp(action,-1+1e-6,1-1e-6)
		log_prob=dist.log_prob(action).sum(-1).item()
		action=self.trans_action(action).cpu().numpy()
		return action,log_prob

	def update_step(self):
		state=torch.as_tensor(self.buffer[0],dtype=torch.float32,device=device)
		action=torch.as_tensor(self.buffer[1],dtype=torch.float32,device=device)
		log_prob=torch.as_tensor(self.buffer[2],dtype=torch.float32,device=device)
		reward=torch.as_tensor(self.buffer[3],dtype=torch.float32,device=device)
		not_done=torch.as_tensor(1-self.buffer[4],dtype=torch.float32,device=device)

		value=self.critic(state).detach()
		adv=torch.zeros(self.num_steps).to(device)

		curr=0
		for t in reversed(range(self.num_steps)):
			if t==self.num_steps-1:
				state_next=torch.as_tensor(self.state_next,dtype=torch.float32,device=device)
				value_next=self.critic(state_next).detach()
			else:
				value_next=value[t+1]
			delta=reward[t]+not_done[t]*self.gamma*value_next-value[t]
			adv[t]=delta+not_done[t]*self.gamma*self.lamda*curr
			curr=adv[t]

		actor_loss,critic_loss=0,0
		inds=np.arange(self.num_steps)
		for epoch in range(self.num_epochs):
			np.random.shuffle(inds)
			for start in range(0,self.num_steps,self.batch_size):
				batch_inds=inds[start:start+self.batch_size]
				info=self.update_batch(state[batch_inds],action[batch_inds],
					log_prob[batch_inds],value[batch_inds],adv[batch_inds])
				actor_loss+=info[0]
				critic_loss+=info[1]

		self.reset_buffer()
		num_updates=self.num_epochs*self.num_batch
		return {"actor_loss":actor_loss/num_updates,"critic_loss":critic_loss/num_updates}

	def update_batch(self,state,action,log_prob,value,adv):
		returns=adv+value
		adv=(adv-adv.mean())/(adv.std()+1e-6)

		pi_new=self.actor(state)
		action=torch.clamp(self.untrans_action(action),-1+1e-6,1-1e-6)
		ratio=torch.clamp(pi_new.log_prob(action).sum(-1)-log_prob,-20.0,20.0).exp()

		actor_loss1=ratio*adv
		actor_loss2=torch.clamp(ratio,1-self.epsilon,1+self.epsilon)*adv
		actor_loss=-torch.min(actor_loss1,actor_loss2).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		nn.utils.clip_grad_norm_(self.actor.parameters(),self.clip_grad)
		self.actor_optimizer.step()

		v_new=self.critic(state)
		critic_loss1=(v_new-returns)**2
		critic_loss2=(value+torch.clamp(v_new-value,-self.epsilon,self.epsilon)-returns)**2
		critic_loss=torch.max(critic_loss1,critic_loss2).mean()

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		nn.utils.clip_grad_norm_(self.critic.parameters(),self.clip_grad)
		self.critic_optimizer.step()

		return (actor_loss.item(),critic_loss.item())
