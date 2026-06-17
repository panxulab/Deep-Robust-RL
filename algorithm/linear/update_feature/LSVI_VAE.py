import copy
import json
import os
import sys
from itertools import chain

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

server=os.path.isdir("/code")
if server: sys.path.append("/code/")
path="/output/" if server else "./"

from algorithm.util import ReplayBuffer
from VAE.dis_VAE import Decoder,Encoder,Feature
from algorithm.linear.update_feature.LSVI_infinite import LSVI

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):

	def __init__(self,state_dim,action_dim,hidden_dim,category_size,class_size,lr=1e-4):
		super().__init__()

		self.state_dim=state_dim
		self.action_dim=action_dim
		self.hidden_dim=hidden_dim
		self.category_size=category_size
		self.class_size=class_size

		self.encoder=Encoder(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim,category_size=category_size,class_size=class_size).to(device)
		self.decoder=Decoder(state_dim=state_dim,hidden_dim=hidden_dim,category_size=category_size,class_size=class_size).to(device)
		self.feature=Feature(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim,category_size=category_size,class_size=class_size).to(device)
		feature_params=chain(self.encoder.parameters(),self.decoder.parameters(),self.feature.parameters())
		self.feature_optimizer=torch.optim.Adam(feature_params,lr=lr)

	def feature_step(self,batch):
		dist_encoder=self.encoder.get_dist(batch.state,batch.action,batch.state_next)
		dist_feature=self.feature.get_dist(batch.state,batch.action)
		z=dist_encoder.rsample()
		x,r=self.decoder(z)
		recon_loss=F.mse_loss(x,batch.state_next)+F.mse_loss(r,batch.reward)

		KL_loss=torch.distributions.kl_divergence(dist_encoder,dist_feature).mean()
		entropy_loss=self.category_size*torch.log(torch.tensor(self.class_size))-dist_feature.entropy().mean()

		loss=recon_loss+KL_loss+0.003*entropy_loss
		self.feature_optimizer.zero_grad()
		loss.backward()
		self.feature_optimizer.step()

		return {"recon_loss":recon_loss.item(),"KL_loss":KL_loss.item(),"entropy_loss":entropy_loss.item(),"VAE_loss":loss.item()}

def norm_state(state):
	return state/np.array([4.8,5.0,0.4,5.0])

class LSVI_VAE(LSVI):

	def __init__(self,env,state_dim,action_dim,feature_dim,action_space,K,beta,lamda,gamma,tau,epsilon,rho=None):
		super().__init__(env,state_dim,action_dim,feature_dim,action_space,K,beta,lamda,gamma,tau,epsilon,rho)

		self.buffer=ReplayBuffer(state_dim=state_dim,action_dim=action_dim)

	def warmup(self):
		action=self.action_space[np.random.randint(len(self.action_space))]
		state_next,reward,terminated,truncated,_=self.env.step(action[0])
		if terminated: reward=0.0
		self.buffer.add(norm_state(self.state),action,norm_state(state_next),reward,terminated)
		if terminated or truncated:
			self.state,_=self.env.reset()
		else:
			self.state=state_next

	def explore(self,feature_func):
		if self.phi_cache is None:
			phi_step=self.get_feature(self.state,feature_func)
		else:
			phi_step=self.phi_cache
		action,phi=self.get_action(phi_step)
		state_next,reward,terminated,truncated,_=self.env.step(action[0])
		if terminated: reward=0.0
		phi_next=self.get_feature(state_next,feature_func)
		self.train_reward+=reward
		self.insert(self.state,action,phi,reward,state_next,phi_next,terminated)
		self.buffer.add(norm_state(self.state),action,norm_state(state_next),reward,terminated)
		if terminated or truncated:
			train_reward=self.train_reward
			self.episode+=1
			self.train_reward=0
			self.state,_=self.env.reset()
			self.phi_cache=None
			return train_reward
		else:
			self.state=state_next
			self.phi_cache=phi_next
			return None

	def evaluate(self,feature_func,eval_numb=50):
		eval_reward=0
		env2=gym.make("CartPole-v1",render_mode="rgb_array")

		for _ in range(eval_numb):
			state,_=env2.reset()
			while True:
				phi_step=self.get_feature(state,feature_func)
				action=self.get_action_test(phi_step)
				state_next,reward,terminated,truncated,_=env2.step(action[0])
				if terminated: reward=0.0
				eval_reward+=reward
				if terminated or truncated: break
				state=state_next
		return eval_reward/eval_numb

# agent=CPU_Unpickler(open("LSVI_VAE.pkl","rb")).load()
# VAE_agent=CPU_Unpickler(open("VAE.pkl","rb")).load()

env=gym.make("CartPole-v1",render_mode="rgb_array")
agent=LSVI_VAE(env=env,state_dim=4,action_dim=1,feature_dim=80,action_space=np.array([[0],[1]]),
	K=10000,beta=40.0,lamda=0.001,gamma=0.99,tau=1.0,epsilon=0.0,rho=None)
VAE_agent=VAE(state_dim=4,action_dim=1,hidden_dim=256,category_size=1,class_size=80)
txt=open(path+"res.txt","w")

def plot_figure(data,name):
	plt.figure()
	keys=list(data.keys())
	epochs=range(1,len(data[keys[0]])+1)
	for key_i in keys:
		plt.plot(epochs,data[key_i],label=key_i)
	plt.xlabel("Epochs")
	plt.ylabel(name)
	plt.legend()
	plt.savefig(path+name+".png")

loss_array=[[],[],[],[]]
for k in range(2000):
	agent.warmup()

for k in range(5000):
	loss=VAE_agent.feature_step(agent.buffer.sample(512))
	if (k+1)%10==0:
		print(loss)
		txt.write(json.dumps(loss)+"\n")
		for a,v in zip(loss_array,loss.values()): a.append(v)

VAE_target=copy.deepcopy(VAE_agent)

def VAE_feature(state,action):
	state=norm_state(state)
	state=torch.tensor(state).float().to(device)
	action=torch.tensor(action).float().to(device)
	return VAE_target.feature.get_feature(state,action).squeeze(dim=0).detach().cpu().numpy()

train_reward=[]
test_reward=[]
for k in range(agent.K):
	res=agent.explore(feature_func=VAE_feature)

	if res is not None:
		train_reward.append(res)
		print(f"Episode {agent.episode}: reward={train_reward[-1]}")
		txt.write(f"Episode {agent.episode}: reward={train_reward[-1]}\n")

	agent.estimate_w()

	for _ in range(9):
		VAE_agent.feature_step(agent.buffer.sample(512))
	loss=VAE_agent.feature_step(agent.buffer.sample(512))
	txt.write(json.dumps(loss)+"\n")
	for a,v in zip(loss_array,loss.values()): a.append(v)

	if (k+1)%500==0:
		test_reward.append(agent.evaluate(feature_func=VAE_feature))
		print(f"Step {k}: test={test_reward[-1]}")
		txt.write(f"Step {k}: test={test_reward[-1]}\n")

	if (k+1)%200==0:
		print("Update VAE")
		VAE_target=copy.deepcopy(VAE_agent)
		agent.update_feature(feature_func=VAE_feature)

for i in range(len(loss_array)):
	loss_array[i]=np.convolve(loss_array[i],np.ones(10)/10,mode="valid")

plot_figure({
	"Recon Loss":loss_array[0],
	"KL Divergence":loss_array[1],
	"Entropy Loss":loss_array[2],
	"VAE Loss":loss_array[3]},"VAE Losses")

plot_figure({"Train Reward":train_reward},"Train Rewards")
plot_figure({"Test Reward":test_reward},"Test Rewards")
