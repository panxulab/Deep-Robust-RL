import argparse
import random
import shutil

import gymnasium as gym
import matplotlib.pyplot as plt

from algorithm.tabular.TD3_KL import *

parser=argparse.ArgumentParser()
parser.add_argument("--seed",type=int,default=0)

args=parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(True)

def norm_state(state):
	return (state-np_mean)/np_std

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

def evaluate(agent,eval_numb=50,perturb=0.0):
	done_numb,eval_reward=0,0
	env2=gym.make("Walker2d-v5")
	env2.reset(seed=args.seed+20000)
	env2.action_space.seed(args.seed+20000)
	state2,_=env2.reset()
	while done_numb<eval_numb:
		action=agent.get_action(norm_state(state2),explore=False)
		if np.random.rand()<perturb:
			action=env2.action_space.sample()
		state2,reward2,terminated2,truncated2,_=env2.step(action)
		eval_reward+=reward2
		if terminated2 or truncated2:
			done_numb+=1
			state2,_=env2.reset()
	env2.close()
	return eval_reward/eval_numb

def train():
	env=gym.make("Walker2d-v5")
	env.reset(seed=args.seed+10000)
	env.action_space.seed(args.seed+10000)
	agent=TD3(state_dim=17,action_dim=6,rho=75,action_space=env.action_space).to(device)

	start_train=2000
	max_steps=1000000
	eval_freq=20000

	train_reward=[]
	test_reward=[]
	episode_reward=0

	visit=[]
	state,_=env.reset()
	for step in range(10000):
		visit.append(state)
		action=env.action_space.sample()
		state,reward,terminated,truncated,_=env.step(action)
		if terminated or truncated:
			state,_=env.reset()

	global np_mean,np_std
	np_mean=np.mean(visit,axis=0)
	np_std=np.std(visit,axis=0)+1e-6
	print(f"state mean: {np_mean}, std: {np_std}")

	state,_=env.reset()
	for step in range(int(max_steps)):
		if step<start_train:
			action=env.action_space.sample()
		else:
			action=agent.get_action(norm_state(state))

		state_next,reward,terminated,truncated,_=env.step(action)
		reward=np.float64(reward)*0.1
		agent.buffer.add(norm_state(state),action,norm_state(state_next),reward,terminated)

		state=state_next
		episode_reward+=reward

		if step>=start_train:
			info=agent.train_batch() if step%agent.update_policy else agent.train_batch(update_actor=True)
			if (step+1)%20==0:
				print(f"Step {step}: info={info}")

		if terminated or truncated:
			state,_=env.reset()
			train_reward.append(episode_reward)
			print(f"Step {step}: train={episode_reward}")
			episode_reward=0

		if (step+1)%eval_freq==0:
			test_reward.append(evaluate(agent))
			print(f"Step {step}: test={test_reward[-1]}")

		if (step+1)%200000==0:
			torch.save(agent.actor.state_dict(),path+f"actor_{step+1}.pth")

	torch.save(agent.actor.state_dict(),path+"actor.pth")
	np.save(path+"norm.npy",np.array([np_mean,np_std]))
	print(np.round([evaluate(agent,perturb=0.05*perturb) for perturb in range(10)],2))

	plot_figure({"train_reward":train_reward},"train_reward")
	plot_figure({"test_reward":test_reward},"test_reward")

if __name__=="__main__":
	train()
