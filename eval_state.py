import json
import sys

import gymnasium as gym
import numpy as np
import torch

if sys.argv[1]=="PPO":
	from algorithm.tabular.PPO import Actor_con
elif sys.argv[1]=="TD3":
	from algorithm.tabular.TD3 import Actor_con
else:
	from algorithm.tabular.LSVI_SAC import Actor_con

print(sys.argv[1])

path=f"/model/i-mzxr/Walker2d_actor/{sys.argv[1]}/"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(id,state_dim,action_dim,env_name,hidden_dim=256//(1+(sys.argv[1]=="PPO"))):
	np_mean,np_std=np.load(path+f"run{id}/norm.npy")
	print(f"\nstate mean: {np_mean}, std: {np_std}")
	actor=Actor_con(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(device)
	actor.load_state_dict(torch.load(path+f"run{id}/actor_1000000.pth"))

	action_space=gym.make(env_name).action_space
	nlow=torch.from_numpy(action_space.low).to(device)
	nhigh=torch.from_numpy(action_space.high).to(device)
	trans_action=lambda action:(1-action)/2*nlow+(1+action)/2*nhigh

	def norm_state(state):
		return (state-np_mean)/np_std

	def get_action(state):
		state=torch.from_numpy(state.astype(np.float32)).to(device)  # (state_dim)
		if sys.argv[1]=="TD3":
			action=actor(state).detach()
		else:
			action=actor(state).mean.detach()
		action=trans_action(action).cpu().numpy()
		return action

	def evaluate(eval_numb=40,perturb=0):
		done_numb,eval_reward=0,0
		env2=gym.make(env_name,xml_file=f"/code/envs/walker2d_v5.xml")
		state2,_=env2.reset()
		while done_numb<eval_numb:
			state2+=np.random.normal(scale=0.1*perturb,size=state2.shape)*np.array([0.06,0.3,0.28,0.42,0.4,0.28,0.4,0.4,0.57,0.98,4.6,6,5.3,7.3,6,5.2,7.2])
			action=get_action(norm_state(state2))
			# action=action*(1-perturb)
			# if np.random.rand()<perturb:
			# 	action=env2.action_space.sample()
			state2,reward2,terminated2,truncated2,_=env2.step(action)
			eval_reward+=reward2
			if terminated2 or truncated2:
				done_numb+=1
				state2,_=env2.reset()
		env2.close()
		return eval_reward/eval_numb

	data=[evaluate(perturb=perturb) for perturb in range(10)]
	print(data)
	return data

if __name__=="__main__":
	res=np.array([run(id=i,state_dim=17,action_dim=6,env_name="Walker2d-v5") for i in range(1,11)])
	with open(f"/output/result-{sys.argv[1]}.txt","w") as f:
		f.write(json.dumps([np.mean(res,axis=0).tolist(),np.std(res,axis=0).tolist()]))
