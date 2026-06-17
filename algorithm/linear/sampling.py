import collections
import random

import numpy as np
import torch

class Standardizer:

	def __init__(self,dim,warmup,eps=1e-6):
		self.dim=dim
		self.warmup=int(warmup)
		self.eps=eps
		self.cnt=0
		self.mean=np.zeros(dim)
		self.n_var=np.zeros(dim)
		self.frozen=False

	def update(self,x):
		self.cnt+=1
		pre_mean=self.mean.copy()
		self.mean+=(x-self.mean)/self.cnt
		self.n_var+=(x-self.mean)*(x-pre_mean)
		if self.cnt==self.warmup:
			self.freeze()

	def transform(self,x):
		return (x-self.mean)/(self.std+self.eps)

	def freeze(self):
		var=self.n_var/(self.cnt-1)
		var=np.maximum(var,self.eps)
		self.std=np.sqrt(var)
		self.frozen=True

class LSHBuckets:

	def __init__(self,state_dim,action_dim,num_table,num_bits,capacity,max_size):
		self.dim=state_dim+action_dim
		self.num_table=num_table
		self.num_bits=num_bits
		self.capacity=capacity
		self.planes=np.array([np.random.randn(num_bits,self.dim) for _ in range(num_table)])
		self.tables=[{} for _ in range(num_table)]

		self.at=0
		self.max_size=num_table*max_size
		self.state=np.zeros((self.max_size,state_dim))
		self.action=np.zeros((self.max_size,action_dim))
		self.state_next=np.zeros((self.max_size,state_dim))
		self.reward=np.zeros((self.max_size,1))
		self.done=np.zeros((self.max_size,1))

	def get_key(self,x,table_i):
		bits=self.planes[table_i]@x>=0
		key=np.packbits(bits.astype(np.uint8)).tobytes()
		return key

	def update(self,idx,token):
		self.state[idx]=token[0]
		self.action[idx]=token[1]
		self.state_next[idx]=token[2]
		self.reward[idx]=token[3]
		self.done[idx]=token[4]

	def insert(self,x,token):
		for table_i in range(self.num_table):
			key=self.get_key(x,table_i)
			if key not in self.tables[table_i]:
				self.tables[table_i][key]=[0,[]]
			item=self.tables[table_i][key]
			if item[0]<self.capacity:
				item[1].append(self.at)
				self.update(self.at,token)
				self.at+=1
			else:
				j=np.random.randint(item[0]+1)
				if j<self.capacity:
					idx=item[1][j]
					self.update(idx,token)
			item[0]+=1

	def sample(self,size):
		chosen=[]

		for table_i in range(self.num_table):
			keys=[*self.tables[table_i]]
			take=size//self.num_table+(table_i<size%self.num_table)

			if len(keys)<take:
				random.shuffle(keys)
			else:
				keys=random.sample(keys,k=take)

			for i,key in enumerate(keys):
				take_i=take//len(keys)+(i<take%len(keys))
				item=self.tables[table_i][key]

				picks=np.random.choice(len(item[1]),size=min(take_i,len(item[1])),replace=False)
				picks=np.concatenate([picks,np.random.choice(len(item[1]),size=max(take_i-len(item[1]),0))])
				for pick in picks:
					chosen.append(item[1][pick])

		chosen=np.array(chosen)
		assert len(chosen)==size
		return self.state[chosen],self.action[chosen],self.state_next[chosen],self.reward[chosen],self.done[chosen]

class ReplayBuffer(object):

	def __init__(self,state_dim,action_dim,max_size=int(3e4),
			num_table=2,num_bits=18,capacity=256,LSH_size=int(1e6),warmup=1000):
		self.at=0
		self.size=0
		self.max_size=max_size

		self.state=np.zeros((max_size,state_dim))
		self.action=np.zeros((max_size,action_dim))
		self.state_next=np.zeros((max_size,state_dim))
		self.reward=np.zeros((max_size,1))
		self.done=np.zeros((max_size,1))

		self.stder=Standardizer(dim=state_dim+action_dim,warmup=warmup)
		self.LSH=LSHBuckets(state_dim=state_dim,action_dim=action_dim,
			num_table=num_table,num_bits=num_bits,capacity=capacity,max_size=LSH_size)

		self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self,state,action,state_next,reward,done):
		self.state[self.at]=state
		self.action[self.at]=action
		self.state_next[self.at]=state_next
		self.reward[self.at]=reward
		self.done[self.at]=done
		self.at=(self.at+1)%self.max_size
		self.size+=1

		sa=np.concatenate([state,action],axis=-1)
		if not self.stder.frozen:
			self.stder.update(sa)
		else:
			sa=self.stder.transform(sa)
			self.LSH.insert(x=sa,token=(state,action,state_next,reward,done))

	def uniform(self,size):
		chosen=np.random.choice(min(self.size,self.max_size),size=size)
		return self.state[chosen],self.action[chosen],self.state_next[chosen],self.reward[chosen],self.done[chosen]

	def sample(self,batch_size,LSH_ratio=1.0):
		num_LSH=round(batch_size*LSH_ratio) if self.stder.frozen else 0
		names=["state","action","state_next","reward","done"]

		samples=dict(zip(names,self.LSH.sample(num_LSH))) if num_LSH else {}
		if num_LSH<batch_size:
			extra=dict(zip(names,self.uniform(batch_size-num_LSH)))
			samples={k:np.concatenate([samples[k],extra[k]],0) for k in names} if samples else extra

		Batch=collections.namedtuple("Batch",names)
		return Batch(
			state=torch.FloatTensor(samples["state"]).to(self.device),
			action=torch.FloatTensor(samples["action"]).to(self.device),
			state_next=torch.FloatTensor(samples["state_next"]).to(self.device),
			reward=torch.FloatTensor(samples["reward"]).to(self.device),
			done=torch.FloatTensor(samples["done"]).to(self.device)
		)
