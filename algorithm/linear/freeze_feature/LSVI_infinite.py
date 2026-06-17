import numpy as np
from scipy.linalg import solve_triangular

class LSVI():

	def __init__(self,env,state_dim,action_dim,feature_dim,action_space,K,beta,lamda,gamma,tau,epsilon,rho=None):
		self.env=env
		self.state_dim=state_dim
		self.action_dim=action_dim
		self.feature_dim=feature_dim
		self.action_space=action_space

		self.state,_=self.env.reset()
		self.episode=0
		self.train_reward=0
		self.phi_cache=None

		self.K=K
		self.beta=beta  # bonus parameter
		self.lamda=lamda  # ridge regression parameter
		self.gamma=gamma  # discount factor
		self.tau=tau  # target weight update parameter
		self.epsilon=epsilon  # epsilon-greedy
		self.H=1/(1-gamma)
		self.rho=rho  # robust parameter

		self.at=0
		self.phi_matrix=np.zeros((K,feature_dim))  # store phi
		self.reward_matrix=np.zeros(K)  # store reward
		self.phi_next=np.zeros((K,len(action_space),feature_dim))  # store V_next
		self.done_matrix=np.zeros(K,dtype=bool)  # store done

		self.Lambda=np.zeros((feature_dim,feature_dim))  # Lambda matrix
		self.Lambda_dec=np.zeros((feature_dim,feature_dim))  # Cholesky decomposition
		self.w_online=np.zeros(feature_dim)
		self.w_target=np.zeros(feature_dim)

		# initialize Lambda matrix
		self.Lambda=lamda*np.eye(feature_dim)
		self.Lambda_dec=np.linalg.cholesky(self.Lambda)

	def insert(self,phi,reward,phi_next,done):
		self.phi_matrix[self.at]=phi
		self.reward_matrix[self.at]=reward
		self.phi_next[self.at]=phi_next
		self.done_matrix[self.at]=done
		self.at+=1

	def get_feature(self,state,feature_func):
		return np.array([feature_func(state,a) for a in self.action_space])

	def get_action(self,phi_full):
		Y=solve_triangular(self.Lambda_dec,phi_full.T,lower=True,check_finite=False)
		Gamma=self.beta*np.linalg.norm(Y,axis=0)
		Q_h=phi_full@self.w_online+Gamma
		action=np.argmax(Q_h)
		if np.random.rand()<self.epsilon:
			action=np.random.randint(len(self.action_space))
		return self.action_space[action],phi_full[action]

	def get_action_test(self,phi_full):
		Q_h=phi_full@self.w_online
		action=np.argmax(Q_h)
		return self.action_space[action]

	def calc_matrix(self):
		phi_matrix=self.phi_matrix[:self.at]
		r_matrix=self.reward_matrix[:self.at]
		not_done=~self.done_matrix[:self.at]
		V_matrix=np.zeros(self.at)

		phi_next=self.phi_next[:self.at][not_done]
		action=np.argmax(phi_next@self.w_online,axis=1)
		V_new=phi_next[np.arange(len(action)),action]@self.w_target
		V_matrix[not_done]=np.maximum(np.minimum(V_new,self.H),0)
		return phi_matrix,r_matrix,V_matrix

	def cholupdate(self,L,x,sign):
		for k in range(self.feature_dim):
			if sign:
				r=np.sqrt(L[k,k]**2+x[k]**2)
			else:
				r=np.sqrt(L[k,k]**2-x[k]**2)
			c=r/L[k,k]
			s=x[k]/L[k,k]
			L[k,k]=r
			if sign:
				L[k+1:,k]=(L[k+1:,k]+s*x[k+1:])/c
			else:
				L[k+1:,k]=(L[k+1:,k]-s*x[k+1:])/c
			x[k+1:]=c*x[k+1:]-s*L[k+1:,k]

	def estimate_w(self):
		phi=self.phi_matrix[self.at-1]
		self.Lambda+=np.outer(phi,phi)
		self.cholupdate(self.Lambda_dec,phi.copy(),True)
		if not (self.at+1)%1000:
			self.Lambda_dec=np.linalg.cholesky(self.Lambda)

		phi_matrix,r_matrix,V_matrix=self.calc_matrix()
		Y=solve_triangular(self.Lambda_dec,phi_matrix.T,lower=True,check_finite=False)
		xi=solve_triangular(self.Lambda_dec.T,Y,lower=False,check_finite=False)

		if self.rho is None:
			self.w_online=xi@(r_matrix+self.gamma*V_matrix)
		else:
			nu=np.zeros(self.feature_dim)
			idxs=np.argsort(V_matrix)
			xi,r_matrix,V_matrix=xi[:,idxs],r_matrix[idxs],V_matrix[idxs]

			for i in range(self.feature_dim):
				xi_i=xi[i]
				res_0,res_1=np.cumsum(xi_i*V_matrix),np.cumsum(xi_i)
				for idx in range(len(idxs)):
					nu[i]=np.maximum(nu[i],res_0[idx]+(res_1[-1]-res_1[idx]-self.rho)*V_matrix[idx])

			self.w_online=xi@r_matrix+self.gamma*nu

		self.w_target=self.tau*self.w_online+(1-self.tau)*self.w_target

	def explore(self,feature_func):
		if self.phi_cache is None:
			phi_step=self.get_feature(self.state,feature_func)
		else:
			phi_step=self.phi_cache
		action,phi=self.get_action(phi_step)
		state_next,reward,terminated,truncated,_=self.env.step(action)
		phi_next=self.get_feature(state_next,feature_func)
		self.train_reward+=reward
		self.insert(phi,reward,phi_next,terminated)
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

	def train(self,feature_func):
		for k in range(self.K):
			self.explore(feature_func)
			self.estimate_w()
