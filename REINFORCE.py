import numpy,sys
import networkActor
import gym
import math
import utils


def initializeForLearn(stepSize,numHidden,activation,hiddenSize):
	environment='Freeway-v0'
	env = gym.make(environment)
	obs=env.reset()
	state_shape=None
	if environment=='CartPole-v0':
		state_shape=4
	elif environment=='Freeway-v0':
		state_shape=env.observation_space.shape
	actor=networkActor.networkLog(state_shape,env.action_space.n,numHidden,activation,hiddenSize,environment)#build actor
	return env,actor,[]

def initializeForEpisode(env):
    return env.reset(),[],[],[],0

def interactOneEpisode(env,actor):
	rep,rewards,reps,actions,t=initializeForEpisode(env)
	rep=rep.reshape((1,)+rep.shape)
	reps=[]
	while True:
		action=actor.action_selection(rep)
		rep_prime,r,done,_= env.step(action)
		rep_prime=rep_prime.reshape((1,)+rep_prime.shape)
		reps.append(rep),actions.append(action),rewards.append(r) 
		rep,t=(rep_prime,t+1)
		if done==True:
			break
	returns=utils.rewardToReturn(rewards,gamma)
	return returns,reps,actions,rewards

def REINFORCEUpdate(actor,returns,reps,actions,num_actions,efficient=True):
	
	weights=actor.model.get_weights()
	T=len(returns)
	gList=[]
	for w in weights:
	    gList.append(numpy.zeros_like(w))
	if efficient and returns[0]<0.1:
		return gList

	reps=numpy.concatenate(reps,axis=0)
	grad=actor.gradient
	gradients=grad([reps])
	#print(T)

	for param_index in range(len(weights)):
		#print(gradients[param_index].shape)
		grad_param=gradients[param_index]
		#print(grad_param.shape)
		for t,(G_t,phi_t,a_t) in enumerate(zip(returns,reps,actions)):
			gList[param_index]=gList[param_index]+G_t*grad_param[t*num_actions +a_t]
		gList[param_index]=gList[param_index]/T
	return gList


###parameters
gamma=0.99999
###parameters

def learn(run,
          stepSize,numHidden,maxEpisode,activation,hiddenSize,
          batchEpisodeNumber):
    
	env,actor,returnPerEpisode=initializeForLearn(stepSize,numHidden,activation,hiddenSize)
	deltaList=[]
	info=[]
	for episode in range(maxEpisode):
			print("episode number:",episode)
			### interact in the environment for one episode and store relevant information
			returns,reps,actions,rewards=interactOneEpisode(env,actor)
			info.append((returns,reps,actions,rewards))
			returnPerEpisode.append(returns[0])
			deltaListEpisode=REINFORCEUpdate(actor,returns,reps,actions,env.action_space.n)
			deltaList.append(deltaListEpisode)
			### interact in the environment for one episode and store relevant information

			if (episode+1)%batchEpisodeNumber==0:# if reached the batch episode size
			    ### update actor by learning
			    actor.update(deltaList,batchEpisodeNumber,stepSize)
			    deltaList=[]
			    ### update actor by learning

			utils.printLog(episode,returnPerEpisode,frequency=100)