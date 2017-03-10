import numpy, sys

def rewardToReturn(rewards,gamma):# takes a list of reward per t and converts it to return per t
    T=len(rewards)
    returns=T*[0]
    returns[T-1]=rewards[T-1] 
    for t in range(T-2,-1,-1):
        returns[t]=rewards[t]+gamma*returns[t+1]
    return returns

def rep_2_rep_and_action(rep,action,actionSize):
        rep_and_action=numpy.zeros((1,rep.shape[1]+actionSize))
        rep_and_action[0,0:rep.shape[1]]=rep
        one_hot_action=numpy.zeros((1,actionSize))
        one_hot_action[0,action]=1
        rep_and_action[0,rep.shape[1]:rep.shape[1]+actionSize]=one_hot_action
        return rep_and_action

def printLog(episode,returnPerEpisode,frequency):
    if episode % frequency ==0:
        print("***")
        print("episode number",episode)
        print("average performance",numpy.mean(numpy.array(returnPerEpisode[-frequency:])))
        print("***")
        sys.stdout.flush()




            
    
