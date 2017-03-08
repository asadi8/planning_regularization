import REINFORCE
import sys

stepSize=0.05
numHidden=1
hiddenSize=16
maxEpisode=50000
activation='relu'

try:
	run=sys.argv[1]
except:
	run=0

batchEpisodeNumber=10

REINFORCE.learn(run,
                stepSize,numHidden,maxEpisode,activation,hiddenSize,
                batchEpisodeNumber)

