import numpy


def tilecode(vector,numBlocks,numTilings,minimum,maximum,tileIndices):
    Range=maximum-minimum
    blockSize=float(Range)/(numBlocks-1)
    vector[0]=vector[0]-minimum
    vector[0]=max(vector[0],0)
    vector[0]=min(vector[0],Range)
    tileIndices[0]=int(numpy.floor( (vector[0]) /(blockSize)))
'''
numBlocks=10
numTilings=5
tileIndices = [-1]*numTilings
vector=[.9,.3,1]
tilecode(vector,numBlocks,numTilings,tileIndices)
print(tileIndices)
print(numpy.power(numBlocks,len(vector))*numTilings)
'''
