import numpy
from keras.layers import Input, Dense, Convolution2D, Flatten
from keras.models import Model
from theano.gradient import jacobian
from keras import backend as K
import keras

class networkLog:        
        gradient=0
        model=[]
        def __init__(self,repSize,actionSize,numHidden,activation,hiddenSize,environment):
            if environment=="CartPole-v0":
                self.build_cartpole(repSize,actionSize,numHidden,activation,hiddenSize)
            elif environment=="Freeway-v0":
                self.build_freeway(repSize,actionSize)
            self.gradient=grad(self)
                
        def build_cartpole(self,repSize,actionSize,numHidden,activation,hiddenSize):
                inputs=Input(shape=(repSize,))
                x=inputs
                for i in range(numHidden):
                        x=Dense(hiddenSize, activation=activation)(x)
                predictions = Dense(actionSize, activation='softmax')(x)
                self.model = Model(input=inputs, output=predictions)        
                self.model.compile(optimizer='adadelta',
                                     loss='mse')

        def build_freeway(self,repSize,actionSize):
            S = Input(shape=repSize)
            h = Convolution2D(16, 8, 8, subsample=(4, 4),
                border_mode='same', activation='relu')(S)
            h = Convolution2D(32, 4, 4, subsample=(2, 2),
                border_mode='same', activation='relu')(h)
            h = Flatten()(h)
            h = Dense(256, activation='relu')(h)
            V = Dense(actionSize,activation='softmax')(h)
            self.model = Model(S, V)

        def update(self,deltaList,batchEpisodeNumber,stepSize):
                deltaListSum=deltaList[0]
                for i in range(1,len(deltaList)):
                        for index,d in enumerate(deltaList[i]):
                            deltaListSum[index]=deltaListSum[index]+d
                deltaListMean=[]
                for index in range(len(deltaListSum)):
                        deltaListMean.append(deltaListSum[index]/batchEpisodeNumber)
    
                weights=self.model.get_weights()
                new_weights=[]
                for w,d in zip(weights,deltaListMean):
                        new_weights.append(w+stepSize*d)
                self.model.set_weights(new_weights)
        def action_selection(self,rep):
            aDist=self.model.predict(rep)
            aDist=aDist/numpy.sum(aDist)
            action=numpy.random.choice(len(aDist[0]), 1, p=aDist[0])[0]
            return action   

def grad(self):
        params=[]
        for layer in self.model.layers:
                params += keras.engine.training.collect_trainable_weights(layer)
        netInputs=self.model.input
        netOutputs=numpy.log(self.model.output.flatten())
        gradients=[jacobian(netOutputs,w) for w in params]
        return K.function(inputs=[netInputs],
                          outputs=gradients,
                          updates=self.model.state_updates)
