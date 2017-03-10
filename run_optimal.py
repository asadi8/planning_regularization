import REINFORCE
import sys
import gym
import os
import numpy as np
import cv2
from utils import *
from keras.models import load_model
import keras.backend as K
from gym import error, spaces
from gym.utils import seeding
import logging

K.set_image_dim_ordering('th')
logger = logging.getLogger(__name__)

MODEL_EXT = '.h5'
FRAME_HEIGHT = 210
FRAME_WIDTH = 160
NB_CHANNELS = 3
NB_FRAMES = 4
REDUCED_HEIGHT = 84
REDUCED_WIDTH = 84
SCALE_FACTOR = 1./255.

def pre_process(data, mean):
  t = data.copy().astype('float64').transpose([2,0,1]) # to [c, w, h]
  t -= mean
  t *= 1./255.
  return t.astype('float32')

def post_process(data, mean):
  t = data.copy().squeeze()
  t /= 1./255.
  t += mean
  t = t.clip(0, 255)
  return t.astype('uint8').squeeze().transpose([1, 2, 0]) # to [w, h, c]
  
def prepare_rmodel_input(x):
  x_ = np.zeros((NB_FRAMES, REDUCED_HEIGHT, REDUCED_WIDTH))
  for i in range(NB_FRAMES):
    x_[i] = shrink_to_gray_image(x[i])
  x_ = np.expand_dims(x_, 0)
  return x_

class ModelBasedAtariEnv(gym.Env):
  """Atari Simulator for Model Based Reinforcement Learning
  """

  def __init__(self, name='freeway', 
    max_steps = 2040,
    nb_actions = 3,
    tmodel_fname = 'models/5step_iter_700000', 
    rmodel_fname = 'models/model_done', 
    sstates_dir = 'start_states',
    load_sstates_to_memory = False,
    mean_fname = 'mean.npy',
    obs_out_mode = 'rgb'):

    self.name = name
    self.max_steps = max_steps
    self.nb_actions = nb_actions
    self.obs_out_mode = obs_out_mode
    self.observation_space = spaces.Box(low=0, high=255, shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))
    self.action_space = spaces.Discrete(self.nb_actions)
    self.transition_model = load_model(os.path.join(name, tmodel_fname + MODEL_EXT))
    self.reward_model = load_model(os.path.join(name, rmodel_fname + MODEL_EXT))
    self.mean = np.load(os.path.join(name, mean_fname))
    self.start_frames = extract_start_states(os.path.join(name, sstates_dir), load_sstates_to_memory)

    self.frame_buffer = np.zeros((NB_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NB_CHANNELS), dtype='uint8') # [F_SIZE, BGR, W, H]
    self.pframe_buffer = np.zeros((NB_FRAMES, NB_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH), dtype='float32') # [F_SIZE, BGR, W, H]
    self.current_observation = None # [W, H, RGB or BGR]

    self.step_counter = 0

    self._seed()
    logger.info("ATARI {} model finished initialization steps.".format(self.name))

  def _seed(self, seed=None):
    self.np_random, seed1 = seeding.np_random(seed)
    # Derive a random seed. This gets passed as a uint, but gets
    # checked as an int elsewhere, so we need to keep it below
    # 2**31.
    seed2 = seeding.hash_seed(seed1 + 1) % 2**31
    return [seed1, seed2]

  def _set_current_observation(self, bgr_observation):
    if self.obs_out_mode == 'rgb':
      self.current_observation = bgr2rgb(bgr_observation)
    else:
      self.current_observation = bgr_observation

  def _reset(self):
    self.step_counter = 0
    index = self.np_random.randint(len(self.start_frames))

    if isinstance(self.start_frames, np.ndarray):
      start_frames = self.start_frames[index]
    else:
      start_frames = np.zeros((NB_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NB_CHANNELS), dtype='uint8')
      for i, file in enumerate(self.start_frames[index]):
        start_frames[i] = read_image(self.start_frames[index][i])

    #fill the frame buffer 
    self.frame_buffer = start_frames
    for i in range(NB_FRAMES):
      self.pframe_buffer[i, : ,:, :] = pre_process(start_frames[i], self.mean)

    #set the current observation
    self._set_current_observation(start_frames[-1])

    return self.current_observation

  
  def _step(self, action):

    self.step_counter += 1
    # calculate reward
    assert action < self.nb_actions
    a = np.zeros((1, self.nb_actions))
    a[:, action] = 1

    x = prepare_rmodel_input(self.frame_buffer)
    reward = self.reward_model.predict([x, a])

    # calculate next frame
    x_ = np.expand_dims(self.pframe_buffer,0)
    a_ = np.expand_dims(a, 0)
    predicted_frame = self.transition_model.predict([x_, a_])
    # from [c, w, h] to [w, h, c]
    predicted_frame = predicted_frame[0]
    post_predicted_frame = post_process(predicted_frame, self.mean)

    # update the frame buffer
    self.frame_buffer[:-1] = self.frame_buffer[1:]
    self.frame_buffer[-1] = post_predicted_frame
    self.pframe_buffer[:-1] = self.pframe_buffer[1:]
    self.pframe_buffer[-1] = predicted_frame

    self._set_current_observation(post_predicted_frame)
    done = self.step_counter > self.max_steps

    return self.current_observation, reward, done, {}

  def _render(self, mode='human', close=False):
    pass

from gym.envs.registration import registry, register, make, spec




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

