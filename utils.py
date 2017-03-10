# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import os
import numpy as np
import cv2

def bgr2rgb(image):
  """Receives a cv2 image (ndarray) in BGR format and 
  returns an image in RGB format
  """
  #b,g,r = cv2.split(image)       # get b,g,r
  #image = cv2.merge([r,g,b])     # switch it to rgb

  #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  #[w,h,c]
  #image = image[:,:,::-1] # start:stop:step
  #image = np.flip(image,2)
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb2bgr(image):
  """Reverse function of bgr2rgb
  """
  return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def read_image(filename):
  """wrapper for reading an image to numpy array format in BGR order
  """
  return cv2.imread(filename)

def shrink_to_gray_image(image):
  """Shrink a given image and convert it to a gray scale
  Assumes 'bgr' image as a input
  """
  # cv2 image - [w,h,c] bgr sequence
  image = np.rint(image[:, :, 2]*0.2989 + image[:, :, 1]*0.5870 + image[:, :, 0]*0.1140).astype(np.uint8) # R - 0.2989, G = 0.5870, B = 01140
  image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)
  return image

def extract_start_states(sstate_dir, load_to_memory = False, width = 210, height = 160):
  episode_dirs = [os.path.join(sstate_dir, x) for x in os.listdir(sstate_dir)]
  episode_dirs = [e for e in episode_dirs if os.path.isdir(e)]

  frame_len = len(episode_dirs)
  if load_to_memory:
    buffer = np.zeros((frame_len, 4, width, height, 3)) # BGR
  else:
    filenames = []

  for i, episode_dir in enumerate(episode_dirs):
    frame_files = [os.path.join(episode_dir, x) for x in os.listdir(episode_dir)]
    frame_files = sorted([f for f in frame_files if f.endswith('.png')])

    if load_to_memory:
      for j, filename in enumerate(frame_files):
        buffer[i, j] = read_image(filename)
    else:
      filenames.append(frame_files)

  if load_to_memory:
    return buffer
  else:
    return filenames
