# Implementation of SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
# Ref: https://arxiv.org/pdf/1904.08779.pdf

import random
import numpy as np
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp


class SpecAugment():
    '''
    Augmentation Parameters for policies
    -----------------------------------------
    Policy | W  | F  | m_F |  T  |  p  | m_T
    -----------------------------------------
    None   |  0 |  0 |  -  |  0  |  -  |  -
    -----------------------------------------
    LB     | 80 | 27 |  1  | 100 | 1.0 | 1
    -----------------------------------------
    LD     | 80 | 27 |  2  | 100 | 1.0 | 2
    -----------------------------------------
    SM     | 40 | 15 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    SS     | 40 | 27 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    
    LB  : LibriSpeech basic
    LD  : LibriSpeech double
    SM  : Switchboard mild
    SS  : Switchboard strong
    W   : Time Warp parameter
    F   : Frequency Mask parameter
    m_F : Number of Frequency masks
    T   : Time Mask parameter
    p   : Parameter for calculating upper bound for time mask
    m_T : Number of time masks
    '''
    
    def __init__(self, mel_spectrogram, policy, zero_mean_normalized=True):
        self.mel_spectrogram = mel_spectrogram
        self.policy = policy
        self.zero_mean_normalized = zero_mean_normalized
        
        # Policy Specific Parameters
        if self.policy == 'LB':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif self.policy == 'LD':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif self.policy == 'SM':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif self.policy == 'SS':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2
        
        
    def time_warp(self):
    
        # Reshape to [Batch_size, time, freq, 1] for sparse_image_warp func.
        
        self.mel_spectrogram = np.reshape(self.mel_spectrogram, (-1, self.mel_spectrogram.shape[0], self.mel_spectrogram.shape[1], 1))
        
        v, tau = self.mel_spectrogram.shape[1], self.mel_spectrogram.shape[2]
        
        horiz_line_thru_ctr = self.mel_spectrogram[0][v//2]
    
        random_pt = horiz_line_thru_ctr[random.randrange(self.W, tau - self.W)] # random point along the horizontal/time axis
        w = np.random.uniform((-self.W), self.W) # distance
        
        # Source Points
        src_points = [[[v//2, random_pt[0]]]]
        
        # Destination Points
        dest_points = [[[v//2, random_pt[0] + w]]]
        
        self.mel_spectrogram, _ = sparse_image_warp(self.mel_spectrogram, src_points, dest_points, num_boundary_points=2)
    
        return self.mel_spectrogram
    
    
    def freq_mask(self):
      v = self.mel_spectrogram.shape[1]  # no. of mel bins

      for i in range(self.m_F):
          f = np.random.uniform(0, self.F)  # [0, F)
          f = tf.cast(f, tf.int32)  # Ensure f is an integer
          f0 = tf.random.uniform(shape=(), minval=0, maxval=v - f, dtype=tf.int32)  # [0, v - f)
          
          # Create mask
          mask = tf.concat([
              tf.ones(shape=(self.mel_spectrogram.shape[0], f0, self.mel_spectrogram.shape[2], 1)),
              tf.zeros(shape=(self.mel_spectrogram.shape[0], f, self.mel_spectrogram.shape[2], 1)),
              tf.ones(shape=(self.mel_spectrogram.shape[0], v - f0 - f, self.mel_spectrogram.shape[2], 1))
          ], axis=1)
          
          # Apply mask
          self.mel_spectrogram = self.mel_spectrogram * mask
          
      return self.mel_spectrogram
    
    
    def time_mask(self):
      tau = self.mel_spectrogram.shape[2]  # time frames

      for i in range(self.m_T):
          t = np.random.uniform(0, self.T)  # [0, T)
          t = tf.cast(t, tf.int32)  # Ensure t is an integer
          t0 = tf.random.uniform(shape=(), minval=0, maxval=tau - t, dtype=tf.int32)  # [0, tau - t)
          
          # Create mask
          mask = tf.concat([
              tf.ones(shape=(self.mel_spectrogram.shape[0], self.mel_spectrogram.shape[1], t0, 1)),
              tf.zeros(shape=(self.mel_spectrogram.shape[0], self.mel_spectrogram.shape[1], t, 1)),
              tf.ones(shape=(self.mel_spectrogram.shape[0], self.mel_spectrogram.shape[1], tau - t0 - t, 1))
          ], axis=2)
          
          # Apply mask
          self.mel_spectrogram = self.mel_spectrogram * mask
          
      return self.mel_spectrogram
