from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
#from scipy.interpolate import spline

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
         self.losses = []

    def on_batch_end(self, batch, logs={}):
         self.losses.append(logs.get('loss'))

        
     
    def smooth_losses(self, n = 10):
        smooth = np.cumsum(self.losses, dtype=float)
        smooth[n:] = smooth[n:] - smooth[:-n]
        return(smooth[n-1:] / n)
  
    def plot_smoothed(self, n = 10):
        plt.plot(self.smooth_losses(n = n))