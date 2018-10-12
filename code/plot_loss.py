import pandas as pd
import matplotlib.pyplot as plt
import os

config = dict()
# directories
config["output_foder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/3DUnet_v2/output/EXP18/") # outputs path
config["logging_file"] = os.path.join(config["output_foder"], "training.log")
config["graph_file"] = os.path.join(config["output_foder"], "loss_graph.png")

# plot training and validation loss
training_df = pd.read_csv(config["logging_file"]).set_index('epoch')
x = [i for i in range(1,len(training_df['loss'].values)+1)]
#print(training_df['val_loss'].values)
plt.plot(x, training_df['loss'].values, label='training loss')
plt.plot(x, training_df['val_loss'].values, label='validation loss')
plt.title(['Min train loss: ', min(training_df['loss'].values), ' Min val loss: ', min(training_df['val_loss'].values)])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim((0, len(training_df['loss'].values)+1))
plt.legend(loc='upper right')
plt.savefig(config["graph_file"])

