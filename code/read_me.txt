correct_bias.py
- data preparation
- needs ANT4 bias correction installed https://github.com/ANTsX/ANTs/releases

find_mean_std.py
- find dataset statistic 

train.py
- training file

trainYellowFin.py
- training fili using yellowfish optimizer (training goes faster, but gives slightly worse results)

plot_loss.py
- plots png file with val and train loss into output folder

predictNiiFile.py
- predicts outputs and saves them to nii files
- if there are GT, calculates the dice coeficients



