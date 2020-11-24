from args import args
import numpy as np
import os

audio_dir = '/data1/yansen/wits/data/audios'

for file in os.listdir(audio_dir): 
    if len(np.load(os.path.join(audio_dir, file)).shape) != 2:
        os.system('rm "%s"' % os.path.join(audio_dir, file))
        print('Now delete %s' % file)
