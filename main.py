import matplotlib.pyplot as plt
from environment import PaperRaceEnv
import numpy as np

trk_col = np.array([99, 99, 99])

segm_list = [
            np.array([350, 60,350,100]),\
            np.array([360, 60,360,100]),\
            np.array([539,116,517,137]),\
            np.array([348,354,348,326]),\
            np.array([ 35,200, 70,200]),\
            np.array([250, 60,250,100])
            ]


env = PaperRaceEnv('PALYA3.bmp', trk_col, 'GG1.bmp', segm_list)
env.draw_track()


#test:
action = np.array([1, 2])
v = np.array([1, 1])
pos= np.array([50, 50])
rajz = True
env.lep(action, v, pos, rajz)

#endest


plt.show()