import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import random

from environment import PaperRaceEnv

plt.ion()
plt.show()

trk_col = np.array([99, 99, 99])

segm_list = [
    np.array([350, 60, 350, 100]),
    np.array([360, 60, 360, 100]),
    np.array([539, 116, 517, 137]),
    np.array([348, 354, 348, 326]),
    np.array([35, 200, 70, 200]),
    np.array([250, 60, 250, 100])
]

env = PaperRaceEnv('PALYA3.bmp', trk_col, 'GG1.bmp', segm_list)

N_hidden = 100
mem_size = 1000
batch_size = 20
explore = 0.9

qn = Sequential()
qn.add(Dense(N_hidden, input_shape=(6,), activation='sigmoid'))
qn.add(Dense(1, activation='linear'))
qn.compile(optimizer='adam', loss='mse')
exp_memory = deque(maxlen=mem_size)

episodes = 10000

for _ in range(episodes):
    plt.clf()
    env.draw_track()
    v = np.array([1, 0])
    pos = np.array(env.kezdo_poz)
    reward = 0
    end = False
    while not end:
        if random.random() < explore:
            action = random.randint(1, 9)
        else:
            qs = [qn.predict(np.array([np.concatenate((pos, v, env.gg_action(act)))]))[0] for act in range(1, 10)]
            action = np.argmax(qs) + 1

        action = env.gg_action(action)
        v_new, pos_new, reward, end = env.step(action, v, pos, True)
        exp_memory.append( (np.concatenate((pos, v)), action, np.concatenate((pos_new, v_new)), reward, end) )

        if len(exp_memory) >= batch_size:
            batch = random.sample(exp_memory, batch_size)
            inputs = np.zeros((batch_size, 6))
            targets = np.zeros((batch_size, 1))
            for i in range(batch_size):
                state = batch[i][0]
                action = batch[i][1]
                new_state = batch[i][2]
                r = batch[i][3]
                done = batch[i][4]

                inp = np.concatenate((state, action))
                inputs[i:i+1] = np.expand_dims(inp, axis=0)
                q_next = qn.predict(np.array([np.concatenate((new_state, env.gg_action(1)))]))[0]
                for aa in range(2, 10):
                    qq = qn.predict(np.array([np.concatenate((new_state, env.gg_action(aa)))]))[0]
                    if qq > q_next:
                        q_next = qq

                if done:
                    targets[i] = reward
                else:
                    targets[i] = reward + q_next
            qn.train_on_batch(inputs, targets)

        v = v_new
        pos = pos_new

        plt.draw()
