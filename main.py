import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import random

from environment import PaperRaceEnv

def softmax(x):
    min = np.min(x)
    max = np.max(x)
    corr = min if np.abs(min) > np.abs(max) else max

    exp = np.exp(x - corr)
    s = sum(exp)
    return exp / s


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
explore_reduction = 0.001

draw = True

load = input('Load saved model (qn.h5)? (y/n)') == 'y'
if load:
    qn = load_model('qn.h5')
else:
    qn = Sequential()
    qn.add(Dense(N_hidden, input_shape=(4,), activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='zeros'))
    qn.add(Dense(9, activation='linear', kernel_initializer='glorot_normal', bias_initializer='zeros'))
    qn.compile(optimizer='sgd', loss='mse')

exp_memory = deque(maxlen=mem_size)

episodes = 10000

for ep in range(episodes):
    print(ep)
    if draw:
        plt.clf()
        env.draw_track()
    v = np.array([1, 0])
    pos = np.array(env.kezdo_poz)
    reward = 0
    end = False
    while not end:

        #e-greedy
        if random.random() < explore:
            action = random.randint(1, 9)
            color = 'yellow'
        else:
            qs = qn.predict(
                env.normalize_data(
                    np.array([np.concatenate((
                        pos, v
                    ))])
                )
            )[0]
            action = np.argmax(qs) + 1
            color = 'red'

        # #softmax
        # qs = [qn.predict(
        #     env.normalize_data(
        #         np.array([np.concatenate((
        #             pos, v, env.gg_action(act)
        #         ))])
        #     )
        # )[0] for act in range(1, 10)]
        # qs = np.array(qs)
        # sm = softmax(qs)
        # cs = np.cumsum(sm)
        # action = 1
        # rand = random.random()
        # for i in range(len(cs)):
        #     if rand > cs[i]:
        #         action = i+2
        #     else:
        #         break
        # color = (sm[action], 1-sm[action], 1)

        gg_action = env.gg_action(action)
        v_new, pos_new, reward, end = env.step(gg_action, v, pos, draw, color)
        exp_memory.append( (np.concatenate((pos, v)), action, np.concatenate((pos_new, v_new)), reward, end) )

        if len(exp_memory) >= batch_size:
            batch = random.sample(exp_memory, batch_size)
            train_inp = np.zeros((batch_size, 4))
            targets = np.zeros((batch_size, 9))
            for i in range(batch_size):
                state = batch[i][0]
                action = batch[i][1]
                new_state = batch[i][2]
                r = batch[i][3]
                done = batch[i][4]

                inp = state
                train_inp[i:i + 1] = inp

                pred_inp = np.expand_dims(np.array(new_state), axis=0)
                pred_inp = env.normalize_data(pred_inp)
                q_next = np.max(qn.predict(pred_inp)[0])

                targets[i, :] = qn.predict(env.normalize_data(np.expand_dims(inp, axis=0)))
                if done:
                    targets[i, action - 1] = reward
                else:
                    targets[i, action - 1] = reward + q_next
            train_inp = env.normalize_data(train_inp)
            qn.train_on_batch(train_inp, targets)

        v = v_new
        pos = pos_new

    explore = max([0, explore - explore_reduction])
    env.reset()

    if draw:
        plt.pause(0.001)
        plt.draw()

    if ep % 1000 == 0:
        qn.save('qn.h5')
