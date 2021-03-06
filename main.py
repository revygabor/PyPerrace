import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import random

from environment import PaperRaceEnv

def softmax(x, tau):
    min_x = np.min(x)
    max_x = np.max(x)
    corr = min_x if np.abs(min_x) > np.abs(max_x) else max_x

    exp = np.exp((x - corr) / tau)
    s = sum(exp)
    return exp / s

draw = False
load = False

print('Loading saved model') if load else print('Creating new model') # model betöltése vagy új létrehozása

if draw:
    plt.ion() # kirajzoláshoz szükséges cuccok
    plt.show()

trk_col = np.array([99, 99, 99]) # pálya színe (szürke)

start_line = np.array([350, 60, 350, 100]) # eredeti pálya
# start_line = np.array([32, 393, 32, 425]) # sigmoid alakú pálya

env = PaperRaceEnv('PALYA3.bmp', trk_col, 'GG1.bmp', start_line, random_init=True, draw=draw) # paperrace környezet létrehozása

N_hidden = 800 # rejtett neuronok száma
mem_size = 1000 # a memória mérete, amiből a batch-be válogatunk
batch_size = 30 # batch mérete, ami a tanítási adatokat tartalmazza
discount_factor = 0.8
explore = 0.9
explore_reduction = 0.001
tau = 5
tau_reduction = 0.0001
episodes = 10000 # hányszor fusson a tanítás


if load:
    qn = load_model('qn.h5')
else: # model létrehozása
    qn = Sequential()
    qn.add(Dense(N_hidden, input_shape=(4,), activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='zeros'))
    qn.add(Dense(9, activation='linear', kernel_initializer='glorot_normal', bias_initializer='zeros'))
    qn.compile(optimizer='sgd', loss='mse')

exp_memory = deque(maxlen=mem_size)


text = None
latest_loss = 0
for ep in range(episodes):
    print(ep) # epizód számának kiírása

    if draw: # ha rajzolunk
        plt.clf()
        env.draw_track()

    v = np.array([1, 0]) # az elején a sebesség jobbra 1
    #ezt könnyen megváltoztatja, tulajdonképen csak arra jó, hogy nem 0
    pos = np.array(env.starting_pos) # kezdőpozíció beállítása
    reward = 0
    end = False
    while not end:

        # #e-greedy
        # if random.random() < explore:
        #     action = random.randint(1, 9)
        #     color = 'yellow'
        # else:
        #     qs = qn.predict(
        #         env.normalize_data(
        #             np.array([np.concatenate((
        #                 pos, v
        #             ))])
        #         )
        #     )[0]
        #     action = np.argmax(qs) + 1
        #     color = 'red'

        #softmax
        qs = qn.predict(
            env.normalize_data(
                np.array([np.concatenate((
                    pos, v
                ))])
            )
        )[0]
        qs = np.array(qs)
        sm = softmax(qs, tau)
        cs = np.cumsum(sm)
        action = 1
        rand = random.random()
        for i in range(len(cs)):
            if rand > cs[i]:
                action = i+2
            else:
                break
        color = (sm[action-1], 1-sm[action-1], 1)


        gg_action = env.gg_action(action) # action-höz tartozó vektor lekérése
        v_new, pos_new, reward, end = env.step(gg_action, v, pos, draw, color) # lépés
        # a háló által kiválasztott gyorsulással

        if draw:
            if text is not None: # adatok kiírása
                text.set_visible(False)
            text = plt.text(0, 0, "Q=" + np.array_str(qs) + "\nsm=" + np.array_str(sm) + "\nChosen action: " + str(action) +
                        " Reward: " + str(reward))

        else:
            if (ep%10 == 0):
                print("Q=" + np.array_str(qs) + "\nsm=" + np.array_str(sm) + "\nChosen action: " + str(action) +
                        " Reward: " + str(reward))

        if draw:
            plt.pause(0.001)
            plt.draw()

        exp_memory.append( (np.concatenate((pos, v)), action, np.concatenate((pos_new, v_new)), reward, end) )
        # lépés adatainak hozzáadaása a memóriához

        if len(exp_memory) >= batch_size: # ha már elég adat van a memóriában
            batch = random.sample(exp_memory, batch_size) # véletlenszerűen kiválasztunk elemeket a memóriából a batchbe
            train_inp = np.zeros((batch_size, 4))
            targets = np.zeros((batch_size, 9))
            for i in range(batch_size):
                state = batch[i][0] # a batchből kivesszük az adatokat
                action = batch[i][1]
                new_state = batch[i][2]
                r = batch[i][3]
                done = batch[i][4]

                inp = state
                train_inp[i] = inp # berakjuk az inputba, amivel tanítunk

                pred_inp = np.expand_dims(np.array(new_state), axis=0)
                pred_inp = env.normalize_data(pred_inp)
                q_next = np.max(qn.predict(pred_inp)[0])

                targets[i, :] = qn.predict(env.normalize_data(np.expand_dims(inp, axis=0)))
                if done:
                    targets[i, action - 1] = reward
                else:
                    targets[i, action - 1] = reward + discount_factor * q_next
            train_inp = env.normalize_data(train_inp)
            latest_loss = qn.train_on_batch(train_inp, targets)

        v = v_new
        pos = pos_new

    explore = max([0, explore - explore_reduction])
    tau = max([1, tau - tau_reduction])
    env.reset()

    if ep % 1000 == 0:
        qn.save('qn.h5')
