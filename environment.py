import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from random import randint
from skimage.morphology import disk
from skimage.color import rgb2gray

class PaperRaceEnv:
    def __init__(self, trk_pic, trk_col, gg_pic, segm_list, random_init=False, track_inside_color=None):
        if track_inside_color is None:
            self.track_inside_color = np.array([255, 0, 0], dtype='uint8')
        else:
            self.track_inside_color = np.array(track_inside_color, dtype='uint8')
        self.trk_pic = mpimg.imread(trk_pic)
        self.trk_col = trk_col        #trk_pic-en a pálya színe
        self.gg_pic = mpimg.imread(gg_pic)
        self.segm_list = segm_list        #szakaszok listája, [x1 y1 x2 y2]
        self.steps = 0        #az eddig megtett lépések száma
        self.segm = 1         #a következő átszakítandó szakasz száma

        start = segm_list[0]  #az első szkasz közepén áll először az autó
        start_x = int(np.floor((start[0] + start[2]) / 2))
        start_y = int(np.floor((start[1] + start[3]) / 2))
        self.starting_pos = np.array([start_x, start_y])
        self.random_init = random_init
        self.gg_actions = None
        self.track_indices = []
        self.prev_dist = 0
        for x in range(self.trk_pic.shape[1]):
            for y in range(self.trk_pic.shape[0]):
                if np.array_equal(self.trk_pic[y, x, :], self.trk_col):
                    self.track_indices.append([x, y])

        self.dists = self.__get_dists(True)

    def draw_track(self):
        plt.imshow(self.trk_pic)
        for i in range(len(self.segm_list)):
            X = np.array([self.segm_list[i][0], self.segm_list[i][2]])
            Y = np.array([self.segm_list[i][1], self.segm_list[i][3]])
            plt.plot(X, Y, color='red')

    def step(self, spd_chn, spd_old, pos_old, draw, color):

        # LEP Ez a függvény számolja a lépést
        # Output:
        #   spd_new: Az új(a lépés utáni) sebességvektor[Vx, Vy]
        #   pos_new: Az új(a lépés utáni) pozíció[Px, Py]
        #   reward: A kapott jutalom
        #   end: logikai érték, vége ha a vége valamiért az epizódnak
        # Inputs:
        #   SpdChn: A sebesség megváltoztatása.(De ez relatívban van!!!) (MÁTRIX!!!)
        #   SpdOld: Az aktuális sebességvektor

        end = False
        reward = 0

        # az aktuális sebesség irányvektora:
        e1_spd_old = spd_old / np.linalg.norm(spd_old)
        e2_spd_old = np.array([-1 * e1_spd_old[1], e1_spd_old[0]])

        spd_chn = np.asmatrix(spd_chn)

        #a valtozás globálisban:
        spd_chn_glb = np.round(np.column_stack((e1_spd_old, e2_spd_old)) * spd_chn.transpose())

        # Az új sebességvektor:
        spd_new = spd_old + np.ravel(spd_chn_glb)
        pos_new = pos_old + spd_new

        if draw:
            X = np.array([pos_old[0], pos_new[0]])
            Y = np.array([pos_old[1], pos_new[1]])
            plt.plot(X, Y, color=color)

        start_line = self.segm_list[0]
        if not self.is_on_track(pos_new):
            reward = -10
            end = True
        elif start_line[1] < pos_new[1] < start_line[3] and pos_old[0] >= start_line[0] > pos_new[0]: #visszafelé indul
            reward = -10
            end = True
        else:
            reward = self.get_reward(pos_new)
        if np.array_equal(spd_new, [0, 0]):
            end = True

        return spd_new, pos_new, reward, end

    def is_on_track(self, pos):
        if pos[0] > np.shape(self.trk_pic)[1] or pos[1] > np.shape(self.trk_pic)[0] or \
         pos[0] < 0 or pos[1] < 0  or np.isnan(pos[0]) or np.isnan(pos[1]):
            return False
        else:
            return np.array_equal(self.trk_pic[int(pos[1]),int(pos[0])], self.trk_col)

    def crosses_finish_line(self, pos, spd, finish_left, finish_right):
        # Ha a Pos-ból húzott Spd vektor metszi a celvonalat (Szakasz(!),
        # nem egynes) akkor 1-et ad vissza (true)
        # t2 az az ertek ami mgmondja hogy a Spd hanyadánál metszi a celvonalat. Ha
        # t2=1 akkor a Spd vektor eppenhogy eleri a celvonalat.

        # keplethez kello ertekek. p1, es p2 pontokkal valamint v1 es v2
        # iranyvektorokkal adott egyenesek metszespontjat nezzuk, ugy hogy a
        # celvonal egyik pontjabol a masikba mutat a v1, a v2 pedig a sebesseg, p2
        # pedig a pozicio
        v1y, v1z = finish_right - finish_left
        v2y, v2z = spd

        p1y, p1z = finish_left
        p2y, p2z = pos

        # t2 azt mondja hogy a p1 pontbol v1 iranyba indulva v1 hosszanak hanyadat
        # kell megtenni hogy elerjunk a metszespontig. Ha t2=1 epp v2vegpontjanal
        # van a metszespopnt. t1,ugyanez csak p1 es v2-vel.
        t2 = (-v1y*p1z+v1y*p2z+v1z*p1y-v1z*p2y)/(-v1y*v2z+v1z*v2y)
        t1 = (p1y*v2z-p2y*v2z-v2y*p1z+v2y*p2z)/(-v1y*v2z+v1z*v2y)

        # Annak eldontese hogy akkor az egyenesek metszespontja az most a
        # szakaszokon belulre esik-e: Ha mindket t, t1 es t2 is kisebb mint 1 és
        # nagyobb mint 0
        celba = 0 <= t1 <= 1 and 0 <= t2 <= 1
        if not(celba):
            t2 = 0
        return celba

    def gg_action(self, action):
        if self.gg_actions is None:
            self.gg_actions = [None] * 9
            for act in range(1, 10):
                if 1 <= act < 9:
                    # a GGpic 41x41-es B&W bmp. A közepétől nézzük, meddig fehér. (A közepén,
                    # csak hogy látszódjon, van egy fekete pont!
                    xsrt, ysrt = 21, 21
                    r = 1
                    pix_in_gg = True
                    x, y = xsrt, ysrt
                    while pix_in_gg:
                        # lépjünk az Act irányba +1 pixelnyit, mik x és y ekkor:
                        rad = np.pi / 4 * (act+3)
                        y = ysrt + round(np.sin(rad)*r)
                        x = xsrt + round(np.cos(rad)*r)
                        r = r + 1

                        #GG-n belül vagyunk-e még?
                        pix_in_gg = np.array_equal(self.gg_pic[int(x-1), int(y-1)], [255,255,255,255])

                    self.gg_actions[act-1] = (-(x-xsrt), y-ysrt)
                else:
                    self.gg_actions[act-1] = (0, 0)
        return self.gg_actions[action - 1]

    def reset(self):
        self.segm = 1
        self.prev_dist = 0
        if self.random_init:
            self.starting_pos = self.track_indices[randint(0, len(self.track_indices) - 1)]

    def normalize_data(self, data_orig):
        n_rows = data_orig.shape[0]
        data = np.zeros((n_rows, 4))
        sizeX = np.shape(self.trk_pic)[1]
        sizeY = np.shape(self.trk_pic)[0]
        data[:, 0] = (data_orig[:, 0] - sizeX/2) / sizeX
        data[:, 2] = (data_orig[:, 2] - sizeX/2) / sizeX
        data[:, 1] = (data_orig[:, 1] - sizeY/2) / sizeY
        data[:, 3] = (data_orig[:, 3] - sizeY/2) / sizeY
        return data

    def get_reward(self, pos_new):
        trk = rgb2gray(self.trk_pic)
        col = rgb2gray(np.reshape(self.track_inside_color, (1,1,3)))
        pos_new = np.array(pos_new, dtype='int32')
        tmp = [0]
        r = 0

        while not np.any(tmp):
            r = r+1
            tmp = trk[pos_new[1]-r:pos_new[1]+r+1, pos_new[0]-r:pos_new[0]+r+1]
            mask = disk(r)
            tmp = np.multiply(mask, tmp)
            tmp[tmp != col] = 0

        indices = [p[0] for p in np.nonzero(tmp)]
        offset = [indices[1]-r, indices[0]-r]
        pos = np.array(pos_new + offset)
        curr_dist = self.dists[tuple(pos)]
        reward = curr_dist - self.prev_dist
        self.prev_dist = curr_dist
        return reward

    def __get_dists(self, rajz=False):
        """
        :return: a dictionary consisting (inner track point, distance) pairs
        """
        dist_dict = {}
        start_point = self.starting_pos
        trk = rgb2gray(self.trk_pic)
        col = rgb2gray(np.reshape(np.array(self.track_inside_color), (1, 1, 3)))[0, 0]
        tmp = [0]
        r = 0
        while not np.any(tmp):
            r = r+1
            mask = disk(r)
            tmp = trk[start_point[1]-r:start_point[1]+r+1, start_point[0]-r:start_point[0]+r+1]
            tmp = np.multiply(mask, tmp)
            tmp[tmp != col] = 0

        indices = [p[0] for p in np.nonzero(tmp)]
        offset = [indices[1]-r, indices[0]-r]
        start_point = np.array(start_point+offset)
        dist_dict[tuple(start_point)] = 0
        dist = 0
        JOBB, FEL, BAL, LE = [1, 0], [0, -1], [-1, 0], [0, 1]  # [x, y], tehát a mátrix indexelésekor fordítva, de a pozícióhoz hozzáadható azonnal
        dirs = [JOBB, FEL, BAL, LE]
        direction_idx = 0
        point = start_point
        if rajz:
            self.draw_track()
        while True:
            dist += 1
            bal_ford = dirs[(direction_idx+1) % 4]
            jobb_ford = dirs[(direction_idx-1) % 4]
            if trk[point[1] + bal_ford[1], point[0] + bal_ford[0]] == col:
                direction_idx = (direction_idx + 1) % 4
                point = point + bal_ford
            elif trk[point[1] + dirs[direction_idx][1], point[0] + dirs[direction_idx][0]] == col:
                point = point + dirs[direction_idx]
            else:
                direction_idx = (direction_idx - 1) % 4
                point = point + jobb_ford
            dist_dict[tuple(point)] = dist
            if rajz:
                plt.plot([point[0]], [point[1]], 'yo')
            if np.array_equal(point, start_point):
                break
        if rajz:
            plt.draw()
            plt.pause(0.001)
        return dist_dict
