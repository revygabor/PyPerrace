import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class PaperRaceEnv:
    def __init__(self, trk_pic, trk_col, gg_pic, segm_list):
        self.trk_pic = mpimg.imread(trk_pic)
        self.trk_col = trk_col        #trk_pic-en a pálya színe
        self.gg_pic = mpimg.imread(gg_pic)
        self.segm_list = segm_list        #szakaszok listája
        self.steps = 0        #az eddig megtett lépések száma
        self.segm = 1         #a következő átszakítandó szakasz száma

        start = segm_list[0]  #az első szkasz közepén áll először az autó
        start_x = np.floor((start[0] + start[2]) / 2)
        start_y = np.floor((start[1] + start[3]) / 2)
        self.kezdo_poz = np.array([start_x, start_y])


    def draw_track(self):
        plt.imshow(self.trk_pic)
        for i in range(len(self.segm_list)):
            X = np.array([self.segm_list[i][0], self.segm_list[i][2]])
            Y = np.array([self.segm_list[i][1], self.segm_list[i][3]])
            plt.plot(X, Y, color='red')

    def step(self, spd_chn, spd_old, pos_old, draw):

        # LEP Ez a függvény számolja a lépést
        # Output:
        #   spd_new: Az új(a lépés utáni) sebességvektor[Vx, Vy]
        #   pos_new: Az új(a lépés utáni) pozíció[Px, Py]
        #   reward: A kapott jutalom
        #   end: logikai érték, vége ha a vége valamiért az epizódnak
        # Inputs:
        #   SpdChn: A sebesség megváltoztatása.(De ez relatívban van!!!) (MÁTRIX!!!)
        #   SpdOld: Az aktuális #sebességvektor

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
            plt.plot(X, Y, color='blue')

        if not self.is_on_track(pos_new):
            reward = -1000
            end = True

        if np.array_equal(spd_new, [0,0]):
            end = True

        if not end and self.crosses_finish_line(pos_old, spd_new, self.segm_list[self.segm][0:2], self.segm_list[self.segm][2:]):
            reward = 1000
            self.segm += 1
            if self.segm > len(self.segm_list):
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
        if 1 <= action < 9:
            # a GGpic 41x41-es B&W bmp. A közepétől nézzük, meddig fehér. (A közepén,
            # csak hogy látszódjon, van egy fekete pont!
            xsrt, ysrt = 21, 21
            r = 1
            pix_in_gg = True
            x, y = xsrt, ysrt
            while pix_in_gg:
                # lépjünk az Act irányba +1 pixelnyit, mik x és y ekkor:
                rad = np.pi / 4 *(action+3)
                y = ysrt + round(np.sin(rad)*r)
                x = xsrt + round(np.cos(rad)*r)
                r = r + 1

                #GG-n belül vagyunk-e még?
                pix_in_gg = np.array_equal(self.gg_pic[int(x-1), int(y-1)], [255,255,255,255])

            return -(x-xsrt), y-ysrt
        else:
            return 0, 0

    def reset(self):
        self.segm = 1

    def normalize_data(self, data_orig):
        data = np.zeros(1, 6)
        sizeX = np.shape(self.trk_pic)[1]
        sizeY = np.shape(self.trk_pic)[0]
        data[0] = (data_orig[0] - sizeX/2) / sizeX
        data[2] = (data_orig[2] - sizeX/2) / sizeX
        data[1] = (data_orig[1] - sizeY/2) / sizeY
        data[3] = (data_orig[3] - sizeY/2) / sizeY
        sizeGGX = np.shape(self.gg_pic)[1]
        sizeGGY = np.shape(self.gg_pic)[0]
        data[4] = data_orig[4] / sizeGGX # Az action vektor átlaga 0 körülinek feltételezhető, így csak a méretét kell normalizálni
        data[5] = data_orig[5] / sizeGGY
