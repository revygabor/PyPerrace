import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class PaperRaceEnv:
    def __init__(self, trk_pic, trk_col, gg_pic, segm_list):
        self.trk_pic = mpimg.imread(trk_pic)
        self.trk_col = trk_col        #trk_pic-en a pálya színe
        self.gg_pic = mpimg.imread(gg_pic)
        self.segm_list = segm_list        #szakaszok listája
        self.lepesek = 0        #az eddig megtett lépések száma
        self.szak = 2       #a következő átszakítandó szakasz száma


        start = segm_list[0] #az első szkasz közepén áll először az autó
        start_x = np.floor((start[0] + start[2]) / 2)
        start_y = np.floor((start[1] + start[3]) / 2)
        self.kezdo_poz = np.array([start_x, start_y])


    def draw_track(self):
        plt.imshow(self.trk_pic)
        for i in range(len(self.segm_list)):
            X = np.array([self.segm_list[i][0], self.segm_list[i][2]])
            Y = np.array([self.segm_list[i][1], self.segm_list[i][3]])
            plt.plot(X, Y, color='red')

    def lep(self, SpdChn, SpdOld, PosOld, rajz):

        # LEP Ez a függvény számolja a lépést
        # Output:
        #   SpdNew: Az új(a lépés utáni) sebességvektor[Vx, Vy]
        #   PosNew: Az új(a lépés utáni) pozíció[Px, Py]
        #   reward: A kapott jutalom
        #   vege: logikai érték, vége ha a vége valamiért az epizódnak
        # Inputs:
        #   SpdChn: A sebesség megváltoztatása.(De ez relatívban van!!!) (MÁTRIX!!!)
        #   SpdOld: Az aktuális #sebességvektor


        vege = False
        reward = 0

        # az aktuális sebesség irányvektora:
        e1SpdOld = np.array(SpdOld / np.linalg.norm(SpdOld))
        e2SpdOld = np.array([-1 * e1SpdOld[1], e1SpdOld[0]])


        SpdChn = np.asmatrix(SpdChn)


        #a valtozás globálisban:
        SpdChnGlb = np.round(np.column_stack((e1SpdOld, e2SpdOld)) * SpdChn.transpose())

        # Az új sebességvektor:
        SpdNew = SpdOld + np.ravel(SpdChnGlb)
        PosNew = PosOld + SpdNew

        #TODO: remove, only for debugging
        if rajz:
            X = np.array([PosOld[0], PosNew[0]])
            Y = np.array([PosOld[1], PosNew[1]])
            plt.plot(X, Y, color='blue')

        #TODO: palyae function
        if not self.palyae(PosNew):
            reward = -1000
            vege = True

        return (SpdNew, PosNew, reward, vege)


