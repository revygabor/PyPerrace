from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def normalize_data(data_orig, h, w):

    data = np.zeros((data_orig.shape[0], data_orig.shape[1]))

    data[0] = (data_orig[0] - w / 2) / w
    data[1] = (data_orig[1] - h / 2) / h
    data[2] = (data_orig[2] - w / 2) / w
    data[3] = (data_orig[3] - h / 2) / h

    return data

def v(w):
    return np.repeat([[0, 1]], w, axis=0)

nn_file = input("Neural Network file path: ")
# trk_color_red = int(input("Trk Color Red: "))
# trk_color_green = int(input("Trk Color Green: "))
# trk_color_blue = int(input("Trk Color Blue: "))
trk_file = input("Track picture path: ")

model = load_model(nn_file)
image = mpimg.imread(trk_file)

h = image.shape[0]
w = image.shape[1]

colors = np.matrix([
    [255, 0, 0],  # |^
    [0, 255, 0],  # /^
    [0, 0, 255],  # ->
    [255, 192, 0],  # \ˇ
    [0, 255, 255],  # |ˇ
    [255, 0, 255],  # ˇ/
    [255, 255, 0],  # <-
    [158, 185, 71],  # ^\
    [255, 255, 255]  # .
])
colors = np.expand_dims(colors, axis=0) # 3D-ssé alakítás (1x9x3)
color_tensor = np.repeat(colors, h, axis=0)

for x in range(w):

    inp = np.full((h, 4), x) # értékűre hozzuk létre a batch mátrixot
    inp[:, 1] = np.arange(h)
    inp[:, 2:4] = v(h)
    inp = normalize_data(inp, h, w)
    qs = model.predict(inp) # (pos(x, y), v)
    q_max = qs.argmax(1)
    image[:, x, :] = color_tensor[:, q_max, :]

    print(x)

plt.imshow(image)
plt.show()
