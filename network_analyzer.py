from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def v(x, y):
    return np.array([0, 1])

nn_file = input("Neural Network file path: ")
# trk_color_red = int(input("Trk Color Red: "))
# trk_color_green = int(input("Trk Color Green: "))
# trk_color_blue = int(input("Trk Color Blue: "))
trk_file = input("Track picture path: ")

model = load_model(nn_file)
image = mpimg.imread(trk_file)

h = image.shape[0]
w = image.shape[1]

colors = [
    (255, 0, 0),  # |^
    (0, 255, 0),  # /^
    (0, 0, 255),  # ->
    (255, 192, 0),  # \ˇ
    (0, 255, 255),  # |ˇ
    (255, 0, 255),  # ˇ/
    (255, 255, 0),  # <-
    (158, 185, 71),  # ^\
    (255, 255, 255)  # .
]

for y in range(h):
    for x in range(w):
        inp = np.concatenate(([x, y], v(x, y)))
        qs = model.predict(np.array([inp]))
        q_max = np.argmax(qs)
        image[y, x] = colors[q_max]

plt.imshow(image)
plt.show()
