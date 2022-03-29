from PIL import Image
import numpy as np


data = []

# filename = "query_image.txt"
filename = "data_image.txt"

with open(filename,"r") as f:
    data = f.readlines()

n,m = data[0].split()
n,m = int(n), int(m)

image = [[[0 for i in range(3)] for j in range(m)] for k in range(n )]

data = data[1].split()

# print(image)

for i in range(n):
    for j in range(m):
        for k in range(3):
            image[i][j][k] = data[i*m*3+j*3+k]
            # print(image[i][j][k])
            # break
        # break
    # break

image = np.asarray(image, dtype=np.uint8)

print(image[n-1][0])
# print(image[120][120])
# print(image[119][119])

img = Image.fromarray(image, 'RGB')
img.save('my.png')

