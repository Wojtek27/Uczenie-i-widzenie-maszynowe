import matplotlib.pyplot as plt
import numpy as np

M = 16
angles = np.arange(0, 2*np.pi, 1/36)
m = np.arange(-M/2, M/2, 1)
n = 3

listOfImages = []

for vel in m:
    r = []
    img = []
    for alpha in angles:
        r.append(np.sin(n*alpha+vel*np.pi/10))
    img.append(angles)
    img.append(r)
    listOfImages.append(img)

listOfMatrix = []
for image in listOfImages:
    x = []
    y = []
    matrix = []
    for i in range(0, len(image[0])):
        x.append(image[1][i]*np.cos(image[0][i]))
        y.append(image[1][i]*np.sin(image[0][i]))
    matrix.append(x)
    matrix.append(y)
    listOfMatrix.append(matrix)

# Create a new image by concatenating every 4th line from each image
newImage = []
for i in range(0, 16, 1):
    line = []
    for matrix in listOfMatrix:
        line += matrix[1][i:i+1] 
    newImage.append(line)
print(len(newImage))

# Create movie frames by repeating the plot sequence
l = 16
K = 256 // l # 16
for k in range(K):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(2,3):
        matrix = listOfMatrix[i*k % 64]
        ax.plot(matrix[0][::4], matrix[1][::4])
    plt.savefig(f'frame_{k+1:03d}.png')
    plt.show()
# Display the new image
ax.imshow(newImage, cmap='gray', extent=[0, 2*np.pi, 0, 16])
plt.show()