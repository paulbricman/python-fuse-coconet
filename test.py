import coconet
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import cv2

picture_sizex = 32
picture_sizey = 32
enhance = 1
#enhance = 4

address = askopenfilename()

img, filename = coconet.load_image(address)

X = coconet.generate_placeholder_tensor(picture_sizex, picture_sizey)
X_SR = coconet.generate_placeholder_tensor(picture_sizex, picture_sizey, enhance = enhance)
Y = coconet.generate_value_tensor(img, picture_sizex, picture_sizey)
model = coconet.generate_model_dense([100] * 10)


#history = model.fit(X, Y, epochs = 1000, batch_size = 128, shuffle = True)
history = model.fit(X, Y, epochs = 1000, batch_size = 1024)
prediction = coconet.predict(model, X_SR, picture_sizex * enhance, picture_sizey * enhance)

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))
plt.show()

coconet.save_image(prediction, address[:-4] + ' FUSED.png')
