import coconet
import cv2

picture_sizex = 192
picture_sizey = 256
#enhance = 1
enhance = 4

address = './61944293.png'

img, filename = coconet.load_image(address)

X = coconet.generate_placeholder_tensor(picture_sizex, picture_sizey, trimensional = True)
X_SR = coconet.generate_placeholder_tensor(picture_sizex, picture_sizey, trimensional = True, enhance = enhance)
Y = coconet.generate_value_tensor(img, picture_sizex, picture_sizey, trimensional = True)
model = coconet.generate_model_conv([32] * 10, dim = 7, slen = 1)


history = model.fit(X, Y, epochs = 1000)
#history = model.fit(X, Y, epochs = 1000, batch_size = 1024)
prediction = coconet.predict(model, X_SR, picture_sizex * enhance, picture_sizey * enhance)

coconet.save_image(prediction, address[:-4] + ' FUSED.png')
