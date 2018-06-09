import coconet

picture_sizex = 256
picture_sizey = 256
address = './butterfly_GT.bmp'
img, filename = coconet.load_image(address)


X = coconet.generate_placeholder_tensor(picture_sizex, picture_sizey, trimensional = True)
Y = coconet.generate_value_tensor(img, picture_sizex, picture_sizey, trimensional = True)
model = coconet.generate_model_conv([64], dim = 5)

history = model.fit(X, Y, epochs = 1000, batch_size = 65536)
prediction = coconet.predict(model, X, picture_sizex, picture_sizey)

print(coconet.compare_images(img, prediction))
coconet.save_image(prediction, 'butterfly_conv.png')

"""
X = coconet.generate_placeholder_tensor(picture_sizex, picture_sizey)
Y = coconet.generate_value_tensor(img, picture_sizex, picture_sizey)
model = coconet.generate_model_dense([32] * 10)

history = model.fit(X, Y, epochs = 1000, batch_size = 65536)
prediction = coconet.predict(model, X, picture_sizex, picture_sizey)

print(coconet.compare_images(img, prediction))
coconet.save_image(prediction, 'butterfly_dense.png')
"""
