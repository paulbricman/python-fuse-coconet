import coconet

picture_sizex = 32
picture_sizey = 32
address = './13_horse.png'
img, filename = coconet.load_image(address)

X = coconet.generate_placeholder_matrix(picture_sizex, picture_sizey)
Y = coconet.generate_value_matrix(img, picture_sizex, picture_sizey)
model = coconet.generate_model_dense([100])

history = model.fit(X, Y, epochs = 3000, batch_size = 1024)
prediction = coconet.predict(model, X, picture_sizex, picture_sizey)

print(coconet.compare_images(img, prediction))
