import qrcode

qr_img = qrcode.make("https://qpredict.onrender.com")

qr_img.save("qr-img.jpg")