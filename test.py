from PIL import Image

# 建立一張 1280x720 的白色圖片
w, h = 150, 220
img = Image.new("RGB", (w, h), (255, 255, 255))
img.show()
