
from open_cv import OpenCV


cv = OpenCV()

img, face = cv.get_face('../bases/1.png')
cv.img_show(img)
cv.img_show(face)
cv.img_write('../bases/result.png', face)

