
from open_cv import OpenCV
import numpy as np


cv = OpenCV()

img = cv.img_read('../bases/1.png')
equ = cv.img_equalize_histogram(img)

img = cv.img_color2gray(img)

res = np.hstack((img,equ)) #stacking images side-by-side
cv.img_show(res)

cv.img_write('../bases/equalize.png', res)

