import cv2 as cv

# open CV Parser
class OpenCV:

    def get_face(self, path):

        face_cascade = cv.CascadeClassifier('../haar/haarcascade_frontalface_default.xml')
        img = cv.imread(path)
        img2 = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            #roi_gray = gray[y:y + h, x:x + w]
            img_face = img2[y:y + h, x:x + w]

        return img, img_face

    def img_show(self, img):

        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def img_write(self, path, img):
        cv.imwrite(path, img)