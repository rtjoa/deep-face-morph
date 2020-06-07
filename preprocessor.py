import cv2
import os

classifierDir = './lib/opencv/'
classifierPaths = ['haarcascade_frontalface_default.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt2.xml']
faceCascades = [cv2.CascadeClassifier(classifierDir + path) for path in classifierPaths]
minNeighborsTests = [7, 6, 5]

source_folder = './data/raw/'
dest_folder = './data/preprocessed/'
outputSize = (128,128)

def preprocess(image):
    faces = detectFaces(image)
    if not len(faces):
        raise ValueError("Face not found.")
    x, y, w, h = [v for v in faces[-1]]
    face_crop = image[y:y+h, x:x+w]
    return cv2.resize(face_crop, dsize=outputSize)
    
def detectFaces(image):
    global faceCascades
    global minNeighborsTests
    for minNeighbors in minNeighborsTests:
        for faceCascade in faceCascades:
            faces = faceCascade.detectMultiScale(image, 1.5, minNeighbors)
            if len(faces):
                return faces
    return []

if __name__ == '__main__':
    
    files = os.listdir(source_folder)
    
    for i, filename in enumerate(files):
        if os.path.exists(dest_folder + filename):
            continue
        image = cv2.imread(source_folder + filename)
        try:
            preprocessed = preprocess(image)
        except ValueError:
            continue
        cv2.imwrite(dest_folder + filename, preprocessed)
        print("{:.2f}%".format(100*i/len(files)))
        