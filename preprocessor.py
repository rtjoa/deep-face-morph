import cv2
import os

SOURCE_FOLDER = './data/raw/'
DEST_FOLDER = './data/preprocessed/'

CLASSIFIER_DIR = './lib/opencv/'
CLASSIFIER_FILES = [
    'haarcascade_frontalface_default.xml',
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_alt2.xml']

MIN_NEIGHBORS_TESTS = [7, 6, 5] # Detect faces with decreasing strictness
OUTPUT_SIZE = (128, 128)

face_cascades = [cv2.CascadeClassifier(CLASSIFIER_DIR + file) for file in CLASSIFIER_FILES]

def preprocess(image, face_cascades, min_neighbors_tests, output_size):
    faces = detect_faces(image, face_cascades, min_neighbors_tests)
    if not len(faces):
        raise ValueError("Face not found.")
    x, y, w, h = [v for v in faces[-1]]
    face_crop = image[y:y+h, x:x+w]
    return cv2.resize(face_crop, dsize=output_size)
    
def detect_faces(image, face_cascades, min_neighbors_tests):
    for min_neighbors in min_neighbors_tests:
        for face_cascade in face_cascades:
            faces = face_cascade.detectMultiScale(image, 1.5, min_neighbors)
            if len(faces):
                return faces
    return []

if __name__ == '__main__':
    files = os.listdir(SOURCE_FOLDER)
    
    for i, filename in enumerate(files):
        if not os.path.exists(DEST_FOLDER + filename):
            image = cv2.imread(SOURCE_FOLDER + filename)
            try:
                preprocessed = preprocess(image, face_cascades,
                                          MIN_NEIGHBORS_TESTS, OUTPUT_SIZE)
                cv2.imwrite(DEST_FOLDER + filename, preprocessed)
            except ValueError:
                pass
        print("{:.2f}%".format(100*i/len(files)))
