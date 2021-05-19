from deepFaceModel import DeepFaceModel
from preprocessor import preprocess
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
from threading import Thread

forceRebuild = True

# Model settings
# Typically 'data/preprocessed/' or 'data/autocrop-preprocessed/'
dataFolder = 'data/autocrop-preprocessed/'
epochs = 50
split_validation = True # whether to use a small portion of data for validation
patience = 5 # if using validation: stop after this many rounds w/o improvement

# Model structure
inputShape = (128,128,3)
filterIncreasePerConv = 30
convolutions = 4
stride = 2
kernelSize = 3
latentDim = 400

# Transformed latent vector of currently shown face
z = np.zeros(latentDim)

# Set a specific feature of z
def adjustFeature(featureIndex, value):
    global z
    global lastSlide
    z[featureIndex] = value
    updateImage()
    lastSlide = time.time() 
    def checkForRelease():
        time.sleep(0.2)
        if time.time() - lastSlide >= 0.2:
            updateLookalikeButton()
    Thread(target=checkForRelease).start()

# Update the image panel with the current z
def updateImage():
    img = model.transformedLatentToImg(z)
    imgTk = ImageTk.PhotoImage(Image.fromarray(img))
    facePanel.configure(image = imgTk)
    facePanel.image = imgTk

# Calculate lookalike and show name on button
def updateLookalikeButton():
    lookalikeButton.config(text = model.lookalike(z)['name'])

# Update sliders to reflect current z
def updateSliders():
    for slider, zVal in zip(sliders,z):
        slider.set(zVal)

# Load and morph to local image
def loadImage():
    filename = tk.filedialog.askopenfilename(title = "Select file",filetypes = (("Image files","*.jpg *.jpeg *.png"),("All files","*.*")))
    if not filename:
        return
    try:
        img = preprocess(cv2.imread(filename))
    except ValueError as e:
        tk.messagebox.showinfo("Deep Face Morph", e.args[0])
        return
    latent = model.encoder.predict(np.array([img]))[0]
    loadZ(model.eigenBasis.T @ latent)

# Morph to current z's lookalike 
def loadLookalike():
    loadZ(model.lookalike(z)['z'])

# Animate morph to a new z
def loadZ(newZ):
    def update():
        global z
        diff = newZ - z
        updateImageThread = Thread()
        for i in range(20):
            time.sleep(0.1)
            z += diff/20
            if True or not updateImageThread.isAlive():
                updateImageThread = Thread(target = updateImage)
                updateImageThread.start()
        updateSliders()
        updateLookalikeButton()
    Thread(target=update).start()

try:
    assert not forceRebuild
    model = DeepFaceModel.load('model')
    print("Loaded model")
except Exception:
    print("Building new model")
    model = DeepFaceModel()
    model.build(inputShape, latentDim, filterIncreasePerConv, convolutions, stride, kernelSize)
    model.train(dataFolder, epochs, split_validation, patience)
    model.save('model')

z = model.transformedLatents[1].copy()

root = tk.Tk()
root.wm_title("Deep Face Morph")
root.iconbitmap('assets/favicon.ico')
root.resizable(False, False)

rightPanel = tk.Frame(root)
rightPanel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

rightMidPanelWrapper = tk.Frame(rightPanel)
rightMidPanelWrapper.pack(side=tk.TOP, expand=1)

rightMidPanel = tk.Frame(rightMidPanelWrapper)
rightMidPanel.pack(side=tk.RIGHT)

facePanel = tk.Label(rightMidPanel)
facePanel.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

loadImageButton = tk.Button(rightMidPanel, text="Load face", command=loadImage)
loadImageButton.pack(side=tk.BOTTOM)

rightBottomPanel = tk.Frame(rightPanel)
rightBottomPanel.pack(side=tk.BOTTOM)

lookalikeText = tk.Label(rightBottomPanel, text="You look like")
lookalikeText.pack(side=tk.TOP)

lookalikeButton = tk.Button(rightBottomPanel, text="", command=loadLookalike)
lookalikeButton.pack(side=tk.BOTTOM)

# Create sliders panel
frame = tk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH)
sliders = []
mins = model.transformedLatents.min(0)
maxs =  model.transformedLatents.max(0)
for i in reversed(range(14)):
    command = lambda value, feature=i: adjustFeature(feature, value)
    slider = tk.Scale(frame, from_=mins[i], to=maxs[i], orient=tk.HORIZONTAL, command=command, showvalue=0)
    slider.set(int(z[i]))
    slider.pack(side=tk.BOTTOM)
    sliders.insert(0,slider)

updateImage()
updateLookalikeButton()
root.mainloop()