from model import DeepFaceModel
from preprocessor import preprocess
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
from threading import Thread

FORCE_REBUILD = False

# Model settings
DATA_PATH = 'data/preprocessed/'
EPOCHS = 50
SPLIT_VALIDATION = True # whether to use a small portion of data for validation
PATIENCE = 5 # if using validation: stop after this many rounds w/o improvement

# Model structure
INPUT_SHAPE = (128,128,3)
FILTER_INCREASE_PER_CONV = 30
CONVOLUTIONS = 4
STRIDE = 2
KERNEL_SIZE = 3
LATENT_DIM = 400

# Transformed latent vector of currently shown face
z = np.zeros(LATENT_DIM)

# Set a specific feature of z
def adjust_feature(feature_index, value):
    global z
    global last_slide
    z[feature_index] = value
    update_image()
    last_slide = time.time()
    def check_for_release():
        time.sleep(0.2)
        if time.time() - last_slide >= 0.2:
            update_lookalike_button()
    Thread(target=check_for_release).start()

# Update the image panel with the current z
def update_image():
    img = model.transformed_latent_to_img(z)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img))
    face_panel.configure(image = img_tk)
    face_panel.image = img_tk

# Calculate lookalike and show name on button
def update_lookalike_button():
    lookalike_button.config(text = model.lookalike(z)['name'])

# Update sliders to reflect current z
def update_sliders():
    global z
    for slider, z_val in zip(sliders, z):
        slider.set(z_val)

# Load and morph to local image
def load_image():
    filename = tk.filedialog.askopenfilename(
        title = "Select file",
        filetypes = (("Image files","*.jpg *.jpeg *.png"),("All files","*.*")))
    if filename:
        try:
            img = preprocess(cv2.imread(filename))
            latent = model.encoder.predict(np.array([img]))[0]
            load_z(model.eigenbasis.T @ latent)
        except ValueError as e:
            tk.messagebox.showinfo("Deep Face Morph", e.args[0])

# Morph to current z's lookalike
def load_lookalike():
    load_z(model.lookalike(z)['z'])

# Animate morph to a new z
def load_z(new_z):
    def update():
        global z
        diff = new_z - z
        update_image_thread = Thread()
        for i in range(20):
            time.sleep(0.1)
            z += diff/20
            if True or not update_image_thread.isAlive():
                update_image_thread = Thread(target = update_image)
                update_image_thread.start()
        update_sliders()
        update_lookalike_button()
    Thread(target=update).start()

try:
    assert not FORCE_REBUILD
    model = DeepFaceModel.load('model')
    print("Loaded model")
except Exception:
    print("Building new model")
    model = DeepFaceModel()
    model.build(INPUT_SHAPE, LATENT_DIM, FILTER_INCREASE_PER_CONV,
        CONVOLUTIONS, STRIDE, KERNEL_SIZE)
    model.train(DATA_PATH, EPOCHS, SPLIT_VALIDATION, PATIENCE)
    model.save('model')

z = model.transformed_latents[1].copy()

# Create GUI

root = tk.Tk()
root.wm_title("Deep Face Morph")
root.iconbitmap('assets/favicon.ico')
root.resizable(False, False)

right_panel = tk.Frame(root)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

right_mid_panel_wrapper = tk.Frame(right_panel)
right_mid_panel_wrapper.pack(side=tk.TOP, expand=1)

right_mid_panel = tk.Frame(right_mid_panel_wrapper)
right_mid_panel.pack(side=tk.RIGHT)

face_panel = tk.Label(right_mid_panel)
face_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

load_image_button = tk.Button(right_mid_panel, text="Load face",
                              command=load_image)
load_image_button.pack(side=tk.BOTTOM)

right_bottom_panel = tk.Frame(right_panel)
right_bottom_panel.pack(side=tk.BOTTOM)

lookalike_text = tk.Label(right_bottom_panel, text="You look like")
lookalike_text.pack(side=tk.TOP)

lookalike_button = tk.Button(right_bottom_panel, text="",
                             command=load_lookalike)
lookalike_button.pack(side=tk.BOTTOM)

# Create sliders panel
frame = tk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH)
sliders = []
mins = model.transformed_latents.min(0)
maxs =  model.transformed_latents.max(0)
for i in reversed(range(14)):
    command = lambda value, feature=i: adjust_feature(feature, value)
    slider = tk.Scale(frame, from_=mins[i], to=maxs[i], orient=tk.HORIZONTAL,
                      command=command, showvalue=0)
    slider.set(int(z[i]))
    slider.pack(side=tk.BOTTOM)
    sliders.insert(0,slider)

update_image()
update_lookalike_button()
root.mainloop()
