from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle

#variables
source = "models"
dest = "results"
choice_set = ("c", "k", "g", "x")
yes_or_no_set = ("yes", "no")
#scaler = StandardScaler()
classification_model = None
kmeans_model = None
gan_model = None
class_scaler = None
kmeans_scaler = None
gan_disc = None

#load models
try: 
    class_path = os.path.join(source, "classification_model.h5")
    kmeans_path = os.path.join(source, "kmeans_model.h5")
    gan_path = os.path.join(source, "gen_model.h5")
    disc_path = os.path.join(source, "disc_model.h5")
    classification_model = load_model(class_path)
    gan_model = load_model(gan_path)
    gan_disc = load_model(disc_path)
    with open(kmeans_path, "rb") as f:
        kmeans_model = pickle.load(f)
except Exception as e:
     print(f"Error with loading model: {e}")
print("Loaded models for app")

try:
    class_scaler_path = os.path.join("scalers", "classification_scaler.pkl")
    kmeans_scaler_path = os.path.join("scalers", "kmeans_scaler.pkl")
    gan_scaler_path = os.path.join("scalars", "gan_scaler.pkl")

    with open(class_scaler_path, "rb") as f:
        class_scaler = pickle.load(f)
except Exception as e:
    print(f"Error with loading scaler: {e}")
print("Loaded scalers for app")

#functions
def classification(model):
    if model is None:
        print("Error: Classification model is not loaded.")
        return
    print("Please ener your performance information.")

   

    #Hours Studied
    while True:
        hs = input(f"How many hours have you used to study for your class?")
        if hs.isdigit():
            if int(hs) < 0:
                hs = 0
            break
        else:
            print("Please enter a valid number")
    #Previous Score
    while True:
        ps = input(f"What is your prevous score?")
        if ps.isdigit():
            if int(ps) < 0:
                ps = 0
            break
        else:
            print("Please enter a valid number")
    #Extracurricular activities
    while True:
        exc = input(f"Do you have any extracurricular activities? Yes or No?")
        if exc.lower() not in yes_or_no_set:
            print("Enter Yes or No.")
        else:
            exc_final = 1 if exc.lower == "yes" else 0
            break
    #Sleep Hours
    while True:
        sl = input(f"How many hours do you usually sleep for?")
        if ps.isdigit():
            if int(sl) < 0:
                sl = 0
            break
        else:
            print("Please enter a valid number")
    #Sample Question Papers Practiced
    while True:
        sqp = input(f"How many sample question papers have you done?")
        if sqp.isdigit():
            if int(sl) < 0:
                sqp = 0
            break
        else:
            print("Please enter a valid number")

    model_input = np.array([int(hs), int(ps), int(exc_final), int(sl), int(sqp)])
    model_input = model_input.reshape((1, 5))
    model_input = class_scaler.transform(model_input)

    #model_input = model_input.reshape(1, -1)

    try:
        prediction = model.predict(model_input)
        pred_perc = prediction[0][0] * 100
        print(f"You have a {pred_perc:.2f}% chance of passing their exams")
    except Exception as e:
        print(f"Error with model prediction: {e}")
    print("Returning to app...\n")

def check_float(i):
    if i.count(".") == 1:
        i = i.replace(".", "")
    return i.isdigit()

def kmeans(model):
    if model is None:
        print("Error: Classification model is not loaded.")
        return
    sAge = None
    sStudy = None
    sAbsences = None
    sGPA = None
    sExtra = None
    print("Please enter you student's information")

    
    name = input("What is your student's name?")
        
    while True:
        sAge = input(f"What is {name}'s age?")
        if sAge.isdigit():
            if int(sAge) >= 20:
                print(f"{name} is too old to be in school")
                print("Returning to app...\n")
                return
            break
        else:
            print("Please enter a valid number")

    while True:
        sStudy = input(f"How many hours does {name} study every week?")
        if sStudy.isdigit():
            if int(sStudy) > 20:
                print(f"{name} is studying too much every week")
                sStudy = 20
            break
        else:
            print("Please enter a valid number")

    while True:
        sAbsences = input(f"How many absences does {name} have this school year?")
        if sAbsences.isdigit():
            if int(sAbsences) > 30:
                print(f"{name} has missed more than 30 classes this school year")
                sAbsences = 30
            break
        else:
            print("Please enter a valid number")
    
    while True:
        sExtra = input(f"Does {name} do any extracurricular activities? Yes or No")
        if sExtra.lower() in yes_or_no_set:
            sExtra = 1 if sExtra.lower() == "yes" else 0
            break
        else:
            print("Please enter a valid response")

    while True:
        sGPA = input(f"What is {name}'s current GPA?")
        if check_float(sGPA):
            if float(sGPA) < 0 or float(sGPA) > 4.0:
                print("Please enter a valid number")
                continue
            else:
                break
        else:
            print("Please enter a valid number")

    model_input = np.array([int(sAge), int(sStudy), int(sAbsences), int(sExtra), float(sGPA)])
    model_input = model_input.reshape(1, -1)

    try:
        prediction = model.predict(model_input)
        if prediction[0] in set([1, 4, 6, 7]):
            print(f"{name} is currently performing well in their academics")
        elif prediction[0] in set([0, 2, 3, 5]) and float(sGPA) < 2.5:
            print(f"{name} is currently not performing well in their academics and requires significant assistance")
        elif prediction[0] in set([0, 2, 3, 5]) and (float(sGPA) >= 2.5 and float(sGPA) < 3.4):
            print(f"{name} is doing alright, but could use some assistance")
    except Exception as e:
        print(f"Error with model prediction: {e}")
    
    print("Returning to app...\n")

def gan(model, disc):
    while True:
        num_inputs = input("How many pictures do you want to generate? (up to 100)")
        if num_inputs.isdigit():
            
            num_inputs = int(num_inputs)

            if num_inputs < 0 or num_inputs > 100:
                print("Please input a number between 0 and 100")
                continue
            else:
                break
        else:
            print("Please input a valid number")
    dest = "results"
    latent_dim = 100
    noise = np.random.normal(0, 1, (num_inputs, latent_dim))
    image_list = []


    try:
        print("Generating images")
        images = model.predict(noise)

        #for debugging
        #print(f"Generated image shape: {images.shape}\n{images[0, 0, 0]}")  

        images = ((images + 1) * 127.5).astype(np.uint8)
    except Exception as e:
        print(f"Error with image generation: {e}")

    print("Saving images")

    try:
        for i in images:
            curr_date = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            img_bgr = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            file_path = os.path.join(dest, f"image_{curr_date}.jpg")
            success = cv2.imwrite(file_path, img_bgr)
            if not success:
                print(f"Failed to save {file_path}")
    except Exception as e:
        print(f"Error with saving images: {e}")

    print("Returning to app...\n")

#main code
while True:
    choice = input("What do you want to do?\nc - check likelihood of passing your exams\nk - group students into learning styles\ng - generate artwork using AI\nx - exit\n").lower()
    if choice not in choice_set:
        print("Enter a valid choice")
        continue
    #check likelihood of passing
    elif choice == "c":
        classification(classification_model)
    #group students into learning styles
    elif choice == "k":
        kmeans(kmeans_model)
    #generate artwork using AI
    elif choice == "g":
        gan(gan_model, gan_disc)
    elif choice == "x":
        print("Exiting app")
        break