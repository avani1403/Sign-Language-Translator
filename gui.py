# Importing Libraries

import numpy as np
import tkinter
from tkinter import *
import cv2
import os, sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

#from hunspell import Hunspell
#import enchant

from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application :

class Application:

    def __init__(self):

        #self.hs = Hunspell('en_US')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open("model/model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("model/model-bw.h5")

        self.json_file_dru = open("model/model-bw_dru.json", "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()

        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("model/model-bw_dru.h5")
        self.json_file_tkdi = open("model/model-bw_tkdi.json", "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()

        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights("model/model-bw_tkdi.h5")
        self.json_file_smn = open("model/model-bw_smn.json", "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()

        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("model/model-bw_smn.h5")


        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")


        self.root = tk.Tk()
        self.root.title("Sign Language To Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.configure(bg="#0f4b6e")
        self.root.state("zoomed")

        self.scr_width = self.root.winfo_screenwidth()
        self.scr_height = self.root.winfo_screenheight()

        self.f3 = tkinter.Frame(self.root, width = self.scr_width//2, height = self.scr_height//2)
        self.f3.grid(row = 1, column = 0, padx = 10, pady = 7)

        self.panel = tk.Label(self.f3,width=self.scr_width//2, height=self.scr_height//2)
        self.panel.grid(row = 1, column = 0)

        self.panel2 = tk.Label(self.f3, width = 275, height = 275)  # initialize image panel
        self.panel2.place(x = 420, y = 0)

        #image part
        self.photosign = tkinter.PhotoImage(file="american_sign_language1.png")
        w6 = tkinter.Label(self.root, image=self.photosign, width=self.scr_width//2 - 20, height=self.scr_height // 2)
        w6.grid(row=1, column=1, padx=10)

        #description part
        self.description = "American Sign Language (ASL) is a natural language. It is the primary sign language used by the deaf and people\n with hearing impairment in the USA and Canada. Translation of text to sign language is also be given as a task\n during sign language study session. This tool can easily produce the correct answers and because the visual stays on\n screen, students can follow the hand movements at their own pace."
        descr = tkinter.Label(self.root, text=self.description, bg="sky blue", font=("arial", 20))
        descr.grid(row=3, column=0, columnspan=2)

        self.T  = tk.Label(self.root, text="Sign Language to Text", bg="sky blue", font=("times new roman", 35))
        self.T.grid(row=0, column=0)

        self.T  = tk.Label(self.root,text="Sign Language Guide" ,bg="sky blue",  font=("times new roman", 35))
        self.T.grid(row=0, column=1)

        self.f1 = tkinter.Frame(self.root, bg="sky blue")
        self.f1.grid(row = 2, column = 0)

        # for predicted value
        self.panel3 = tk.Label(self.f1, width = 5)
        self.panel3.grid(row = 1, column = 0 ,  pady = 7)

        self.T1 = tk.Label(self.f1)
        self.T1.grid(row = 0, column = 0)
        self.T1.config(text="Predicted Value", bg = "sky blue", font=("times new roman", 25))

        self.f2 = tkinter.Frame(self.root, bg = "sky blue")
        self.f2.grid(row=2, column=1, padx = 10, pady = 10)
        self.panel4 = tk.Label(self.f2, width = 15)  # Word
        self.panel4.grid(row = 1, column = 0, padx = 10, pady = 10)

        self.T2 = tk.Label(self.f2)
        self.T2.grid(row = 0, column = 0)
        self.T2.config(text="Word", bg = "sky blue", font=("times new roman", 25))



        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            cv2image = cv2image[y1: y2, x1: x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)

            imgtk = ImageTk.PhotoImage(image=self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            self.panel3.config(text=self.current_symbol, font=("times new roman", 45))

            self.panel4.config(text=self.word, font=("times new roman", 45))



        self.root.after(5, self.video_loop)

    def predict(self, test_image):

        test_image = cv2.resize(test_image, (128, 128))

        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))

        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))

        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))

        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))

        prediction = {}

        prediction['blank'] = result[0][0]

        inde = 1

        for i in ascii_uppercase:
            prediction[i] = result[0][inde]

            inde += 1

        # LAYER 1

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

        self.current_symbol = prediction[0][0]

        # LAYER 2

        if (self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
            prediction = {}

            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]

            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

            self.current_symbol = prediction[0][0]

        if (
                self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
            prediction = {}

            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]

            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

            self.current_symbol = prediction[0][0]

        if (self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):

            prediction1 = {}

            prediction1['M'] = result_smn[0][0]
            prediction1['N'] = result_smn[0][1]
            prediction1['S'] = result_smn[0][2]

            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)

            if (prediction1[0][0] == 'S'):

                self.current_symbol = prediction1[0][0]

            else:

                self.current_symbol = prediction[0][0]

        if (self.current_symbol == 'blank'):

            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if (self.ct[self.current_symbol] > 60):

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 20:
                    self.ct['blank'] = 0

                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0

            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':

                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "

                    self.str += self.word

                    self.word = ""

            else:

                if (len(self.str) > 16):
                    self.str = ""

                self.blank_flag = 0

                self.word += self.current_symbol

    def action1(self):

        predicts = self.hs.suggest(self.word)

        if (len(predicts) > 0):
            self.word = ""

            self.str += " "

            self.str += predicts[0]

    def action2(self):

        predicts = self.hs.suggest(self.word)

        if (len(predicts) > 1):
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):

        predicts = self.hs.suggest(self.word)

        if (len(predicts) > 2):
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def action4(self):

        predicts = self.hs.suggest(self.word)

        if (len(predicts) > 3):
            self.word = ""
            self.str += " "
            self.str += predicts[3]

    def action5(self):

        predicts = self.hs.suggest(self.word)

        if (len(predicts) > 4):
            self.word = ""
            self.str += " "
            self.str += predicts[4]

    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()