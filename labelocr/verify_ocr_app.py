import atexit
import glob
import json
import logging
import os
import shutil
import sys
import tkinter as tk
import threading
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
import pygubu
from PIL import Image, ImageTk
from deprecated import deprecated

PROJECT_PATH = os.path.dirname(__file__)
PROJECT_UI = os.path.join(PROJECT_PATH, "verify_ocr.ui")


FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("LabelOcr")

class VerifyOcrApp:
    def __init__(self, master):
        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)
        builder.add_from_file(PROJECT_UI)
        self.master = master
        self.mainwindow = builder.get_object('master', master)
        builder.connect_callbacks(self)

        self.config_dir = os.path.join(os.path.expanduser("~"),".ocr_labeling")
        self.last_session_path = os.path.join(self.config_dir, 'last_session_verify_ocr')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)

        self.image_dir = builder.get_variable("var_img_dir")
        self.label_path = builder.get_variable("var_label_path")

        self.image_name = builder.get_variable("var_image_name")
        self.label_ocr = builder.get_variable("var_label")
        self.cur_index = builder.get_variable("var_cur_index")


    def load_data(self):
        if self.image_dir.get() is not None and self.label_path.get() is not None and len(
                self.image_dir.get()) > 0 and len(self.label_path.get()) > 0:
            if self.label_in_filename:
                self.list_file = list(glob.glob(f"{self.image_dir.get()}/*.png"))
                self.list_label = [os.path.splitext(os.path.basename(file))[0] for file in self.list_file]
                self.list_label = [self._parse_label(x) for x in self.list_label]
            else:
                df_label = pd.read_csv(self.label_path.get(), header=0, names=['filename', 'label'], dtype={"filename": str, "label": str})
                self.list_file = df_label['filename'].tolist()
                self.list_label = df_label['label'].tolist()
            self._show_image()
        else:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found label to save.")



def main():
    root = tk.Tk()
    app = VerifyOcrApp(root)
    app.run()