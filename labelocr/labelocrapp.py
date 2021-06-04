import atexit
import csv
import glob
import json
import logging
import os
import sys
import tkinter as tk
import unicodedata
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
import pygubu
from PIL import Image, ImageTk
from deprecated import deprecated

PROJECT_PATH = os.path.dirname(__file__)
PROJECT_UI = os.path.join(PROJECT_PATH, "label_ocr.ui")

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("LabelOcr")


class LabelOcrApp:
    def __init__(self, master):
        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)
        builder.add_from_file(PROJECT_UI)
        self.master = master
        self.mainwindow = builder.get_object('master', master)
        builder.connect_callbacks(self)

        self.config_dir = os.path.join(os.path.expanduser("~"), ".ocr_labeling")
        self.last_session_path = os.path.join(self.config_dir, 'last_session')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)

        self.image_dir = builder.get_variable("var_img_path")
        self.label_path = builder.get_variable("var_label_path")
        self.label_in_filename = None  # Not use in the future
        self.image_name = builder.get_variable("var_image_name")
        self.label_ocr = builder.get_variable("var_label")
        self.label_ocr_show = builder.get_variable("var_label_show")
        self.cur_index = builder.get_variable("var_cur_index")
        self.txt_label = builder.get_object("txt_label")
        self.label_score = builder.get_variable('var_score')
        self.filter_score = builder.get_variable("var_filter_score")

        self.index_goto = builder.get_variable("var_index_goto")

        self.canvas = builder.get_object("canvas")
        self.lbl_image = builder.get_object("lbl_image")

        self.btn_keep_exist_label = builder.get_object("btn_keep_exist_label")
        self.keep_exist_label = True
        self.btn_keep_exist_label.select()

        self.btn_remove_accent = builder.get_object("btn_remove_accent")
        self.remove_accent = False

        self.btn_upper = builder.get_object("btn_upper")
        self.upper_case = False

        self.btn_title = builder.get_object("btn_title")
        self.title_case = False

        self.btn_lower = builder.get_object("btn_lower")
        self.lower_case = False

        self.progress_label = builder.get_variable("var_progress_label")

        self.list_file = None
        self.list_label = None
        self.list_score = None
        self.index = None

        self.cur_img = None
        self.is_resized = True
        self.scale_width_img = builder.get_variable("var_scale_width_img")

    def choose_image_dir(self):
        image_dir = filedialog.askdirectory(
            initialdir=PROJECT_PATH
        )
        LOGGER.info("Open image dir:", image_dir)
        self.image_dir.set(image_dir)

    def choose_label_file(self):
        label_path = filedialog.askopenfilename(initialdir=self.image_dir.get())
        LOGGER.info("Open label file: ", label_path)
        self.label_path.set(label_path)

    def load_data(self):
        if self.image_dir.get() is not None and self.label_path.get() is not None and len(
                self.image_dir.get()) > 0 and len(self.label_path.get()) > 0 and os.path.exists(
            self.image_dir.get()) and os.path.exists(self.label_path.get()):
            if self.label_in_filename:
                self.list_file = list(glob.glob(f"{self.image_dir.get()}/*.png"))
                self.list_label = [os.path.splitext(os.path.basename(file))[0] for file in self.list_file]
                self.list_label = [self._parse_label(x) for x in self.list_label]
            else:
                df_label = pd.read_csv(self.label_path.get(), header=0,
                                       dtype={"filename": object, "label": object}, keep_default_na=False,
                                       na_values=[''])
                self.list_file = df_label['filename'].tolist()
                self.list_label = df_label['label'].tolist()
                if 'conf' in df_label.columns:
                    self.list_score = df_label['conf'].tolist()
            if self.index >= len(self.list_file):
                self.index = len(self.list_file) - 1
            self._show_image()
        else:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found label to save.")

    def change_txt_label(self, event=None):
        self._save_label()

    def save_all(self):
        if self.image_dir.get() is not None and self.label_path.get() is not None and len(
                self.image_dir.get()) > 0 and len(self.label_path.get()) > 0:
            LOGGER.info("Save all....")
            if self.list_file is None or self.list_label is None:
                messagebox.showerror("Input Error", "Please choose folder image and label file.")
                LOGGER.info("Not found label to save.")
                return
            data = {"filename": self.list_file, "label": self.list_label}
            if self.list_score is not None:
                data['conf'] = self.list_score
            df = pd.DataFrame(data)
            df.to_csv(self.label_path.get(), index=False, quoting=csv.QUOTE_ALL)

    def _get_next_idx(self, is_next=True) -> int:
        current_idx = self.index
        step = 1 if is_next else -1
        if self.list_score is None:
            return current_idx + step
        filter_score = self.filter_score.get().strip()
        try:
            filter_score = float(filter_score)
            if is_next:
                for i in range(current_idx + 1, len(self.list_score)):
                    if self.list_score[i] < filter_score:
                        return i
            else:
                for i in range(current_idx - 1, 0, -1):
                    if self.list_score[i] < filter_score:
                        return i

        except:
            return current_idx + step

    def next_img(self, event=None):
        if self.list_label is None or self.list_file is None:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found label to next.")
            return
        if self.txt_label.focus_get() != ".!entry":
            if self.index < len(self.list_file) - 2:
                self._save_label()
                if self.index % 50 == 0:
                    self.save_all()
                self.index = self._get_next_idx(True)
                self._show_image()

    def prev_img(self, event=None):
        if self.list_label is None or self.list_file is None:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found label to next.")
            return
        if self.txt_label.focus_get() != ".!entry":
            if self.index > 0:
                self._save_label()
                self.index = self._get_next_idx(False)
                self._show_image()

    def change_keep_exist_label(self, event=None):
        self.keep_exist_label = not self.keep_exist_label
        LOGGER.info(f"Change_keep_exist_label: {self.keep_exist_label}")
        if self.keep_exist_label:
            self.btn_keep_exist_label.select()
        else:
            self.btn_keep_exist_label.deselect()
        self._show_image()

    def change_remove_accent(self, event=None):
        self.remove_accent = not self.remove_accent
        LOGGER.info(f"change_remove_accent: {self.remove_accent}")
        if self.remove_accent:
            self.btn_remove_accent.select()
        else:
            self.btn_remove_accent.deselect()
        self._show_image()

    def change_upper(self, event=None):
        if self.title_case:
            self.change_title()
        if self.lower_case:
            self.change_lower()

        self.upper_case = not self.upper_case
        LOGGER.info(f"change_upper: {self.upper_case}")
        if self.upper_case:
            self.btn_upper.select()
        else:
            self.btn_upper.deselect()
        self._show_image()

    def change_title(self, event=None):
        if self.upper_case:
            self.change_upper()
        if self.lower_case:
            self.change_lower()

        self.title_case = not self.title_case
        LOGGER.info(f"change_title: {self.title_case}")

        if self.title_case:
            self.btn_title.select()
        else:
            self.btn_title.deselect()
        self._show_image()

    def change_lower(self, event=None):
        if self.title_case:
            self.change_title()
        if self.upper_case:
            self.change_upper()

        self.lower_case = not self.lower_case
        LOGGER.info(f"change_lower: {self.lower_case}")

        if self.lower_case:
            self.btn_lower.select()
        else:
            self.btn_lower.deselect()
        self._show_image()

    def clean_text(self, event=None):
        self.label_ocr.set("")

    def choose_output_tf(self):
        output_dir = filedialog.askdirectory(
            initialdir=PROJECT_PATH
        )
        LOGGER.info("Choose output dir: {}".format(output_dir))
        self.output_path.set(output_dir)

    def resize_img(self):
        if self.is_resized:
            img = ImageTk.PhotoImage(self.cur_img)
        else:
            img = ImageTk.PhotoImage(self.resize_image(self.cur_img, (600, 100)))

        self.lbl_image.configure(image=img)
        self.lbl_image.image = img
        self.is_resized = not self.is_resized

    def scale_image(self, event=None):
        new_width = self.scale_width_img.get()
        if new_width >= 6:
            img = ImageTk.PhotoImage(self.resize_image(self.cur_img, (new_width, int(new_width / 6))))
            self.lbl_image.configure(image=img)
            self.lbl_image.image = img
            self.is_resized = True

    def delete_image(self):
        if self.list_label is None or self.list_file is None:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found label to delete.")
            return
        self.list_file.pop(self.index)
        self.list_label.pop(self.index)
        if self.list_score:
            self.list_score.pop(self.index)
        self._show_image()

    def goto(self):
        if self.list_label is None or self.list_file is None:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found label to goto.")
            return
        input_index = self.index_goto.get()
        if len(self.list_file) > input_index >= 0:
            self._save_label()
            self.index = input_index - 1
            self._show_image()

    def run(self):
        sys.excepthook = self.show_exception_and_exit
        self._read_last_config()
        if self.image_dir.get() is not None and self.label_path.get() is not None and len(
                self.image_dir.get()) > 0 and len(self.label_path.get()) > 0:
            self.load_data()

        self.master.bind('<Up>', self.next_img)
        self.master.bind('<Down>', self.prev_img)
        self.master.bind('<Return>', self.next_img)
        self.master.bind('<Escape>', self.clean_text)
        self.master.protocol('WM_DELETE_WINDOW', self._save_last_config)
        self.mainwindow.mainloop()
        atexit.register(self._save_last_config)

    @deprecated(reason="Not use in the future because label is load from csv file")
    def _parse_label(self, text):
        return text.split("_")[1].strip()

    def _show_image(self):
        if self.list_label is None or self.list_file is None:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found image to show.")
            return
        filename = os.path.basename(self.list_file[self.index])
        label = self.list_label[self.index]
        if self.list_score is not None:
            self.label_score.set(self.list_score[self.index])
        LOGGER.info("Load image: {}: {}".format(filename, label))
        self.image_name.set(filename)

        self.cur_img = img_origin = Image.open(os.path.join(self.image_dir.get(), filename))
        img_resized = self.resize_image(img_origin, (600, 100))
        self.is_resized = True
        img_resized = ImageTk.PhotoImage(img_resized)

        self.lbl_image.configure(image=img_resized)
        self.lbl_image.image = img_resized

        self.scale_image()
        # self.canvas.create_image(0, 0, image=img_origin, anchor=tk.NW)
        if self.remove_accent:
            label = remove_accents(label)

        if self.upper_case:
            label = label.upper()

        if self.lower_case:
            label = label.lower()

        if self.title_case:
            label = label.title()

        self.label_ocr_show.set(label)
        if self.keep_exist_label:
            self.label_ocr.set(label)
        else:
            self.label_ocr.set("")

        self.progress_label.set((self.index + 1) / len(self.list_label))
        self.cur_index.set(f"Index: {self.index + 1}/{len(self.list_file)}")

    def _save_last_config(self):
        LOGGER.info("Save session...")
        self.save_all()
        config = {
            "index": self.index,
            "img_dir": self.image_dir.get(),
            "label_path": self.label_path.get(),
        }
        with open(self.last_session_path, 'w') as f:
            json.dump(config, f)

        self.master.quit()

    def _read_last_config(self):
        if os.path.exists(self.last_session_path):
            LOGGER.info("Read last session....")
            with open(self.last_session_path, 'r') as f:
                config = json.load(f)
                self.index = config['index']
                if self.index is None:
                    self.index = 0
                self.image_dir.set(config['img_dir'])
                self.label_path.set(config['label_path'])

    def _save_label(self):
        if self.index is not None and self.index >= 0:
            if len(self.label_ocr.get().strip()) > 0:
                self.label_ocr_show.set(self.label_ocr.get())
            self.list_label[self.index] = self.label_ocr_show.get().strip()

    def show_exception_and_exit(self, exc_type, exc_value, tb):
        import traceback
        traceback.print_exception(exc_type, exc_value, tb)
        self._save_last_config()
        sys.exit(-1)

    def resize_image(self, image, size=(1000, 150)):
        image = np.asarray(image)
        src_h, src_w = image.shape[:2]
        target_w, target_h = size

        # calculate padding scale and padding offset
        scale_w = target_w / src_w
        scale_h = target_h / src_h

        if scale_w > scale_h:
            new_h = target_h
            new_w = int(scale_h * src_w)
        else:
            new_w = target_w
            new_h = int(scale_w * src_h)
            return Image.fromarray(cv2.resize(image, (target_w, target_h)).astype(np.uint8))

        mask = np.zeros(shape=(target_h, target_w, 3), dtype=np.uint8)
        tmp_img = cv2.resize(image, (new_w, new_h))

        mask[0:new_h, 0:new_w] = tmp_img
        return Image.fromarray(mask.astype(np.uint8))


def remove_accents(input_str):
    if input_str is None:
        return None
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode('utf8')


NULL_CHAR = '<nul>'
LONGEST_TEXT = ''
CHARSET_CONTAINS = set()


def get_labels_in_filename(img_dirs: list):
    file_paths = []
    labels = []
    for dir in img_dirs:
        for file in glob.glob(dir + "/*.png"):
            filename = os.path.basename(file)[:-4]
            img_index, info_index, label = filename.split("_")
            file_paths.append(file)
            labels.append(label)
    return pd.DataFrame({"filename": file_paths, "label": labels})


def main():
    root = tk.Tk()
    app = LabelOcrApp(root)
    app.run()


main()
