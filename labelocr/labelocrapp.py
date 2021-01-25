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
# import tensorflow as tf
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

        self.config_dir = os.path.join(os.path.expanduser("~"),".ocr_labeling")
        self.last_session_path = os.path.join(self.config_dir, 'last_session')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)

        self.image_dir = builder.get_variable("var_img_path")
        self.label_path = builder.get_variable("var_label_path")
        self.label_in_filename = None  # Not use in the future
        self.image_name = builder.get_variable("var_image_name")
        self.label_ocr = builder.get_variable("var_label")
        self.cur_index = builder.get_variable("var_cur_index")
        self.txt_label = builder.get_object("txt_label")

        self.index_goto = builder.get_variable("var_index_goto")

        self.canvas = builder.get_object("canvas")
        self.lbl_image = builder.get_object("lbl_image")

        self.progress_label = builder.get_variable("var_progress_label")
        self.progress_generate = builder.get_variable("var_progress_generate")
        self.progress_generate.set(0)

        self.output_path = builder.get_variable("var_output_path")
        self.start_label = builder.get_variable("var_start_label")
        self.end_label = builder.get_variable("var_end_label")

        self.data_name = builder.get_variable("var_data_name")
        self.output_width = builder.get_variable("var_output_width")
        self.output_width.set(225)
        self.output_height = builder.get_variable("var_output_height")
        self.output_height.set(75)
        self.charsets = builder.get_variable("var_charset")
        self.max_len = builder.get_variable("var_max_len")
        self.max_len.set(10)
        self.test_size = builder.get_variable("var_test_size")
        self.test_size.set(0.1)

        self.list_file = None
        self.list_label = None
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
            df = pd.DataFrame({"filename": self.list_file, "label": self.list_label})
            df.to_csv(self.label_path.get(), index=False)

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
                self.index += 1
                self._show_image()

    def prev_img(self, event=None):
        if self.list_label is None or self.list_file is None:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found label to next.")
            return
        if self.txt_label.focus_get() != ".!entry":
            if self.index > 0:
                self._save_label()
                self.index -= 1
                self._show_image()

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

    def generate_tf(self):
        img_dir = self.image_dir.get()
        label_file = self.label_path.get()
        start_index = self.start_label.get()
        end_index = self.end_label.get()
        output_dir = self.output_path.get()
        data_name = self.data_name.get()
        output_w = self.output_width.get()
        output_h = self.output_height.get()
        charset = self.charsets.get()
        max_len = self.max_len.get()
        test_size = self.test_size.get()

        thread = threading.Thread(target=tf_records,
                                  args=([img_dir], label_file, start_index, end_index, output_dir, data_name, charset,
                                        (output_w, output_h, 3),
                                        max_len, test_size, self.progress_generate))
        thread.setDaemon(True)
        thread.start()

    def delete_image(self):
        if self.list_label is None or self.list_file is None:
            messagebox.showerror("Input Error", "Please choose folder image and label file.")
            LOGGER.info("Not found label to delete.")
            return
        self.list_file.pop(self.index)
        self.list_label.pop(self.index)
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
        LOGGER.info("Load image: {}: {}".format(filename, label))
        self.image_name.set(filename)

        self.cur_img = img_origin = Image.open(os.path.join(self.image_dir.get(), filename))
        img_resized = self.resize_image(img_origin, (600, 100))
        self.is_resized = True
        img_resized = ImageTk.PhotoImage(img_resized)

        self.lbl_image.configure(image=img_resized)
        self.lbl_image.image = img_resized
        # self.canvas.create_image(0, 0, image=img_origin, anchor=tk.NW)

        self.label_ocr.set(label)

        self.progress_label.set((self.index + 1) / len(self.list_label))
        self.cur_index.set(f"Index: {self.index + 1}/{len(self.list_file)}")

    def _save_last_config(self):
        LOGGER.info("Save session...")
        self.save_all()
        config = {
            "index": self.index,
            "img_dir": self.image_dir.get(),
            "label_path": self.label_path.get(),
            "output_path": self.output_path.get(),
            "start_label": self.start_label.get(),
            "end_label": self.end_label.get(),
            "dataset_name": self.data_name.get(),
            "output_width": self.output_width.get(),
            "output_height": self.output_height.get(),
            "charsets": self.charsets.get(),
            "max_len": self.max_len.get(),
            "test_size": self.test_size.get()
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
                self.image_dir.set(config['img_dir'])
                self.label_path.set(config['label_path'])
                if 'output_path' in config:
                    self.output_path.set(config['output_path'])
                    self.start_label.set(config['start_label'])
                    self.end_label.set(config['end_label'])
                    self.data_name.set(config['dataset_name'])
                    self.output_width.set(config['output_width'])
                    self.output_height.set(config['output_height'])
                    self.charsets.set(config['charsets'])
                    self.max_len.set(config['max_len'])
                    self.test_size.set(config['test_size'])

    def _save_label(self):
        if self.index is not None and self.index >= 0:
            self.list_label[self.index] = self.label_ocr.get().strip()

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


NULL_CHAR = '<nul>'
LONGEST_TEXT = ''
CHARSET_CONTAINS = set()


def generate_charset(charset: str, output_path: str):
    output_file = output_path + "=" + str(len(charset) + 1) + ".txt"
    f = open(output_file, 'w')
    f.write(f"{0}\t{NULL_CHAR}\n")
    for i, c in enumerate(charset):
        f.write(f"{i + 1}\t{c}\n")
    f.close()
    return output_file


def get_char_mapping(charset_file):
    char_mapping = {}
    rev_char_mapping = {}
    with open(charset_file, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            if line[-1] != " ":
                id, char = line.split()
            else:
                id = line.strip()
                char = " "
            id = int(id)
            char_mapping[char] = id
            rev_char_mapping[id] = char
    if NULL_CHAR not in char_mapping:
        char_mapping[NULL_CHAR] = 200
        rev_char_mapping[200] = NULL_CHAR
    return char_mapping, rev_char_mapping


def encode_utf8_string(text, charset, length):
    for c in text:
        CHARSET_CONTAINS.add(c)
    char_ids_unpadded = [charset[c] for c in text]
    char_ids_padded = char_ids_unpadded + [charset[NULL_CHAR] for i in range(length - len(char_ids_unpadded))]
    return char_ids_unpadded, char_ids_padded


def decode_string(ids, char_mapping):
    chars = [char_mapping[id] for id in ids]
    for id in ids:
        c = char_mapping[id]
        if c == NULL_CHAR:
            break
        chars.append(c)
    return ''.join(chars)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def letterbox_resize(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding

    :param
        image: origin image to be resize
        target_size: target image size,
            tuple of format (width, height).

    :returns
    """
    src_h, src_w, src_c = image.shape
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale_w = target_w / src_w
    scale_h = target_h / src_h

    if scale_w > scale_h:
        new_h = target_h
        new_w = int(scale_h * src_w)
    else:
        return cv2.resize(image, (target_w, target_h))

    mask = np.zeros(shape=(target_h, target_w, src_c), dtype=np.uint8)
    tmp_img = cv2.resize(image, (new_w, new_h))
    tmp_img = np.reshape(tmp_img, (new_h, new_w, src_c))
    mask[0:new_h, 0:new_w] = tmp_img
    return mask


def _tf_example(img_file, label, char_mapping, image_shape, max_str_len, num_of_views):
    w, h, c = image_shape
    img_array = cv2.imread(img_file)
    if c == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = np.expand_dims(img_array, axis=-1)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = letterbox_resize(img_array, (w, h))
    retval, image = cv2.imencode('.png', img_array)
    image_data = image.tostring()
    # image = tf.image.convert_image_dtype(img_array, dtype=tf.uint8)
    # image = tf.image.encode_png(image)
    # image_data = sess.run(image)
    # img = gfile.FastGFile(img_file, 'rb').read()

    char_ids_unpadded, char_ids_padded = encode_utf8_string(label, char_mapping, max_str_len)

    features = tf.train.Features(feature={
        'image/format': _bytes_feature([b"PNG"]),
        'image/encoded': _bytes_feature([image_data]),
        'image/class': _int64_feature(char_ids_padded),
        'image/unpadded_class': _int64_feature(char_ids_unpadded),
        'image/width': _int64_feature([img_array.shape[1]]),
        'image/orig_width': _int64_feature([int(img_array.shape[1] / num_of_views)]),
        'image/text': _bytes_feature([label.encode('utf-8')])
    })

    example = tf.train.Example(features=features)
    return example


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


def gen_code_data(output_dir, data_name, traing_size, test_size, charset_filename, image_shape, num_of_views,
                  max_sequence_length, null_code):
    code = f"""from . import fsns

DEFAULT_DATASET_DIR = 'datasets/data/{data_name}'

DEFAULT_CONFIG = {{
    'name':
        '{data_name}',
    'splits': {{
        'train': {{
            'size': {traing_size},
            'pattern': 'train*'
        }},
        'test': {{
            'size': {test_size},
            'pattern': 'test*'
        }}
    }},
    'charset_filename':
        '{charset_filename}',
    'image_shape': ({image_shape[1]}, {image_shape[0]}, {image_shape[2]}),
    'num_of_views':
        {num_of_views},
    'max_sequence_length':
        {max_sequence_length},
    'null_code':
        {null_code},
    'items_to_descriptions': {{
        'image':
            'A [150 x 600 x 3] color image.',
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }}
}}


def get_split(split_name, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = DEFAULT_CONFIG

  return fsns.get_split(split_name, dataset_dir, config) 
"""
    print(code, file=open(os.path.join(output_dir, "{}.py".format(data_name)), 'w'))


def tf_records(img_dir, labels_file, start_index, end_index, output_dir, data_name, charset_file, image_shape, max_len,
               test_size=0.2,
               progress_bar=None):
    char_mapping, rev_char_mapping = get_char_mapping(charset_file)

    output_dir_data = os.path.join(output_dir, 'data', data_name)
    os.makedirs(output_dir_data, exist_ok=True)
    train_file = os.path.join(output_dir_data, 'train.tfrecord')
    test_file = os.path.join(output_dir_data, 'test.tfrecord')
    charset_filename = os.path.basename(charset_file)
    output_charset_file = os.path.join(output_dir_data, charset_filename)
    if not os.path.exists(output_charset_file):
        shutil.copyfile(charset_file, output_charset_file)

    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    train_writer = tf.io.TFRecordWriter(train_file)
    test_writer = tf.io.TFRecordWriter(test_file)

    if labels_file is not None:
        label_df = pd.read_csv(labels_file, header=0, names=['filename', 'label'], dtype={"filename": str, "label": str})
    else:
        label_df = get_labels_in_filename(img_dir)
    label_df = label_df.iloc[start_index: end_index]
    label_df = label_df.sample(frac=1)
    n_train = int(len(label_df) * (1 - test_size))
    n_test = len(label_df) - n_train

    print("Split data: {} for trains and {} for test".format(n_train, n_test))

    global LONGEST_TEXT
    total = len(label_df)
    j = 1
    for i, row in label_df.iterrows():
        img_path = row['filename']
        if not img_path.startswith('/'):
            for dir in img_dir:
                img_path = os.path.join(dir, os.path.basename(row['filename']))
                if os.path.exists(img_path):
                    break
        else:
            for dir in img_dir:
                img_path = os.path.join(dir, os.path.basename(row['filename']))
                if os.path.exists(img_path):
                    break
        label = row['label']
        if len(LONGEST_TEXT) < len(label):
            LONGEST_TEXT = label

        if progress_bar is not None:
            progress_bar.set(j / total)

        example = _tf_example(img_file=img_path,
                              label=label,
                              char_mapping=char_mapping,
                              image_shape=image_shape,
                              max_str_len=max_len,
                              num_of_views=1)

        if j < n_train:
            train_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())

        j += 1

    train_writer.close()
    test_writer.close()
    gen_code_data(output_dir=output_dir,
                  data_name=data_name,
                  traing_size=n_train,
                  test_size=n_test,
                  charset_filename=charset_filename,
                  image_shape=image_shape,
                  num_of_views=1,
                  max_sequence_length=max_len,
                  null_code=char_mapping[NULL_CHAR]
                  )

    print("============> Complete export tfrecords file <==============")
    print("Longest text: '{}' has length is {}".format(LONGEST_TEXT, len(LONGEST_TEXT)))
    print("CHARSET CONTAINS: {} characters: {}".format(len(CHARSET_CONTAINS), sorted(CHARSET_CONTAINS)))
    print("Train file has: {:6} samples, that is saved at {}".format(n_train, train_file))
    print("Test file has : {:6} samples, that is saved at {}".format(n_test, test_file))


def main():
    root = tk.Tk()
    app = LabelOcrApp(root)
    app.run()
