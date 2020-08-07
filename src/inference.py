# pylint: disable=E0401, W0611
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import argparse
import numpy as np
import codecs
import json

import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LITE_PREDICT = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_quantized_256.tflite'
LITE_TRANSFER = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_transfer_quantized_dynamic.tflite'

style_predict_path = get_file('style_predict.tflite', LITE_PREDICT)
style_transform_path = get_file('style_transform.tflite', LITE_TRANSFER)


class Inference:

    # Function to load an image from a file, and add a batch dimension.
    def load_img(self, path_to_img):
        # img = tf.io.read_file(path_to_img)
        # img = tf.io.decode_image(img, channels=3)
        img = tf.io.decode_image(path_to_img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]

        return img

    def load_style_img(self, path_to_img):
        img = Image.open(path_to_img).convert('RGB')
        img_arr = img_to_array(img)
        img_arr = tf.convert_to_tensor(img_arr, dtype=tf.float32)
        img_arr = img_arr[tf.newaxis, :]

        return img_arr

    # Function to pre-process style image input.
    def preprocess_style_image(self, style_image):
        # Resize the image so that the shorter dimension becomes 256px.
        target_dim = 256
        shape = tf.cast(tf.shape(style_image)[1:-1], tf.float32)
        short_dim = min(shape)
        scale = target_dim / short_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        style_image = tf.image.resize(style_image, new_shape)

        # Central crop the image.
        style_image = tf.image.resize_with_crop_or_pad(style_image,
                                                       target_dim,
                                                       target_dim)
        style_image /= 255.0

        return style_image

    def preprocess_content_image(self, content_image):
        # Function to pre-process content image input.

        # Central crop the image.
        shape = tf.shape(content_image)[1:-1]
        short_dim = min(shape)
        content_image = tf.image.resize_with_crop_or_pad(content_image,
                                                         short_dim,
                                                         short_dim)

        return content_image

    def open_image(self, content_path, style_path):
        # Load the input images.
        content_image = self.load_img(content_path)
        style_image = self.load_img(style_path)

        # Preprocess the input images.
        preprocessed_content_image = self.preprocess_content_image(content_image)
        preprocessed_style_image = self.preprocess_style_image(style_image)

        return preprocessed_content_image, preprocessed_style_image

    # Function to run style prediction on preprocessed style image.
    def run_style_predict(self, preprocessed_style_image):
        # Load the model.
        interpreter = tf.lite.Interpreter(model_path=style_predict_path)

        # Set model input.
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

        # Calculate style bottleneck.
        interpreter.invoke()
        style_bottleneck = interpreter.tensor(
            interpreter.get_output_details()[0]["index"])()

        return style_bottleneck

    # Run style transform on preprocessed style image
    def run_style_transform(self, style_bottleneck, preprocessed_content_image):
        # Load the model.
        interpreter = tf.lite.Interpreter(model_path=style_transform_path)

        # Set model input.
        input_details = interpreter.get_input_details()
        interpreter.resize_tensor_input(input_details[0]["index"],
                                        preprocessed_content_image.shape)
        interpreter.allocate_tensors()

        # Set model inputs.
        interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
        interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
        interpreter.invoke()

        # Transform content image.
        stylized_image = interpreter.tensor(
            interpreter.get_output_details()[0]["index"])()

        return stylized_image

    # def imshow(self, image, title=None):
    #     if len(image.shape) > 3:
    #         image = tf.squeeze(image, axis=0)
    #
    #     plt.imshow(image)
    #     if title:
    #         plt.title(title)
    #
    #     plt.show()


def prerun_stylebottleneck(self):
    # To pre-prepare Style Bottleneck

    StylePath = 'img/'
    style_path = StylePath+'Paul_Klee_19.jpg'

    # load style
    style_image = self.load_img(style_path)

    # Preprocess the input images.
    preprocessed_style_image = self.preprocess_style_image(style_image)

    # To pre-run style bottleneck for selected artists
    style_bottleneck = self.run_style_predict(preprocessed_style_image)

    style_bottleneck_list = style_bottleneck.tolist()

    # Set the file_path
    file_path = ("paul_klee.json")

    json.dump(style_bottleneck_list, codecs.open(file_path, 'w', encoding='utf-8'),
              separators=(',', ':'), sort_keys=True, indent=4)  # this saves the array in .json format


if __name__ == '__main__':

    BasePath = 'img/'
    #StylePath = 'img/'
    content_path = BasePath+'download.jfif'
    #style_path = StylePath+'Paul_Klee_19.jpg'

    # jason file_path
    jason_path = ("paul_klee.json")

    # Define content blending ratio between [0..1].
    # 0.0: 0% style extracts from style image.
    # 1.0: 100% style extracted from content image.

    blend_ratio = 0.5

    # parse the command line argument
    parser = argparse.ArgumentParser(description='Style Image')
    parser.add_argument('contentimage', default=content_path)
    #parser.add_argument('styleimage', default=style_path)
    parser.add_argument('blendratio', default=blend_ratio)

    args = parser.parse_args()

    # Init Inference class
    inf = Inference()

    # Load the input images.
    content_image = inf.load_img(args.contentimage)
    #style_image = inf.load_img(args.styleimage)

    # Preprocess the input images.
    preprocessed_content_image = inf.preprocess_content_image(content_image)
    #preprocessed_style_image = inf.preprocess_style_image(style_image)

    # Calculate style bottleneck for the preprocessed style image.
    #style_bottleneck = inf.run_style_predict(preprocessed_style_image)

    # import the json file and un-jasonfy style_bottleneck
    obj_text = codecs.open(jason_path, 'r', encoding='utf-8').read()
    style_jason = json.loads(obj_text)
    style_bottleneck = np.array(style_jason,  dtype=np.float32)

    # Calculate style bottleneck for content image (for blending purpose)
    style_bottleneck_content = inf.run_style_predict(
        preprocess_style_image(content_image))

    # Ratio to blend the images
    content_blending_ratio = args.blendratio
    content_blending_ratio = float(content_blending_ratio)

    # Blend the style bottleneck of style image and content image
    style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
        + (1 - content_blending_ratio) * style_bottleneck

    # Stylize the content image using the style bottleneck.
    stylized_image = inf.run_style_transform(style_bottleneck_blended, preprocessed_content_image)
    inf.imshow(stylized_image, 'Stylized Image')
