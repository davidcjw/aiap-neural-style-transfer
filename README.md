# AIAP Neural Style Transfer

This project is done as part of AIAP Batch 5 assignment 8. In this project, we 
created a web app that showcases the concept of *style transfer*. 

Please try it out [here][5]!

## Architecture of Model

Our web app's underlying model is a pre-trained model from Tensorflow Lite. It
consists of two submodels:

1. **Style Prediction Model**: A MobilenetV2-based neural network that takes an
input style image to a 100-dimension style bottleneck vector.

2. **Style Transform Model**: A neural network that takes a style bottleneck
vector to a content image and creates a stylized image.

By default, our model uses a ``blend_ratio = 0.5``, which means that the style
and content losses are given equal weight. Using a higher blend ratio would
give more weight to the style image and vice versa. In otherwise words,
``blend_ratio = 1`` would yield a fully stylized photograph (with almost no
resemblance of the content image).

### Training of (pre-trained) model

Source: https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization#train-a-model-on-a-large-dataset-with-data-augmentation-to-run-on-mobile

Training was done on the Painter by Number (PBN) dataset [4] and Describable
Textures Dataset (DTD) [5] for the image stylization model while content images
were a combination of ImageNet, PBN and DTD images.

### Loading the Model

Since the model requires the above two components, we intended to take in user
input for *both* the content and style image. However, we realised that 
inference took ~10 seconds and we wanted to minimize that. As such, we decided
not to allow the user to upload the style image because 1) the user might not
have a style image to upload and 2) it reduces inference time significantly.

We decided to support a fixed set of styles, allowing us to compute the style
bottleneck vectors in advance. This allows us to exclude the
**Style Prediction Model** from our app. This reduces inference time from 
~10 seconds to ~2-3 seconds.

### Image Dimensions

The content and style image can take in any image with standard formats
(.jpeg, .jpg, .png) of any sizes. Image size handling is done within 
``inference.py``. 

The *style image* is resized to 256 x 256 while the *content image* will be 
central (square) cropped based on the shorter of the width/height. 

The output stylized image will be a square image with the same dimensions as
the cropped content image.

## Getting Started

### Prerequisites

We will be using Flask as our web application framework because it is lightweight
and designed to make getting started quick and easy. It wraps around Werkzeug and
Jinja2. We will also require Tensorflow and sci-kit learn for our machine learning
frameworks. Finally, we will require Pillow to deal with images.

To install TensorFlow 2.0, please refer to [TensorFlow installation page][1]
regarding the specific install command for your platform.

To install [Flask][2], please follow their installation documentation.  

To install [Pillow][3], please follow their installation documentation.

### Usage

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes
on how to deploy the project on a live system.

#### To perform inference on an image from Terminal:

```
git clone <repo_link>
cd team2
python -m src.inference path_to_base_img/base_img.jpg path_to_style_img/style_img.jpg
```

#### To perform inference on web app:

Run ``app.py`` locally by typing the commands into terminal:

```
python src/app.py
```

This will run the app on your local machine's port 8000. Copy the link into
the browser or type localhost:8000 in the url section.

Simply upload a (base) image and select the style you would like and wait a
few seconds to see the stylized version of your base image!

## Deployment

Our model is deployed on a Docker container and hosted on AISG's cluster.

The web application is created using Flask, together with Pure-CSS as the
CSS template and Vue.js as the Javascript framework.

The folder structure is as such:

    src							# Main project folder
    ├── README.md				# This README
    ├── app.py					# Main file containing Flask
    ├── inference.py            # Python file containing our Model
    │   ├── static				# Folder to contain static assets
    |	|	├── assets
    |	|	|	├──	images		# Stores various images
	|	|	├── css
	|	|	|	├── base.css	# Base CSS file that applies for all endpoints
	|	|	|	├── main.css	# CSS file for index endpoint
	|	|	|	├── model.css	# CSS file for model.html
	|	├── templates			# HTML templates folder
	|	|	├── index.html		# Contains bulk of our HTML body
	|	|	├── base.html		# Contains base HTML structure with links/scripts
	|	|	├── model.html		# Contains HTML for /model endpoint

[1]: https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available
[2]: https://pypi.org/project/Flask/
[3]: https://pillow.readthedocs.io/en/stable/installation.html
[4]: https://www.kaggle.com/c/painter-by-numbers
[5]: https://www.robots.ox.ac.uk/~vgg/data/dtd/