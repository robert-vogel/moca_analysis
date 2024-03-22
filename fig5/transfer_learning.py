"""Train TensorFlow models and print predictions of validation set

By: Robert Vogel


Note: base modesl will be downloaded from TensorFlow, consequently
an internet connection is required for running script

In applying the TensorFlow models to this data set and our purpose
we followed the Transfer Learning instructions published 
in the 
[TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
tutorials.  Parts of our code will match, or be minor modifications,
that in this tutorial created
and shared by Google under the Apache 2.0 License.
"""

import sys
import os
import argparse
import tensorflow as tf


IMG_CHANS = 3
IMG_PX_NUM = 128
WEIGHTS = "imagenet"
EPOCHS=1

CLASS_ENCODING={
        "benign":"0",
        "malignant":"1"
        }

def _parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir",
                        type=str,
                        required=True,
                        help="Path to directory containing training data.")
    parser.add_argument("--validate_dir",
                        type=str,
                        required=True,
                        help="Path to directory containing validation data.")

    parser.add_argument("--validate_set_metadata",
                        type=str,
                        required=True,
                        help=("Path to validation set images labels and"
                              " class"))

    parser.add_argument("--out_dir",
                        type=str,
                        required=True,
                        help=("Path to directory in which validation"
                              " scores are."))

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed for random number generation.")
    return parser.parse_args()
 

def mk_model(keras_base_model, base_model_preprocessing):
    """Make model with preprocessing and classification.

    Args:
        keras_base_model: tf.keras.Model
        base_model_preprocessing: 
            keras provided tools for correctly preprocessing
            data according to a specific model

    Returns:
        tf.keras.Model instance with preprocessing,
        removal the default top (classification) layer, and
        replace with binary classification with sigmoidal
        activation.
    """

    keras_base_model = keras_base_model(input_shape=(IMG_PX_NUM,
                                                     IMG_PX_NUM,
                                                     IMG_CHANS),
                                        include_top=False,
                                        weights=WEIGHTS)

    # my custom layers
    resize = tf.keras.layers.Resizing(IMG_PX_NUM, IMG_PX_NUM)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect")
        ])

    inference = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
        ])


    inputs = tf.keras.Input(shape=(None, None, IMG_CHANS))
    x = resize(inputs)
    x = base_model_preprocessing(x) 
    x = data_augmentation(x)
    x = keras_base_model(x, training=False)
    outputs = inference(x)

    base_model = tf.keras.Model(inputs, outputs)

    base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                       loss=tf.keras.losses.BinaryCrossentropy())
    return base_model


def mk_models():
    return [mk_model(tf.keras.applications.mobilenet.MobileNet,
                     tf.keras.applications.mobilenet.preprocess_input),
            mk_model(tf.keras.applications.inception_v3.InceptionV3,
                     tf.keras.applications.inception_v3.preprocess_input),
            mk_model(tf.keras.applications.resnet50.ResNet50,
                     tf.keras.applications.resnet50.preprocess_input),
            mk_model(tf.keras.applications.nasnet.NASNetMobile,
                     tf.keras.applications.nasnet.preprocess_input),
            mk_model(tf.keras.applications.vgg19.VGG19,
                     tf.keras.applications.vgg19.preprocess_input),
            mk_model(tf.keras.applications.xception.Xception,
                     tf.keras.applications.xception.preprocess_input)]


def main(args):
    args = _parse_args(args)

    data = tf.keras.utils.image_dataset_from_directory(
            args.train_dir,
            labels="inferred",
            label_mode="binary",
            seed=args.seed)

    models = mk_models()

    for model in models:
        model.fit(data, epochs=EPOCHS)

    with (open(os.path.join(args.out_dir, "predictions.csv"), "w") as fout,
          open(args.validate_set_metadata, "r") as fin):

        header = ["sample_id", "class"]

        for i in range(len(models)):
            header.append(f"model_{i}")

        fout.write(','.join(header))

        for tline in fin:

            tline = tline.strip()
            tline = tline.split('\t')
            tline[1] = CLASS_ENCODING[tline[1]]

            img_fname = os.path.join(args.validate_dir,
                                     f"{tline[0]}.jpeg")

            img = tf.image.decode_jpeg(tf.io.read_file(img_fname))
            img = tf.reshape(img, (1, *img.shape))

            for model in models:
                tmp = model(img, training=False).numpy()
                tline.append(str(tmp.squeeze()))

            tline = ','.join(tline)
            fout.write(f"\n{tline}")


if __name__ == "__main__":
    main(sys.argv[1:])
