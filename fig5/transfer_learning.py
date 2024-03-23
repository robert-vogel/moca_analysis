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
import glob
import re
import tensorflow as tf


IMG_CHANS = 3
IMG_PX_NUM = 128
WEIGHTS = "imagenet"

CLASS_ENCODING={
        "benign":"0",
        "malignant":"1"
        }

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

    keras_base_model.trainable=False

    # my custom layers
    resize = tf.keras.layers.Resizing(IMG_PX_NUM, IMG_PX_NUM)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect")
        ])

    inference = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
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


def progress_bar(iter_num, total_iters, total_chars=50):
    frac = iter_num / total_iters
    progress = int(total_chars * frac)
    complement = total_chars - progress 

    print_end = '\r'

    if iter_num == total_iters:
        print_end = "\n"

    print(f"|{progress *'*'}{complement * '-'}| {frac:0.2f}",
          end = print_end)


def mk_models():
    models = {
                "mobilenet": (tf.keras.applications.mobilenet.MobileNet,
                            tf.keras.applications.mobilenet.preprocess_input),
                "inception_v3": (tf.keras.applications.inception_v3.InceptionV3,
                                tf.keras.applications.inception_v3.preprocess_input),
                "resnet50": (tf.keras.applications.resnet50.ResNet50,
                            tf.keras.applications.resnet50.preprocess_input),
                "nasnet": (tf.keras.applications.nasnet.NASNetMobile,
                            tf.keras.applications.nasnet.preprocess_input),
                "vgg19": (tf.keras.applications.vgg19.VGG19,
                            tf.keras.applications.vgg19.preprocess_input),
                "xception" : (tf.keras.applications.xception.Xception,
                            tf.keras.applications.xception.preprocess_input)
            }

    for key, val in models.items():
        yield key, mk_model(*val)


def _run_train(args):
    data = tf.keras.utils.image_dataset_from_directory(
            args.train_dir,
            labels="inferred",
            label_mode="binary",
            seed=args.seed)

    ckpt_dir = "checkpoints"
    for model_name, model in mk_models():

        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        ckpt_fname = os.path.join(ckpt_dir, f"cpkt_{model_name}.weights.h5")

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_fname,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         save_freq="epoch")


        model.fit(data,
                  epochs=args.epochs,
                  callbacks = [ckpt_callback])


def _run_predict(args):
    # load models and weights

    model_file_names = glob.glob(os.path.join(args.ckpt_dir,
                                              "*.h5"))

    model_file_names.sort()

    models = list(mk_models())

    # checkpoint files have model name 
    # embedded, therefore a simple search is sufficient

    for model_file_name in model_file_names:
        for mname, model in models:

            if re.match(mname, model_file_name) is None:
                continue

            model.load_weights(model_file_name)
            break

    with open(args.validate_set_metadata, "r") as fin:
        for total_lines, tline in enumerate(fin):
            pass

    with (open(os.path.join(args.out_dir,
                f"sample_scores.csv"), "w") as fout,
          open(args.validate_set_metadata, "r") as fin):

        header = ["sample_id", "class"]

        for i in range(len(models)):
            header.append(f"model_{i}")

        fout.write(','.join(header))

        for i, tline in enumerate(fin):
            progress_bar(i, total_lines)

            tline = tline.strip()
            tline = tline.split('\t')
            tline[1] = CLASS_ENCODING[tline[1]]

            img_fname = os.path.join(args.validate_dir,
                                     f"{tline[0]}.jpeg")

            img = tf.image.decode_jpeg(tf.io.read_file(img_fname))
            img = tf.reshape(img, (1, *img.shape))

            for _, model in models:
                tmp = model(img, training=False).numpy()
                tline.append(str(tmp.squeeze()))

            tline = ','.join(tline)
            fout.write(f"\n{tline}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title="subcommands",
                                       dest="subparser_name")

    train_parser = subparsers.add_parser("train")

    train_parser.add_argument("--train_dir",
                        type=str,
                        required=True,
                        help="Path to directory containing training data.")
    train_parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed for random number generation.")
    train_parser.add_argument("--epochs",
                               type=int,
                               default=10,
                               help="Number of epochs for training")


    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--validate_dir",
                        type=str,
                        required=True,
                        help="Path to directory containing validation data.")

    predict_parser.add_argument("--validate_set_metadata",
                        type=str,
                        required=True,
                        help=("Path to validation set images labels and"
                              " class"))
    predict_parser.add_argument("--ckpt_dir",
                                type=str,
                                required=True,
                                help="path do directory of TensorFlow ckpts")
    predict_parser.add_argument("--out_dir",
                        type=str,
                        required=True,
                        help=("Path to directory in which validation"
                              " scores are."))

    args = parser.parse_args(sys.argv[1:])

    print(args.subparser_name)

    if args.subparser_name == "train":
        _run_train(args)

    if args.subparser_name == "predict":
        _run_predict(args)
