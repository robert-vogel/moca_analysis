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
import json
import argparse
import glob
import re
import numpy as np
import tensorflow as tf


IMG_CHANS = 3
IMG_PX_NUM = 128
WEIGHTS = "imagenet"
INTERPOLATION="bilinear"

CLASS_ENCODING={
        "benign":"0",
        "malignant":"1"
        }

def mk_model(keras_base_model, base_model_preprocessing,
             l1_reg_constant,learning_rate):
    """Make model with preprocessing and classification.

    Args:
        keras_base_model: tf.keras.Model
        base_model_preprocessing: 
            keras provided tools for correctly preprocessing
            data according to a specific model
        l1_reg_constant: (float)
            postive float of l1 regularization constant applied to
            bias and weights

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
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        ])


    l1_reg_bias = tf.keras.regularizers.L1(l1_reg_constant)
    l1_reg_kernel = tf.keras.regularizers.L1(l1_reg_constant)

    inference = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1,
                    activation="sigmoid",
                    kernel_regularizer=l1_reg_kernel,
                    bias_regularizer=l1_reg_bias)
        ])


    inputs = tf.keras.Input(shape=(None,None, IMG_CHANS))
    x = data_augmentation(inputs)
    x = base_model_preprocessing(x) 
    x = keras_base_model(x, training=False)
    outputs = inference(x)

    base_model = tf.keras.Model(inputs, outputs)

    base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5,
                                                               name='accuracy'),
                                tf.keras.metrics.AUC(name='auc')])
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


def mk_models(l1_reg_constant, learning_rate):
    models = {
                "mobilenet": (tf.keras.applications.mobilenet.MobileNet,
                            tf.keras.applications.mobilenet.preprocess_input),
                "inception_v3": (tf.keras.applications.inception_v3.InceptionV3,
                                tf.keras.applications.inception_v3.preprocess_input),
                #                "resnet50": (tf.keras.applications.resnet50.ResNet50,
                #                            tf.keras.applications.resnet50.preprocess_input),
                #                "nasnet": (tf.keras.applications.nasnet.NASNetMobile,
                #                            tf.keras.applications.nasnet.preprocess_input),
                "xception" : (tf.keras.applications.xception.Xception,
                            tf.keras.applications.xception.preprocess_input)
            }

    for key, val in models.items():
        yield key, mk_model(*val, l1_reg_constant, learning_rate)


def _run_train(args):
    data = tf.keras.utils.image_dataset_from_directory(
            args.train_dir,
            shuffle=True,
            batch_size=32,
            labels="inferred",
            interpolation=INTERPOLATION,
            label_mode="binary",
            image_size=(IMG_PX_NUM,IMG_PX_NUM),
            seed=args.seed)

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    with open(os.path.join(args.ckpt_dir, "train_pars.json"), "w") as fout:
        parameters = args.__dict__.copy()
        parameters["img_px_num"] = IMG_PX_NUM
        parameters["img_chans"] = IMG_CHANS
        parameters["weights"] = WEIGHTS
        json.dump(parameters, fout, indent=2)


    for model_name, model in mk_models(args.l1_reg_constant, args.learning_rate):

        ckpt_fname = os.path.join(args.ckpt_dir, f"cpkt_{model_name}.weights.h5")

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_fname,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         save_freq="epoch")


        model.fit(data,
                  epochs=args.epochs,
                  callbacks = [ckpt_callback])


def _run_predict(args):
    # load models and weights

    with open(os.path.join(args.ckpt_dir, "train_pars.json"), "r") as fid:
        train_pars = json.load(fid)

    model_file_names = glob.glob(os.path.join(args.ckpt_dir,
                                              "*.h5"))
    img_size = (train_pars["img_px_num"], train_pars["img_px_num"])

    model_file_names.sort()

    models = list(mk_models(train_pars["l1_reg_constant"],
                            train_pars["learning_rate"]))

    # checkpoint files have model name 
    # embedded, therefore a simple search is sufficient

    for model_file_name in model_file_names:
        for mname, model in models:

            if re.search(mname, model_file_name) is None:
                continue

            print(f"loading {mname} weights")
            model.load_weights(model_file_name)
            break

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    print("read in image set")
    positive_files = glob.glob(os.path.join(args.positive_dir,'*.jpeg'))
    negative_files = glob.glob(os.path.join(args.negative_dir, "*.jpeg"))

    all_files = positive_files + negative_files
    sample_class = np.zeros(len(all_files))
    sample_class[:len(positive_files)] = 1.

    images = []
    sample_label = []

    for file_name in all_files:
        sample_label.append(os.path.basename(file_name))

        images.append(tf.keras.utils.load_img(file_name,
                                               target_size=img_size,
                                               interpolation=INTERPOLATION))

    images = np.array(images)
    predictions = np.zeros(shape=(images.shape[0], len(models)))

    print("running predictions")

    i = 0
    progress_bar(i, len(models))
    for mname, model in models:
        predictions[:,i] = model.predict(images,
                                         verbose=0).squeeze()
        i += 1
        progress_bar(i, len(models))

    predictions = predictions.astype(str)


    str_out = ""
    for k in range(predictions.shape[0]):
        str_out += f'\n{sample_label[k]},{sample_class[k]},'
        str_out += ','.join(predictions[k,:])


    print("writing to file")
    with open(os.path.join(args.out_dir, "sample_scores.csv"),"w") as fout:
        header = ["sample_id", "class"]
        for i in range(len(models)):
            header.append(f"model_{i+1:02d}")

        fout.write(','.join(header))

        fout.write(str_out)


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
    train_parser.add_argument("--l1_reg_constant",
                              type=float,
                              default=0.01,
                              help="Set regularization constant during training.")
    train_parser.add_argument("--epochs",
                               type=int,
                               default=10,
                               help="Number of epochs for training")
    train_parser.add_argument("--ckpt_dir",
                              type=str,
                              default="checkpoints",
                              help="Name of directory to store model weights")
    train_parser.add_argument("--learning_rate",
                                type=float,
                                default=0.0001,
                                help="Learning rate")


    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--positive_dir",
                        type=str,
                        required=True,
                        help="Directory of positive class samples")
    predict_parser.add_argument("--negative_dir",
                        type=str,
                        required=True,
                        help="Directory of negative class samples")
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
