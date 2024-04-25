import os
import tensorflow as tf
import numpy as np


def get_id_img(directory):
    """
    :param directory: a string representing the directory
    :return: a dictionary mapping each class to image paths for that class
    """
    res = {}
    for identity in os.listdir(directory):
        temp_dir = os.path.join(directory, identity)
        lst = []
        for img_path in os.listdir(temp_dir):
            lst.append(os.path.join(directory, identity, img_path))
        res[int(identity)] = lst

    return res


def get_triplet_dataset(data):
    """
    Data are split into [anchor, positive, negative] each of which are image paths
    :param data: should be a dictionary mapping classes to their list of image paths
    :return: A list containing [anchor, positive, negative] for each sample
    """
    res = []
    cnt = 0
    num_classes = len(data)
    min_class = min(data.keys())
    for identity in data.keys():
        id_images = data[identity]
        for i in range(len(id_images) - 1):
            # Anchor image
            anchor_path = id_images[i]
            # Take an image with the same identity for positive sample
            positive_path = id_images[i + 1]

            # Take an image with different identity for negative sample
            addition = np.random.randint(1, num_classes)
            negative_id = (identity + addition) % num_classes + min_class
            neg_id_images = data[negative_id]
            negative_path = neg_id_images[i]

            res.append([anchor_path, positive_path, negative_path])

        cnt += 1
        print(f'\r{cnt:>5} out of {len(data.keys()):>5}', end='')
    print()
    return res


def load_dataset_from_directory(train_dir,
                                valid_dir,
                                AUTOTUNE=tf.data.experimental.AUTOTUNE):
    """
    Creates traininig and validation pipeline for Siamese-Network
    :param train_dir: str representing directory to the training set
    :param valid_dir: str representing directory to the validation set
    :param AUTOTUNE:
    :return:
        train_ds: a pipeline that returns [anchor, positive, negative] images
        valid_ds: a pipeline that returns [anchor, positive, negative] images
    """
    def _get_image(path):
        file = tf.io.read_file(path)
        img = tf.image.decode_png(file, channels=3)
        return img

    def _get_images(anchor_path, positive_path, negative_path):
        anchor = _get_image(anchor_path)
        positive = _get_image(positive_path)
        negative = _get_image(negative_path)
        return anchor, positive, negative


    def _get_dataset(inputs):
        anchor_path, positive_path, negative_path = inputs[0], inputs[1], inputs[2]

        # Getting images
        anchor, positive, negative = _get_images(anchor_path, positive_path, negative_path)

        return anchor, positive, negative


    train_data = get_id_img(train_dir)
    train_triplet = get_triplet_dataset(train_data)
    train_ds = tf.data.Dataset.from_tensor_slices(train_triplet)
    train_ds = train_ds.map(_get_dataset, num_parallel_calls=AUTOTUNE)

    valid_data = get_id_img(valid_dir)
    valid_triplet = get_triplet_dataset(valid_data)
    valid_ds = tf.data.Dataset.from_tensor_slices(valid_triplet)
    valid_ds = valid_ds.map(_get_dataset, num_parallel_calls=AUTOTUNE)

    return train_ds, valid_ds