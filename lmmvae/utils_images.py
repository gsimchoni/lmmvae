import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def divide(a, b):
    if a % b == 0:
        return a // b
    else:
        return a // b + 1

def factors(n):    # (cf. https://stackoverflow.com/a/15703327/849891)
    j = 2
    while n > 1:
        for i in range(j, int(np.sqrt(n+0.05)) + 1):
            if n % i == 0:
                n /= i ; j = i
                yield i
                break
        else:
            if n > 1:
                yield n; break

def get_batchsize_steps(n):
    factors_n = list(factors(n))
    if len(factors_n) > 1:
        batch_size = factors_n[-2]
    else:
        batch_size = 1
    steps = n // batch_size
    return batch_size, steps

def get_generators(images_df, images_dir, train_samp_subj, valid_samp_subj, test_samp_subj,
                   batch_size, img_file_col, RE_col, filter_col, img_height, img_width):
    train_datagen = ImageDataGenerator(rescale = 1./255)
    valid_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    train_generator = train_datagen.flow_from_dataframe(
        images_df[images_df[filter_col].isin(train_samp_subj)],
        directory = images_dir,
        x_col = img_file_col,
        y_col = RE_col,
        target_size = (img_height, img_width),
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = True,
        validate_filenames = False
    )
    valid_generator = valid_datagen.flow_from_dataframe(
        images_df[images_df[filter_col].isin(valid_samp_subj)],
        directory = images_dir,
        x_col = img_file_col,
        y_col = RE_col,
        target_size = (img_height, img_width),
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = False,
        validate_filenames = False
    )
    test_generator = test_datagen.flow_from_dataframe(
        images_df[images_df[filter_col].isin(test_samp_subj)],
        directory = images_dir,
        x_col = img_file_col,
        y_col = RE_col,
        target_size = (img_height, img_width),
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = False,
        validate_filenames = False
    )
    return train_generator, valid_generator, test_generator

def sample_split(seed, train_index_subj, valid_frac = 0.1):
    np.random.seed(seed)
    train_n = len(train_index_subj)
    valid_samp = np.random.choice(train_n, int(valid_frac * train_n), replace=False)
    valid_index_subj = train_index_subj[valid_samp]
    train_index_subj = np.delete(train_index_subj, valid_samp)
    return train_index_subj, valid_index_subj

def custom_generator_fit(generator, epochs, with_RE):
    count = 0
    while True:
        if count == generator.n * epochs:
            generator.reset()
            break
        count += generator.batch_size
        data = generator.next()
        imgs = data[0]
        if with_RE:
            Z = data[1]
            yield (imgs, Z), (imgs, Z)
        else:
            yield imgs, imgs

def custom_generator_predict(generator, epochs, with_RE):
    count = 0
    while True:
        if count == generator.n * epochs:
            generator.reset()
            break
        count += generator.batch_size
        data = generator.next()
        imgs = data[0]
        if with_RE:
            Z = data[1]
            yield (imgs, Z), None
        else:
            yield imgs, None

def get_full_RE_cols_from_generator(generator):
    generator.reset()
    prev_shuffle_state = generator.shuffle
    generator.shuffle = False
    steps = divide(generator.n, generator.batch_size)
    Z_list = []
    for i in range(steps):
        _, Z_i = generator.next()
        Z_list.append(Z_i)
    Z = np.concatenate(Z_list, axis=0)
    generator.shuffle = prev_shuffle_state
    return Z

