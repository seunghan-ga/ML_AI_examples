import tensorflow as tf


def get_model(input_shape=(32, 32, 3), rate=0.3, weight_path=None, include_top=True):
    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=1)(img_input)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(rate)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=1)(x)
    x = tf.keras.layers.Dropout(rate)(x)

    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(rate)(x)
        x = tf.keras.layers.Dense(6)(x)
        x = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.models.Model(inputs=img_input, outputs=x)

    if weight_path is not None:
        model.load_weights(weight_path)

    return model


def checkpoints(path, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True):
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                       monitor=monitor,
                                                       verbose=verbose,
                                                       save_best_only=save_best_only,
                                                       save_weights_only=save_weights_only)

    return cb_checkpoint


def earlystopping(monitor='val_loss', mode='min', verbose=1, patience=30):
    cb_earlystopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                        mode=mode,
                                                        verbose=verbose,
                                                        patience=patience)

    return cb_earlystopping


def train(model, X_train, y_train, X_valid, y_valid, epochs=200, batch_size=64, verbose=1,
          checkpoints=False, cb_checkpoints=None, earlystopping=False, cb_earlystopping=None):
    callbacks = None
    if checkpoints is True or earlystopping is True:
        callbacks = list()
    if checkpoints is True:
        callbacks.append(cb_checkpoints)
    if cb_earlystopping is True:
        callbacks.append(cb_earlystopping)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X=X_train,
                        y=y_train,
                        epochs=epochs,
                        validation_data=(X_valid, y_valid),
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=verbose).history

    return model, history


def data_generator(path, target_size=(32, 32), batch_size=64, class_mode='categorical',
                   scale=(1 / 255.), shuffle=True):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=scale)
    generator_from_directory = generator.flow_from_directory(path,
                                                             target_size=target_size,
                                                             batch_size=batch_size,
                                                             class_mode=class_mode,
                                                             shuffle=shuffle)

    return generator_from_directory


if __name__ == "__main__":
    pass
