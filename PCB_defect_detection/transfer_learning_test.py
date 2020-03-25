from PCB_defect_detection import defect_detection
import tensorflow as tf


if __name__ == "__main__":
    """pre trained models -> save only weight"""
    # loaded_model = tf.keras.models.load_model("./models/pcb_0.0313.hdf5")
    # model = tf.keras.models.Sequential()
    # model.add(loaded_model)
    # model.summary()
    # model.save_weight("./models/pre_treined_weight.h5")

    """all layer weight -> save only convolution layer weight"""
    # pre_trained = defect_detection.get_model(include_top=True, input_shape=(32, 32, 3))
    # pre_trained.load_weights("./models/pre_trained_weight.h5")
    # pre_trained.summary()
    # pre_trained.layers.pop()
    # pre_trained = tf.keras.models.Model(inputs=pre_trained.inputs, outputs=pre_trained.layers[-7].output)
    # pre_trained.save_weights("pre_trained_weight_notop.h5")

    """create transfer learning models"""
    pre_trained = defect_detection.get_model(include_top=False, input_shape=(32, 32, 3))
    pre_trained.load_weights("./models/pre_trained_weight_notop.h5")
    model = tf.keras.models.Sequential()
    model.add(pre_trained)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    """pre processing training data"""
    train_dir = "./train"
    validation_dir = "./test"
    target_size = (32, 32)
    batch_size = 64

    train_generator = defect_detection.data_generator(train_dir, target_size, batch_size)
    validation_generator = defect_detection.data_generator(validation_dir, target_size, batch_size)

    class_num = len(train_generator.class_indices)
    custom_labels = list(validation_generator.class_indices.keys())
    custom_values = list(validation_generator.class_indices.values())
    print(custom_labels)

    """training"""
    epochs = 300
    paths = './models/checkpoints/pcb_demo_{epoch:02d}_{val_loss:.4f}.h5'

    cb_checkpoint = defect_detection.checkpoints(paths)
    cb_earlystopping = defect_detection.earlystopping()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.samples / train_generator.batch_size,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.samples / validation_generator.batch_size,
                                  callbacks=[cb_checkpoint, cb_earlystopping],
                                  verbose=1)
