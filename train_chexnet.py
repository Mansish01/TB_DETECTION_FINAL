import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_chexnet_model(input_shape, num_classes):
    # Load CheXNet model architecture
    base_model = tf.keras.applications.DenseNet121(weights=None, include_top=False, input_shape=input_shape)

    # Add custom classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(model, train_data_dir, input_shape, num_classes, batch_size=32, num_epochs=10):
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Set up data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Set up data generator
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Train the model
    model.fit(train_generator, epochs=num_epochs)

    return model

if __name__ == "__main__":
    input_shape = (224, 224, 3)
    num_classes = 2  # TB and non-TB

    model = create_chexnet_model(input_shape, num_classes)

    train_data_dir = 'path_to_train_data_directory'
    trained_model = train_model(model, train_data_dir, input_shape, num_classes)

    # Save the trained model
    trained_model.save('tb_classification_model.h5')
