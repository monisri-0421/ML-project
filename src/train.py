import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def train():
    mlflow.tensorflow.autolog()
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory('./data/dataset', target_size=(224,224),
                                             batch_size=32, subset='training', class_mode='binary')
    val_data = datagen.flow_from_directory('./data/dataset', target_size=(224,224),
                                           batch_size=32, subset='validation', class_mode='binary')
    model.fit(train_data, epochs=3, validation_data=val_data)
    model.save('models/model.h5')

    a = 100
if __name__ == "__main__":
    train()
