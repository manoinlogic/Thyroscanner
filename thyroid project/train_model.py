import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define dataset paths
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

# Data augmentation (balanced for accuracy and speed)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# Get the number of classes dynamically
num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} classes: {train_generator.class_indices}")

# Load MobileNetV2 (pretrained on ImageNet) & freeze base layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layers to train only the classifier

# Build custom classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)  # Prevent overfitting
output = Dense(num_classes, activation='softmax')(x)  # ✅ FIXED: Pass x explicitly

# Create final model
model = Model(inputs=base_model.input, outputs=output)

# Compile model (using Adam optimizer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train the model
model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=callbacks)

# Save the model
model.save('model/optimized_model.h5')
