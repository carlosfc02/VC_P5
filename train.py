import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# --- CONFIGURACIÓN ---
img_height, img_width = 48, 48  # Tamaño estándar de FER-2013
batch_size = 64
epochs = 15  # Con 15 o 20 suele bastar para aprobar
num_classes = 7 # angry, disgust, fear, happy, neutral, sad, surprise

# Rutas (AJUSTA ESTO si tu carpeta se llama diferente)
train_dir = 'images/train'
val_dir = 'images/validation'

# --- 1. GENERADORES DE DATOS (Data Augmentation) ---
# Esto crea variaciones de las fotos para que el modelo aprenda mejor
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale", # Importante: FER-2013 es en blanco y negro
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

# --- 2. DEFINIR EL MODELO (CNN) ---
model = Sequential([
    # Bloque 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Bloque 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Bloque 3
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Capas Densas (Clasificación)
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') # Salida: 7 emociones
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. ENTRENAR ---
print("🚀 Iniciando entrenamiento... ve a por un café ☕")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# --- 4. GUARDAR ---
model.save('modelo_emociones.h5')
print("✅ Modelo guardado como 'modelo_emociones.h5'")