import os
import glob
import cv2
import keras
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt


def load_images_from_folder(folder_path, target_size=(256, 256)):
    """
    Loads images and their filenames from the specified folder.
    Returns a list of normalized grayscale images and their corresponding filenames.
    """
    images = []
    filenames = []
    for filepath in glob.glob(os.path.join(folder_path, "*")):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        images.append(img_norm)
        filenames.append(os.path.basename(filepath))  # Capture filename
    return images, filenames


# Paths
data_folder = "e:/MAZUR AI/termowizja zdjecia hot spotow nasze/"
output_folder = "e:/MAZUR AI/Wyniki/"
os.makedirs(output_folder, exist_ok=True)

# Load images and filenames
images, filenames = load_images_from_folder(data_folder, target_size=(256, 256))  # Now returns filenames
print(f"Loaded {len(images)} thermal images.")

# Prepare input data
X = np.array(images)[..., np.newaxis]


def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = keras.Model(input_img, decoded)
    return autoencoder


# Build and train the model
input_shape = X.shape[1:]
autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X, X, epochs=1000, batch_size=8, validation_split=0.1, verbose=1)


def detect_and_mark_hotspots(img, decoded_img, output_path, output_filename):
    """
    Saves the processed image with hotspots using the original filename.
    """
    orig_img = (img * 255).astype(np.uint8)
    recon_img = (decoded_img * 255).astype(np.uint8)
    error_map = cv2.absdiff(orig_img, recon_img)

    # Thresholding
    _, mask_otsu = cv2.threshold(error_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = np.mean(error_map) + 2 * np.std(error_map)
    mask_thresh = np.zeros_like(error_map)
    mask_thresh[error_map > thresh_val] = 255
    mask = mask_otsu  # Using Otsu's method

    # Find contours and draw rectangles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_color = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= 50:
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save with original filename
    cv2.imwrite(os.path.join(output_path, output_filename), img_color)
    return mask, img_color


# Process and save images with original filenames
decoded = autoencoder.predict(X)
for i in range(len(X)):
    # Use the original filename for output
    mask, result_img = detect_and_mark_hotspots(
        X[i].squeeze(),
        decoded[i].squeeze(),
        output_folder,
        output_filename=filenames[i]  # Pass the original filename
    )

# Display an example result
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title("Detected Hotspots (Red Boxes)")
plt.axis('off')
plt.show()