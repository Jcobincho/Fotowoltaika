import os
import glob
import cv2
import numpy as np
import sys

# Sprawdzenie i instalacja TensorFlow jeśli potrzeba
try:
    import tensorflow as tf

    print(f"TensorFlow version: {tf.__version__}")

    try:
        from tensorflow import keras

        print(f"Keras version: {keras.__version__}")
    except ImportError:
        print("Keras nie jest dostępny w tej instalacji TensorFlow")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tensorflow"])
        import tensorflow as tf
        from tensorflow import keras

except ImportError:
    print("TensorFlow nie jest zainstalowany. Instaluję...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
    from tensorflow import keras

try:
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                                         BatchNormalization, Dropout, Concatenate,
                                         Activation, Add)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("Import przez tensorflow.keras - OK")

except ImportError:
    try:
        from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                                  BatchNormalization, Dropout, Concatenate,
                                  Activation, Add)
        from keras.models import Model
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        print("Import przez keras - OK")
    except ImportError:
        print("Błąd: Nie można zaimportować keras. Sprawdź instalację TensorFlow.")
        sys.exit(1)

import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Instaluję scikit-learn...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.model_selection import train_test_split


def get_project_paths():
    """Automatyczne wykrywanie ścieżek projektu"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    while os.path.basename(project_root) != "Fotowoltaika" and project_root != os.path.dirname(project_root):
        project_root = os.path.dirname(project_root)

    if os.path.basename(project_root) != "Fotowoltaika":
        project_root = script_dir

    data_folder = os.path.join(project_root, "data", "thermal_images")
    output_folder = os.path.join(project_root, "results")
    model_save_path = os.path.join(project_root, "models")

    print(f"Katalog projektu: {project_root}")
    print(f"Katalog danych: {data_folder}")
    print(f"Katalog wyników: {output_folder}")
    print(f"Katalog modeli: {model_save_path}")

    return data_folder, output_folder, model_save_path


def load_and_preprocess_images(folder_path, target_size=(256, 256)):
    """
    Ładuje obrazy termowizyjne z lepszym przetwarzaniem
    """
    images = []
    filenames = []

    if not os.path.exists(folder_path):
        print(f"UWAGA: Folder {folder_path} nie istnieje!")
        os.makedirs(folder_path, exist_ok=True)
        print(f"Utworzono folder: {folder_path}")
        return images, filenames

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']

    for ext in extensions:
        for filepath in glob.glob(os.path.join(folder_path, ext)):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Zmiana rozmiaru
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

            # Normalizacja do [0,1]
            img_norm = img_resized.astype(np.float32) / 255.0

            images.append(img_norm)
            filenames.append(os.path.basename(filepath))

    return images, filenames


def create_synthetic_normal_images(images, num_synthetic=None):
    """
    Tworzy syntetyczne "normalne" obrazy poprzez usunięcie hotspotów
    z oryginalnych obrazów termowizyjnych
    """
    if num_synthetic is None:
        num_synthetic = len(images)

    synthetic_images = []

    for i, img in enumerate(images[:num_synthetic]):
        # Konwersja do uint8 dla operacji OpenCV
        img_uint8 = (img * 255).astype(np.uint8)

        # Detekcja bardzo jasnych obszarów (potencjalne hotspoty)
        # Użyj percentyla zamiast stałej wartości
        threshold_value = np.percentile(img_uint8, 85)  # 85% percentyl
        _, hotspot_mask = cv2.threshold(img_uint8, threshold_value, 255, cv2.THRESH_BINARY)

        # Rozszerzenie maski hotspotów
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hotspot_mask_dilated = cv2.dilate(hotspot_mask, kernel, iterations=2)

        # Inwersja maski (obszary do zachowania)
        normal_mask = cv2.bitwise_not(hotspot_mask_dilated)

        # Inpainting - wypełnienie obszarów hotspotów
        img_inpainted = cv2.inpaint(img_uint8, hotspot_mask_dilated, 3, cv2.INPAINT_TELEA)

        # Dodatkowe wygładzenie dla naturalnego wyglądu
        img_smoothed = cv2.GaussianBlur(img_inpainted, (5, 5), 0)

        # Mieszanie oryginalnego obrazu (bez hotspotów) z wygładzonym
        img_mixed = cv2.bitwise_and(img_uint8, img_uint8, mask=normal_mask) + \
                    cv2.bitwise_and(img_smoothed, img_smoothed, mask=hotspot_mask_dilated)

        # Konwersja z powrotem do float32 i normalizacja
        synthetic_img = img_mixed.astype(np.float32) / 255.0
        synthetic_images.append(synthetic_img)

    return synthetic_images


def build_lightweight_autoencoder(input_shape):
    """
    Lżejsza architektura autoenkodera lepiej dostosowana do zadania
    """
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D(2)(x)

    # Decoder
    x = Conv2D(128, 3, padding='same', activation='relu')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D(2)(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(2)(x)

    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(2)(x)

    decoded = Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    model = Model(inputs, decoded)
    return model


def thermal_hotspot_detection(original_img, reconstructed_img, sensitivity=0.3):
    """
    Specjalizowana detekcja hotspotów dla obrazów termowizyjnych
    """
    # Konwersja do uint8
    orig_uint8 = (original_img * 255).astype(np.uint8)
    recon_uint8 = (reconstructed_img * 255).astype(np.uint8)

    # Mapa różnic
    diff_map = cv2.absdiff(orig_uint8, recon_uint8)

    # Wygładzenie
    diff_smooth = cv2.GaussianBlur(diff_map, (5, 5), 0)

    # Adaptacyjne thresholding z uwzględnieniem lokalnych warunków
    mean_diff = np.mean(diff_smooth)
    std_diff = np.std(diff_smooth)
    threshold = mean_diff + sensitivity * std_diff

    # Dodatkowy warunek - minimum brightness w oryginalnym obrazie
    brightness_threshold = np.percentile(orig_uint8, 70)  # 70% percentyl jasności
    brightness_mask = orig_uint8 > brightness_threshold

    # Kombinacja warunków: różnica I jasność
    _, diff_mask = cv2.threshold(diff_smooth, threshold, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_and(diff_mask, brightness_mask.astype(np.uint8) * 255)

    # Operacje morfologiczne
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    return cleaned_mask, diff_smooth


def analyze_and_mark_hotspots(original_img, mask, filename, output_path):
    """
    Analizuje i zaznacza hotspoty na obrazie
    """
    # Konwersja do kolorowego obrazu
    img_color = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Znajdowanie konturów
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspot_count = 0
    hotspot_info = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filtrowanie na podstawie wielkości
        if 50 < area < 5000:  # Dostosuj te wartości do swoich potrzeb
            # Obliczanie boundingRect
            x, y, w, h = cv2.boundingRect(contour)

            # Sprawdzenie proporcji (unikamy bardzo wydłużonych kształtów)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 3.0:
                hotspot_count += 1

                # Zaznaczanie hotspotu
                cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img_color, f'H{hotspot_count}', (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Dodatkowe informacje o hotspocie
                center_x, center_y = x + w // 2, y + h // 2
                hotspot_info.append({
                    'id': hotspot_count,
                    'center': (center_x, center_y),
                    'area': area,
                    'bbox': (x, y, w, h)
                })

    # Zapisanie oznaczonego obrazu
    output_file = os.path.join(output_path, f"detected_{filename}")
    cv2.imwrite(output_file, img_color)

    return img_color, hotspot_count, hotspot_info


def main():
    # Pobranie ścieżek
    data_folder, output_folder, model_save_path = get_project_paths()

    # Tworzenie katalogów
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    print("Ładowanie obrazów...")
    images, filenames = load_and_preprocess_images(data_folder, target_size=(256, 256))

    if len(images) == 0:
        print("Nie znaleziono obrazów!")
        return

    print(f"Załadowano {len(images)} obrazów.")

    # KLUCZOWA ZMIANA: Tworzenie syntetycznych "normalnych" obrazów
    print("Tworzenie syntetycznych normalnych obrazów (bez hotspotów)...")
    normal_images = create_synthetic_normal_images(images)

    # Przygotowanie danych treningowych
    X_normal = np.array(normal_images)[..., np.newaxis]  # Normalne obrazy jako target
    X_original = np.array(images)[..., np.newaxis]  # Oryginalne obrazy jako input

    # Podział danych
    if len(X_normal) > 2:
        X_train_normal, X_val_normal, X_train_orig, X_val_orig = train_test_split(
            X_normal, X_original, test_size=0.2, random_state=42
        )
    else:
        X_train_normal = X_val_normal = X_normal
        X_train_orig = X_val_orig = X_original

    print(f"Dane treningowe: {len(X_train_normal)}")

    # Budowa modelu
    print("Budowa modelu...")
    autoencoder = build_lightweight_autoencoder(X_original.shape[1:])

    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("Architektura modelu:")
    autoencoder.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]

    # KLUCZOWE: Trening na oryginalnych obrazach -> normalne obrazy
    print("Rozpoczęcie treningu...")
    print("Model uczy się rekonstruować obrazy BEZ hotspotów z obrazów Z hotspotami")

    history = autoencoder.fit(
        X_train_orig, X_train_normal,  # Input: z hotspotami, Target: bez hotspotów
        validation_data=(X_val_orig, X_val_normal),
        epochs=150,
        batch_size=4,
        callbacks=callbacks,
        verbose=1
    )

    # Zapisanie modelu
    model_path = os.path.join(model_save_path, "thermal_hotspot_detector.h5")
    autoencoder.save(model_path)
    print(f"Model zapisany: {model_path}")

    # Detekcja hotspotów
    print("\nRozpoczęcie detekcji hotspotów...")
    reconstructed = autoencoder.predict(X_original, batch_size=1)

    total_hotspots = 0
    results_summary = []

    for i in range(len(X_original)):
        print(f"Analizowanie obrazu {i + 1}/{len(X_original)}: {filenames[i]}")

        # Detekcja hotspotów
        hotspot_mask, diff_map = thermal_hotspot_detection(
            X_original[i].squeeze(),
            reconstructed[i].squeeze(),
            sensitivity=0.4  # Dostosuj wrażliwość
        )

        # Analiza i oznaczanie
        marked_img, hotspot_count, hotspot_info = analyze_and_mark_hotspots(
            X_original[i].squeeze(),
            hotspot_mask,
            filenames[i],
            output_folder
        )

        total_hotspots += hotspot_count
        results_summary.append((filenames[i], hotspot_count, hotspot_info))

        print(f"  -> Wykryto {hotspot_count} hotspotów")

        # Zapisanie mapy różnic
        diff_path = os.path.join(output_folder, f"diff_map_{filenames[i]}")
        cv2.imwrite(diff_path, diff_map)

    # Podsumowanie
    print(f"\n=== WYNIKI ANALIZY ===")
    print(f"Przeanalizowano obrazów: {len(X_original)}")
    print(f"Łączna liczba hotspotów: {total_hotspots}")
    if len(X_original) > 0:
        print(f"Średnia hotspotów na obraz: {total_hotspots / len(X_original):.1f}")

    # Szczegółowy raport
    report_path = os.path.join(output_folder, "hotspot_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPORT ANALIZY HOTSPOTÓW - PANELE FOTOWOLTAICZNE\n")
        f.write("=" * 60 + "\n\n")

        for filename, count, hotspots in results_summary:
            f.write(f"Plik: {filename}\n")
            f.write(f"Liczba hotspotów: {count}\n")

            for hotspot in hotspots:
                f.write(f"  - Hotspot {hotspot['id']}: centrum({hotspot['center'][0]}, "
                        f"{hotspot['center'][1]}), powierzchnia: {hotspot['area']:.0f}px\n")
            f.write("\n")

        f.write(f"PODSUMOWANIE:\n")
        f.write(f"Całkowita liczba hotspotów: {total_hotspots}\n")
        if len(X_original) > 0:
            f.write(f"Średnia na obraz: {total_hotspots / len(X_original):.2f}\n")

    print(f"\nWyniki zapisane w: {output_folder}")
    print(f"Szczegółowy raport: {report_path}")
    print("\nAnaliza zakończona!")


if __name__ == "__main__":
    main()