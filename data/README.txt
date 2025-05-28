
# INSTRUKCJA UŻYCIA

## Struktura folderów:
- data/thermal_images/ - umieść tutaj swoje obrazy termowizyjne
- results/ - tutaj będą zapisane wyniki analizy
- models/ - tutaj będą zapisane wytrenowane modele

## Obsługiwane formaty obrazów:
- JPG, JPEG
- PNG
- TIFF, TIF
- BMP

## Jak używać:
1. Skopiuj swoje obrazy termowizyjne do folderu 'data/thermal_images/'
2. Uruchom program ponownie
3. Program automatycznie przetworzy wszystkie obrazy i zapisze wyniki

## Wyniki:
- hotspots_[nazwa_pliku] - obraz z zaznaczonymi hotspotami
- error_map_[nazwa_pliku] - mapa błędów pokazująca różnice
- detection_report.txt - raport z liczbą wykrytych hotspotów
