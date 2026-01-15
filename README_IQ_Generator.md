# Générateur de Données IQ Radar - Usage Éducatif

Ce script génère des données IQ (In-phase/Quadrature) simulées pour différents systèmes radar navals, basé sur les caractéristiques de fréquence de vrais radars. Utile pour l'entraînement de modèles d'IA de classification radar.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Génération du dataset complet

```bash
python generate_iq_radar_data.py
```

Cela générera des données IQ pour tous les bateaux définis dans `boats.json` :
- **CDG** (Charles de Gaulle) : 4 systèmes radar
- **FLF** (Frégate Légère Furtive) : 3 systèmes radar
- **FREMM** : 3 systèmes radar
- **FDA** (Frégate de Défense Aérienne) : 4 systèmes radar

### Structure des fichiers générés

```
iq_data/
├── CDG/
│   ├── smart_s_mK2_EF_2950MHz_sample000.iq
│   ├── smart_s_mK2_EF_2950MHz_sample000.csv
│   ├── smart_s_mK2_EF_2950MHz_sample001.iq
│   ├── ...
│   └── metadata.json
├── FLF/
│   ├── ...
├── FREMM/
│   ├── ...
└── FDA/
    ├── ...
```

## Format des données IQ

### Fichiers .iq (binaire)
- Format : **float32** entrelacé (I, Q, I, Q, ...)
- Taux d'échantillonnage : 10 MHz par défaut
- Normalisation : valeurs entre -1.0 et 1.0

### Fichiers .csv (texte, échantillon uniquement)
Colonnes :
- `sample_index` : Index de l'échantillon
- `I` : Composante In-phase
- `Q` : Composante Quadrature
- `magnitude` : Magnitude du signal √(I² + Q²)
- `phase` : Phase du signal arctan2(Q, I)

### metadata.json
Contient les informations sur chaque fichier généré :
- Nom du radar
- Bande de fréquence
- Fréquence centrale
- Largeur de bande
- Nombre d'échantillons

## Caractéristiques de la simulation

### Signal radar simulé
- **Type de modulation** : LFM (Linear Frequency Modulation / Chirp)
- **Enveloppe** : Gaussienne pour chaque pulse
- **PRF** : 1000 Hz (Pulse Repetition Frequency)
- **Durée de pulse** : 1 ms
- **Nombre de pulses** : 10 par échantillon

### Bruit
- Bruit gaussien ajouté (niveau configurable 0.05-0.15)
- Bruit de fond continu entre les pulses
- Simule les conditions réelles de capture radar

## Utilisation pour l'IA

### Chargement des données en Python

```python
import numpy as np
import json

# Charger les métadonnées
with open('iq_data/CDG/metadata.json', 'r') as f:
    metadata = json.load(f)

# Charger un fichier IQ
iq_data = np.fromfile('iq_data/CDG/smart_s_mK2_EF_2950MHz_sample000.iq', dtype=np.float32)

# Séparer I et Q
I = iq_data[0::2]
Q = iq_data[1::2]

# Signal complexe
signal = I + 1j * Q
```

### Exemple de dataset pour entraînement

```python
import numpy as np
from pathlib import Path
import json

def load_iq_dataset(data_dir='iq_data'):
    """Charge tous les fichiers IQ avec leurs labels."""
    X = []  # Signaux
    y = []  # Labels (nom du bateau)

    for boat_dir in Path(data_dir).iterdir():
        if boat_dir.is_dir():
            boat_name = boat_dir.name

            # Charger tous les fichiers .iq
            for iq_file in boat_dir.glob('*.iq'):
                iq_data = np.fromfile(iq_file, dtype=np.float32)
                I = iq_data[0::2]
                Q = iq_data[1::2]
                signal = I + 1j * Q

                X.append(signal)
                y.append(boat_name)

    return np.array(X, dtype=object), np.array(y)

# Utilisation
X, y = load_iq_dataset()
print(f"Dataset: {len(X)} échantillons")
print(f"Classes: {np.unique(y)}")
```

### Exemple de traitement signal pour features

```python
import numpy as np
from scipy import signal

def extract_features(iq_signal, sample_rate=10e6):
    """Extrait des features du signal IQ pour l'IA."""

    # 1. Spectre de puissance (FFT)
    fft = np.fft.fft(iq_signal)
    power_spectrum = np.abs(fft)**2

    # 2. Spectrogramme
    f, t, Sxx = signal.spectrogram(iq_signal, fs=sample_rate)

    # 3. Features statistiques
    features = {
        'mean_power': np.mean(power_spectrum),
        'peak_power': np.max(power_spectrum),
        'bandwidth': estimate_bandwidth(power_spectrum),
        'spectral_centroid': spectral_centroid(power_spectrum),
        # ... autres features
    }

    return features

def spectral_centroid(spectrum):
    """Calcule le centroïde spectral."""
    freqs = np.fft.fftfreq(len(spectrum))
    return np.sum(freqs * spectrum) / np.sum(spectrum)

def estimate_bandwidth(spectrum, threshold=0.5):
    """Estime la largeur de bande à -3dB."""
    max_power = np.max(spectrum)
    above_threshold = spectrum > (max_power * threshold)
    return np.sum(above_threshold)
```

## Personnalisation

### Modifier les paramètres de simulation

```python
# Dans le script generate_iq_radar_data.py
simulator = RadarIQSimulator(sample_rate=20e6)  # Taux d'échantillonnage 20 MHz

simulator.generate_dataset_for_boat(
    boat_name=boat_name,
    radar_systems=boat_info['radar_systems'],
    output_dir="iq_data",
    num_samples=10  # Générer 10 échantillons par radar
)
```

### Modifier les caractéristiques des pulses

```python
I, Q = self.generate_multi_pulse_train(
    center_freq_mhz=center_freq,
    bandwidth_mhz=bandwidth,
    pulse_duration_ms=2.0,  # Pulse plus long
    pulse_repetition_freq_hz=500,  # PRF plus basse
    num_pulses=20,  # Plus de pulses
    noise_level=0.2  # Plus de bruit
)
```

## Avertissement

Ce code génère des **signaux simulés à des fins éducatives**. Les caractéristiques sont basées sur des données publiques et ne représentent pas forcément avec exactitude les signaux réels des systèmes radar militaires.

## Objectifs pédagogiques

- Comprendre la structure des signaux IQ
- Apprentissage du traitement du signal radar
- Entraînement de modèles de classification
- Analyse spectrale et extraction de features
- Développement d'algorithmes de détection
