# Générateur de Signaux Radar Simulés

Générateur de fichiers audio WAV simulant des signaux radar de navires militaires français. Conçu pour l'entraînement de modèles d'intelligence artificielle de détection et classification radar.

##  Description

Ce projet génère des signaux radar simulés au format WAV basés sur les caractéristiques réelles de systèmes radar navals :
- **4 classes de navires** : CDG (Charles de Gaulle), FLF, FREMM, FDA
- **14 systèmes radar** différents avec leurs signatures uniques
- **Paramètres aléatoires** : SNR, puissance, bruit, durée

##  Installation

```bash
pip install -r requirements.txt
```

**Dépendances** :
- `numpy` : Génération de signaux

##  Structure du Projet

```
detec_boat/
├── boats.json                    # Base de données des radars
├── generate_radar_wav.py         # Script de génération
├── requirements.txt              # Dépendances Python
└── iq_data/                      # Dossier de sortie (généré)
    ├── CDG/
    │   ├── smart_s_mK2/
    │   │   ├── smart_s_mK2_sample_000_snr12.5dB.wav
    │   │   ├── ...
    │   │   └── metadata.json
    │   ├── drbv_15C-EF/
    │   ├── SCANTER_6200_I/
    │   └── ARABEL-I/
    ├── FLF/
    ├── FREMM/
    └── FDA/
```

##  Utilisation

### Génération Automatique (Tous les Radars)

```bash
# Génère 20 fichiers WAV par radar (280 fichiers au total)
python generate_radar_wav.py

# Générer 50 fichiers par radar
python generate_radar_wav.py --num-files 50

# Changer le dossier de sortie
python generate_radar_wav.py --output-base-dir mes_donnees
```

### Lister les Radars Disponibles

```bash
python generate_radar_wav.py --list
```

### Options Disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--num-files`, `-n` | Nombre de fichiers par radar | 20 |
| `--duration-min` | Durée minimale (secondes) | 5.0 |
| `--duration-max` | Durée maximale (secondes) | 10.0 |
| `--sample-rate` | Taux d'échantillonnage (Hz) | 48000 |
| `--output-base-dir`, `-o` | Dossier de sortie | iq_data |
| `--list`, `-l` | Liste les radars disponibles | - |

##  Données Générées

### Format des Fichiers

- **Format** : WAV 16-bit mono
- **Taux d'échantillonnage** : 48 kHz (par défaut)
- **Durée** : 5-10 secondes (aléatoire)
- **Nom** : `{radar}_sample_{num}_snr{snr}dB.wav`

### Caractéristiques du Signal

Chaque radar possède sa **signature unique** définie par :
- **Fréquence centrale** : Bande de fréquence du radar (MHz)
- **Bande passante** : Largeur spectrale (MHz)
- **DI** (Durée d'Impulsion) : Durée du pulse (ms)
- **PRI** (Période de Répétition) : Intervalle entre pulses (ms)
- **PRF** : Fréquence de répétition des pulses (Hz)
- **Duty Cycle** : Rapport cyclique (%)

### Paramètres Aléatoires par Fichier

- **SNR** (Signal-to-Noise Ratio) : 5-30 dB
- **Puissance du signal** : 0.3-1.0
- **Niveau de bruit** : 0.01-0.3
- **Variation de fréquence** : ±5% autour du centre

### Fichier metadata.json

Chaque dossier radar contient un fichier `metadata.json` avec :
```json
{
  "radar_name": "smart_s_mK2",
  "boat": "CDG",
  "band": "EF",
  "center_freq_mhz": 2950.0,
  "bandwidth_mhz": 200,
  "DI_ms": 45,
  "PRI_ms": 450,
  "PRF_hz": 2.22,
  "duty_cycle": 10.0,
  "sample_rate_hz": 48000,
  "num_files": 20,
  "files": [
    {
      "filename": "smart_s_mK2_sample_000_snr12.5dB.wav",
      "duration_sec": 7.23,
      "signal_power": 0.78,
      "noise_power": 0.15,
      "snr_db": 12.5,
      "center_freq_mhz": 2947.3,
      "num_samples": 347040
    }
  ]
}
```

##  Utilisation pour l'IA

### Chargement des Données

```python
import numpy as np
import wave
import json
from pathlib import Path

def load_wav(filepath):
    """Charge un fichier WAV."""
    with wave.open(filepath, 'r') as wav:
        signal = np.frombuffer(wav.readframes(-1), dtype=np.int16)
        signal = signal.astype(np.float32) / 32768.0  # Normalisation
        return signal, wav.getframerate()

def load_dataset(base_dir='iq_data'):
    """Charge tous les signaux avec leurs labels."""
    X, y = [], []

    for boat_dir in Path(base_dir).iterdir():
        if not boat_dir.is_dir():
            continue

        boat_name = boat_dir.name

        for radar_dir in boat_dir.iterdir():
            if not radar_dir.is_dir():
                continue

            radar_name = radar_dir.name

            for wav_file in radar_dir.glob('*.wav'):
                signal, sr = load_wav(str(wav_file))
                X.append(signal)
                y.append(f"{boat_name}_{radar_name}")

    return X, y

# Utilisation
X, y = load_dataset()
print(f"Dataset : {len(X)} échantillons, {len(set(y))} classes")
```

### Extraction de Features

```python
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq

def extract_features(signal, sample_rate=48000):
    """Extrait des features du signal pour l'entraînement."""

    # 1. FFT - Analyse fréquentielle
    fft_vals = fft(signal)
    fft_freq = fftfreq(len(signal), 1/sample_rate)
    power_spectrum = np.abs(fft_vals)**2

    # 2. Spectrogramme
    f, t, Sxx = sp_signal.spectrogram(signal, fs=sample_rate)

    # 3. Features statistiques
    features = {
        'mean_power': np.mean(power_spectrum),
        'max_power': np.max(power_spectrum),
        'spectral_centroid': np.sum(fft_freq[:len(fft_freq)//2] * power_spectrum[:len(power_spectrum)//2]) / np.sum(power_spectrum[:len(power_spectrum)//2]),
        'bandwidth': np.std(power_spectrum),
        'snr_estimate': 10 * np.log10(np.max(power_spectrum) / np.mean(power_spectrum))
    }

    return features, Sxx
```

### Exemple d'Entraînement CNN

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape):
    """Crée un CNN pour classification radar."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(256, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(14, activation='softmax')  # 14 radars
    ])

    return model
```

## 📡 Radars Disponibles

### CDG (Charles de Gaulle)
- **smart_s_mK2** - Bande EF (2850-3050 MHz) - DI: 45ms, PRI: 450ms
- **drbv_15C-EF** - Bande EF (3000-3100 MHz) - DI: 35ms, PRI: 380ms
- **SCANTER_6200_I** - Bande I (9300-9400 MHz) - DI: 15ms, PRI: 220ms
- **ARABEL-I** - Bande I (9200-9600 MHz) - DI: 28ms, PRI: 420ms

### FLF (Frégate Légère Furtive)
- **DRBV_15_C** - Bande C (3000-3100 MHz) - DI: 32ms, PRI: 410ms
- **DRBN_34_I** - Bande I (9300-9400 MHz) - DI: 18ms, PRI: 290ms
- **KUMC_J** - Bande J (14000-17000 MHz) - DI: 8ms, PRI: 150ms

### FREMM
- **HERAKLES_EF** - Bande EF (2800-2900 MHz) - DI: 55ms, PRI: 480ms
- **SCANTER_6002_I** - Bande I (9300-9400 MHz) - DI: 12ms, PRI: 260ms
- **SCANTER_2001_I** - Bande I (9300-9400 MHz) - DI: 16ms, PRI: 280ms

### FDA (Frégate de Défense Aérienne)
- **LRR_CD** - Bande CD (1300-1400 MHz) - DI: 65ms, PRI: 500ms
- **EMPAR_GH** - Bande GH (5500-5700 MHz) - DI: 25ms, PRI: 340ms
- **SCANTER_6002_I** - Bande EF (9300-9400 MHz) - DI: 14ms, PRI: 240ms
- **ORION_RTN_2SX_J** - Bande J (12000-13000 MHz) - DI: 6ms, PRI: 160ms

## 🔧 Personnalisation

### Modifier boats.json

```json
{
  "NOUVEAU_BATEAU": {
    "radar_systems": [
      {
        "name": "mon_radar",
        "description": "Description du radar",
        "frequency_ranges": [
          {
            "band": "X",
            "min_mhz": 8000,
            "max_mhz": 12000
          }
        ],
        "DI_ms": 30,
        "PRI_ms": 250
      }
    ]
  }
}
```

### Ajouter de Nouveaux Radars

1. Éditez `boats.json`
2. Ajoutez votre radar avec ses caractéristiques
3. Relancez la génération

## Avertissement

Ce code génère des **signaux simulés à des fins éducatives**. Les caractéristiques sont basées sur des données publiques et ne représentent pas nécessairement avec exactitude les signaux réels des systèmes radar militaires.

**Usage autorisé** : Recherche, éducation, entraînement d'IA
**Usage interdit** : Applications militaires offensives, espionnage

## Licence

Usage éducatif uniquement.

## Contribution

Les contributions sont les bienvenues pour :
- Ajouter de nouveaux radars
- Améliorer la simulation de signal
- Optimiser les performances
- Ajouter des exemples d'utilisation

## Contact

Pour toute question sur l'utilisation éducative de ce projet.
