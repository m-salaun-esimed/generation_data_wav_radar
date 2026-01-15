# Générateur de Signaux Radar WAV - Usage Éducatif

Ce script génère des signaux radar simulés au format WAV pour différents systèmes radar navals, basé sur les caractéristiques de fréquence de vrais radars. Utile pour l'entraînement de modèles d'IA de classification radar.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Génération automatique de tous les radars

```bash
python generate_radar_wav.py
```

Cela générera automatiquement des fichiers WAV pour tous les bateaux et radars définis dans `boats.json` :
- **CDG** (Charles de Gaulle) : 4 systèmes radar
- **FLF** (Frégate Légère Furtive) : 3 systèmes radar
- **FREMM** : 3 systèmes radar
- **FDA** (Frégate de Défense Aérienne) : 4 systèmes radar

Par défaut, 20 fichiers WAV sont générés par radar (280 fichiers au total).

### Options disponibles

```bash
# Générer 50 fichiers par radar au lieu de 20
python generate_radar_wav.py --num-files 50

# Changer le dossier de sortie
python generate_radar_wav.py --output-base-dir mes_donnees

# Modifier la durée des signaux (min et max en secondes)
python generate_radar_wav.py --duration-min 3.0 --duration-max 8.0

# Lister tous les radars disponibles
python generate_radar_wav.py --list
```

### Structure des fichiers générés

```
iq_data/
├── CDG/
│   ├── smart_s_mK2/
│   │   ├── smart_s_mK2_sample_000_snr12.5dB.wav
│   │   ├── smart_s_mK2_sample_001_snr18.3dB.wav
│   │   ├── ...
│   │   └── metadata.json
│   ├── drbv_15C-EF/
│   ├── SCANTER_6200_I/
│   └── ARABEL-I/
├── FLF/
│   ├── DRBV_15_C/
│   ├── DRBN_34_I/
│   └── KUMC_J/
├── FREMM/
│   ├── HERAKLES_EF/
│   ├── SCANTER_6002_I/
│   └── SCANTER_2001_I/
└── FDA/
    ├── LRR_CD/
    ├── EMPAR_GH/
    ├── SCANTER_6002_I/
    └── ORION_RTN_2SX_J/
```

## Format des données

### Fichiers .wav
- Format : **WAV 16-bit mono**
- Taux d'échantillonnage : 48 kHz par défaut
- Durée : 5-10 secondes (aléatoire par fichier)
- Normalisation : valeurs entre -1.0 et 1.0

### metadata.json
Contient les informations sur chaque fichier généré :
- Nom du radar et bateau
- Bande de fréquence
- Fréquence centrale
- Largeur de bande
- DI (Durée d'Impulsion) en ms
- PRI (Période de Répétition) en ms
- PRF (Fréquence de Répétition) en Hz
- Duty Cycle (%)
- Paramètres par fichier : SNR, puissance, bruit, durée

## Caractéristiques de la simulation

### Signal radar simulé
- **Type de modulation** : LFM (Linear Frequency Modulation / Chirp)
- **Mode** : Pulsé avec PRF et duty cycle réalistes
- **DI** : Durée d'Impulsion (fixe par radar, signature unique)
- **PRI** : Période de Répétition des Impulsions (fixe par radar, 100-500 ms)
- **PRF** : Calculée automatiquement (1000 / PRI_ms Hz)
- **Duty Cycle** : Calculé automatiquement (DI_ms / PRI_ms)

### Paramètres aléatoires par fichier
- **SNR** : Signal-to-Noise Ratio (5-30 dB)
- **Puissance du signal** : 0.3-1.0
- **Niveau de bruit** : 0.01-0.3
- **Durée** : 5-10 secondes
- **Variation de fréquence** : ±5% autour du centre

### Bruit
- Bruit gaussien ajouté (niveau aléatoire par fichier)
- Bruit de fond continu entre les pulses
- Simule les conditions réelles de capture radar

## Utilisation pour l'IA

### Chargement des données en Python

```python
import numpy as np
import wave
import json

# Charger les métadonnées d'un radar
with open('iq_data/CDG/smart_s_mK2/metadata.json', 'r') as f:
    metadata = json.load(f)

# Charger un fichier WAV
def load_wav(filepath):
    """Charge un fichier WAV."""
    with wave.open(filepath, 'r') as wav:
        signal = np.frombuffer(wav.readframes(-1), dtype=np.int16)
        signal = signal.astype(np.float32) / 32768.0  # Normalisation
        return signal, wav.getframerate()

signal, sample_rate = load_wav('iq_data/CDG/smart_s_mK2/smart_s_mK2_sample_000_snr12.5dB.wav')
```

### Exemple de dataset pour entraînement

```python
import numpy as np
from pathlib import Path
import wave

def load_dataset(data_dir='iq_data'):
    """Charge tous les fichiers WAV avec leurs labels."""
    X = []  # Signaux
    y = []  # Labels (nom du bateau)

    for boat_dir in Path(data_dir).iterdir():
        if not boat_dir.is_dir():
            continue

        boat_name = boat_dir.name

        for radar_dir in boat_dir.iterdir():
            if not radar_dir.is_dir():
                continue

            radar_name = radar_dir.name

            # Charger tous les fichiers .wav
            for wav_file in radar_dir.glob('*.wav'):
                with wave.open(str(wav_file), 'r') as wav:
                    signal = np.frombuffer(wav.readframes(-1), dtype=np.int16)
                    signal = signal.astype(np.float32) / 32768.0

                X.append(signal)
                y.append(f"{boat_name}_{radar_name}")

    return X, y

# Utilisation
X, y = load_dataset()
print(f"Dataset: {len(X)} échantillons")
print(f"Classes: {len(set(y))} radars différents")
```

### Exemple de traitement signal pour features

```python
import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq

def extract_features(wav_signal, sample_rate=48000):
    """Extrait des features du signal WAV pour l'IA."""

    # 1. Spectre de puissance (FFT)
    fft_vals = fft(wav_signal)
    fft_freq = fftfreq(len(wav_signal), 1/sample_rate)
    power_spectrum = np.abs(fft_vals)**2

    # 2. Spectrogramme
    f, t, Sxx = sp_signal.spectrogram(wav_signal, fs=sample_rate)

    # 3. Features statistiques
    features = {
        'mean_power': np.mean(power_spectrum),
        'peak_power': np.max(power_spectrum),
        'spectral_centroid': np.sum(fft_freq[:len(fft_freq)//2] * power_spectrum[:len(power_spectrum)//2]) / np.sum(power_spectrum[:len(power_spectrum)//2]),
        'bandwidth': np.std(power_spectrum),
        'snr_estimate': 10 * np.log10(np.max(power_spectrum) / np.mean(power_spectrum))
    }

    return features, Sxx
```

## Personnalisation

### Modifier les paramètres de génération

```bash
# Générer plus de fichiers par radar
python generate_radar_wav.py --num-files 100

# Changer la durée des signaux
python generate_radar_wav.py --duration-min 3.0 --duration-max 15.0

# Utiliser un taux d'échantillonnage différent
python generate_radar_wav.py --sample-rate 96000
```

### Modifier boats.json pour ajouter des radars

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

## Avertissement

Ce code génère des **signaux simulés à des fins éducatives**. Les caractéristiques sont basées sur des données publiques et ne représentent pas forcément avec exactitude les signaux réels des systèmes radar militaires.

## Objectifs pédagogiques

- Comprendre la structure des signaux IQ
- Apprentissage du traitement du signal radar
- Entraînement de modèles de classification
- Analyse spectrale et extraction de features
- Développement d'algorithmes de détection
