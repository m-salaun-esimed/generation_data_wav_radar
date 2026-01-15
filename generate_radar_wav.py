#!/usr/bin/env python3
"""
Générateur de fichiers WAV pour simulation de signaux radar.
Usage éducatif pour l'entraînement de modèles d'IA de détection radar.

Usage:
    python generate_radar_wav.py                    # Génère pour TOUS les radars
    python generate_radar_wav.py --num-files 50     # 50 fichiers par radar
    python generate_radar_wav.py --list             # Liste les radars disponibles
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple
import wave
import struct


class RadarWAVGenerator:
    """Générateur de signaux radar au format WAV avec paramètres aléatoires."""

    def __init__(self, sample_rate: int = 48000):
        """
        Initialise le générateur.

        Args:
            sample_rate: Taux d'échantillonnage audio en Hz (48 kHz par défaut)
        """
        self.sample_rate = sample_rate

    def load_radar_info(self, boat_name: str, radar_name: str, json_file: str = 'boats.json') -> Dict:
        """
        Charge les informations d'un radar spécifique depuis le fichier JSON.

        Args:
            boat_name: Nom du bateau (ex: "CDG", "FREMM", "FLF", "FDA")
            radar_name: Nom du radar (ex: "smart_s_mK2", "HERAKLES_EF")
            json_file: Fichier JSON contenant les données

        Returns:
            Dictionnaire avec les informations du radar

        Raises:
            ValueError: Si le bateau ou le radar n'est pas trouvé
        """
        with open(json_file, 'r') as f:
            boats_data = json.load(f)

        # Vérifier si le bateau existe
        if boat_name not in boats_data:
            raise ValueError(f"Bateau '{boat_name}' non trouvé dans {json_file}. Bateaux disponibles: {list(boats_data.keys())}")

        # Recherche du radar dans le bateau spécifié
        boat_info = boats_data[boat_name]
        for radar in boat_info['radar_systems']:
            if radar['name'] == radar_name:
                radar_info = radar.copy()
                radar_info['boat'] = boat_name
                return radar_info

        # Si radar non trouvé, lister les radars disponibles pour ce bateau
        available_radars = [r['name'] for r in boat_info['radar_systems']]
        raise ValueError(f"Radar '{radar_name}' non trouvé sur le bateau '{boat_name}'. Radars disponibles: {available_radars}")

    def generate_radar_signal(self,
                             center_freq_mhz: float,
                             bandwidth_mhz: float,
                             duration_sec: float,
                             signal_power: float,
                             noise_power: float,
                             prf_hz: float = None,
                             duty_cycle: float = 0.1) -> np.ndarray:
        """
        Génère un signal radar avec paramètres spécifiques.

        Args:
            center_freq_mhz: Fréquence centrale en MHz
            bandwidth_mhz: Largeur de bande en MHz
            duration_sec: Durée totale du signal en secondes
            signal_power: Puissance du signal (0-1)
            noise_power: Puissance du bruit (0-1)
            prf_hz: Pulse Repetition Frequency en Hz (None = mode continu)
            duty_cycle: Rapport cyclique pour les pulses (0-1)

        Returns:
            Signal audio (numpy array)
        """
        num_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, num_samples)

        # Conversion de la fréquence radar en fréquence audio
        # On transpose la fréquence radar vers le domaine audio (20 Hz - 20 kHz)
        # en gardant les proportions
        audio_center_freq = 1000 + (center_freq_mhz % 1000)  # Fréquence audio entre 1-2 kHz
        audio_bandwidth = min(bandwidth_mhz * 10, 5000)  # Largeur de bande audio

        if prf_hz is None:
            # Mode continu (CW - Continuous Wave)
            # Signal avec modulation FM (chirp)
            chirp_rate = audio_bandwidth / duration_sec
            phase = 2 * np.pi * (audio_center_freq * t + 0.5 * chirp_rate * t**2)
            signal = signal_power * np.sin(phase)

        else:
            # Mode pulsé
            pulse_period = 1.0 / prf_hz
            pulse_duration = pulse_period * duty_cycle
            num_pulses = int(duration_sec / pulse_period)

            signal = np.zeros(num_samples)

            for pulse_idx in range(num_pulses):
                # Génération d'un pulse avec chirp
                pulse_start_time = pulse_idx * pulse_period
                pulse_end_time = pulse_start_time + pulse_duration

                # Masque temporel pour ce pulse
                pulse_mask = (t >= pulse_start_time) & (t < pulse_end_time)

                if np.any(pulse_mask):
                    t_pulse = t[pulse_mask] - pulse_start_time
                    chirp_rate = audio_bandwidth / pulse_duration
                    phase = 2 * np.pi * (audio_center_freq * t_pulse + 0.5 * chirp_rate * t_pulse**2)

                    # Envelope gaussienne pour chaque pulse
                    envelope = np.exp(-((t_pulse - pulse_duration/2) / (pulse_duration/6))**2)
                    signal[pulse_mask] = signal_power * envelope * np.sin(phase)

        # Ajout de bruit gaussien
        noise = np.random.normal(0, noise_power, num_samples)
        signal_with_noise = signal + noise

        return signal_with_noise

    def calculate_snr(self, signal_power: float, noise_power: float) -> float:
        """Calcule le SNR en dB."""
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power**2 / noise_power**2)

    def save_wav(self, signal: np.ndarray, filename: str, normalize: bool = True):
        """
        Sauvegarde le signal au format WAV 16-bit.

        Args:
            signal: Signal audio à sauvegarder
            filename: Nom du fichier WAV
            normalize: Normaliser le signal à [-1, 1] avant sauvegarde
        """
        # Normalisation
        if normalize:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val

        # Conversion en int16
        signal_int16 = np.int16(signal * 32767)

        # Sauvegarde WAV
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(signal_int16.tobytes())

    def generate_dataset(self,
                        boat_name: str,
                        radar_name: str,
                        output_dir: str = None,
                        num_files: int = 20,
                        duration_range: Tuple[float, float] = (5.0, 10.0),
                        power_range: Tuple[float, float] = (0.3, 1.0),
                        noise_range: Tuple[float, float] = (0.01, 0.3)):
        """
        Génère un dataset de fichiers WAV pour un radar spécifique.

        Args:
            boat_name: Nom du bateau
            radar_name: Nom du radar
            output_dir: Dossier de sortie (par défaut = nom du radar)
            num_files: Nombre de fichiers à générer
            duration_range: Plage de durées en secondes (min, max)
            power_range: Plage de puissance du signal (min, max)
            noise_range: Plage de puissance du bruit (min, max)
        """
        # Chargement des infos du radar
        try:
            radar_info = self.load_radar_info(boat_name, radar_name)
        except ValueError as e:
            print(f"Erreur: {e}")
            print("\nUtilisez --list pour voir les bateaux et radars disponibles")
            return

        # Création du dossier de sortie
        if output_dir is None:
            output_dir = radar_name

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print(f"Génération de {num_files} fichiers WAV pour le radar: {radar_name}")
        print(f"Bateau: {radar_info['boat']}")
        print("=" * 70)

        # Récupération des caractéristiques du radar
        freq_ranges = radar_info['frequency_ranges']
        if len(freq_ranges) == 0:
            print("Erreur: Aucune plage de fréquence définie pour ce radar")
            return

        # Utilisation de la première plage de fréquence
        freq_range = freq_ranges[0]
        band = freq_range['band']
        min_freq = freq_range['min_mhz']
        max_freq = freq_range['max_mhz']
        center_freq = (min_freq + max_freq) / 2
        bandwidth = max_freq - min_freq

        # Récupération des paramètres de signature du radar (DI et PRI)
        di_ms = radar_info.get('DI_ms', 50)  # Durée d'impulsion en millisecondes
        pri_ms = radar_info.get('PRI_ms', 1000)  # Période de répétition en millisecondes

        # Calcul du PRF et duty cycle basé sur DI et PRI
        prf_hz = 1000.0 / pri_ms  # PRF en Hz
        pulse_duration_sec = di_ms / 1000.0  # Durée d'impulsion en secondes
        pulse_period_sec = pri_ms / 1000.0  # Période en secondes
        duty_cycle = pulse_duration_sec / pulse_period_sec  # Duty cycle

        print(f"\nCaractéristiques radar:")
        print(f"  Bande: {band}")
        print(f"  Fréquence: {min_freq} - {max_freq} MHz")
        print(f"  Centre: {center_freq} MHz")
        print(f"  Bande passante: {bandwidth} MHz")
        print(f"  DI (Durée d'Impulsion): {di_ms} ms")
        print(f"  PRI (Période de Répétition): {pri_ms} ms")
        print(f"  PRF (Pulse Repetition Frequency): {prf_hz:.2f} Hz")
        print(f"  Duty Cycle: {duty_cycle*100:.2f}%")
        print(f"\nParamètres de génération:")
        print(f"  Durée: {duration_range[0]}-{duration_range[1]} secondes")
        print(f"  Puissance signal: {power_range[0]}-{power_range[1]}")
        print(f"  Puissance bruit: {noise_range[0]}-{noise_range[1]}")
        print()

        # Métadonnées du dataset
        metadata = {
            "radar_name": radar_name,
            "boat": radar_info['boat'],
            "band": band,
            "center_freq_mhz": center_freq,
            "bandwidth_mhz": bandwidth,
            "DI_ms": di_ms,
            "PRI_ms": pri_ms,
            "PRF_hz": prf_hz,
            "duty_cycle": duty_cycle,
            "sample_rate_hz": self.sample_rate,
            "num_files": num_files,
            "files": []
        }

        # Génération des fichiers
        for i in range(num_files):
            # Paramètres aléatoires pour ce fichier
            duration = np.random.uniform(*duration_range)
            signal_power = np.random.uniform(*power_range)
            noise_power = np.random.uniform(*noise_range)

            # Variation de fréquence (±5% autour du centre)
            freq_variation = np.random.uniform(-bandwidth*0.05, bandwidth*0.05)
            actual_center_freq = center_freq + freq_variation

            # Calcul du SNR
            snr_db = self.calculate_snr(signal_power, noise_power)

            # Génération du signal avec les paramètres fixes du radar (PRF et duty cycle)
            signal = self.generate_radar_signal(
                center_freq_mhz=actual_center_freq,
                bandwidth_mhz=bandwidth,
                duration_sec=duration,
                signal_power=signal_power,
                noise_power=noise_power,
                prf_hz=prf_hz,  # PRF fixe basé sur PRI du radar
                duty_cycle=duty_cycle  # Duty cycle fixe basé sur DI et PRI
            )

            # Nom du fichier
            filename = f"{radar_name}_sample_{i:03d}_snr{snr_db:.1f}dB.wav"
            filepath = output_path / filename

            # Sauvegarde
            self.save_wav(signal, str(filepath))

            # Métadonnées de ce fichier
            file_metadata = {
                "filename": filename,
                "duration_sec": duration,
                "signal_power": signal_power,
                "noise_power": noise_power,
                "snr_db": snr_db,
                "center_freq_mhz": actual_center_freq,
                "num_samples": len(signal)
            }
            metadata["files"].append(file_metadata)

            # Affichage de la progression
            print(f"[{i+1:2d}/{num_files}] {filename:60s} | "
                  f"Durée: {duration:4.2f}s | SNR: {snr_db:5.1f} dB")

        # Sauvegarde des métadonnées
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 70)
        print(f"Génération terminée!")
        print(f"Dossier de sortie: {output_path.absolute()}")
        print(f"Fichiers générés: {num_files}")
        print(f"Métadonnées: {metadata_file}")
        print("=" * 70)

    def list_available_radars(self, json_file: str = 'boats.json'):
        """Liste tous les radars disponibles dans le fichier JSON."""
        with open(json_file, 'r') as f:
            boats_data = json.load(f)

        print("\nRadars disponibles par bateau:")
        print("-" * 70)
        for boat_name, boat_info in boats_data.items():
            print(f"\n{boat_name}:")
            for radar in boat_info['radar_systems']:
                freq_ranges = radar['frequency_ranges']
                di_ms = radar.get('DI_ms', 'N/A')
                pri_ms = radar.get('PRI_ms', 'N/A')

                if freq_ranges:
                    freq_info = ", ".join([f"{fr['band']}: {fr['min_mhz']}-{fr['max_mhz']} MHz"
                                          for fr in freq_ranges])
                    print(f"  - {radar['name']:30s} | {freq_info:30s} | DI: {di_ms:3}ms | PRI: {pri_ms:4}ms")
                else:
                    print(f"  - {radar['name']}")


def main():
    """Fonction principale - génère automatiquement pour tous les radars."""
    parser = argparse.ArgumentParser(
        description='Générateur de fichiers WAV de signaux radar simulés',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python generate_radar_wav.py
  python generate_radar_wav.py --num-files 50
  python generate_radar_wav.py --list
        """
    )

    parser.add_argument('--num-files', '-n', type=int, default=20,
                       help='Nombre de fichiers à générer par radar (défaut: 20)')

    parser.add_argument('--duration-min', type=float, default=5.0,
                       help='Durée minimale en secondes (défaut: 5.0)')

    parser.add_argument('--duration-max', type=float, default=10.0,
                       help='Durée maximale en secondes (défaut: 10.0)')

    parser.add_argument('--sample-rate', type=int, default=48000,
                       help='Taux d\'échantillonnage audio en Hz (défaut: 48000)')

    parser.add_argument('--list', '-l', action='store_true',
                       help='Liste tous les radars disponibles')

    parser.add_argument('--output-base-dir', '-o', type=str, default='iq_data',
                       help='Dossier de base pour la sortie (défaut: iq_data)')

    args = parser.parse_args()

    # Création du générateur
    generator = RadarWAVGenerator(sample_rate=args.sample_rate)

    # Liste des radars disponibles
    if args.list:
        generator.list_available_radars()
        return

    # Chargement du fichier JSON
    with open('boats.json', 'r') as f:
        boats_data = json.load(f)

    print("=" * 70)
    print("Génération automatique pour TOUS les radars")
    print("=" * 70)

    # Compteur global
    total_radars = sum(len(boat_info['radar_systems']) for boat_info in boats_data.values())
    radar_count = 0

    # Itération sur tous les bateaux et tous les radars
    for boat_name, boat_info in boats_data.items():
        for radar in boat_info['radar_systems']:
            radar_count += 1
            radar_name = radar['name']

            print(f"\n[{radar_count}/{total_radars}] Traitement: {boat_name} / {radar_name}")
            print("-" * 70)

            # Dossier de sortie : output_base_dir/BATEAU/RADAR/
            output_dir = f"{args.output_base_dir}/{boat_name}/{radar_name}"

            # Génération du dataset
            generator.generate_dataset(
                boat_name=boat_name,
                radar_name=radar_name,
                output_dir=output_dir,
                num_files=args.num_files,
                duration_range=(args.duration_min, args.duration_max)
            )

    print("\n" + "=" * 70)
    print(f"GÉNÉRATION COMPLÈTE!")
    print(f"Total: {total_radars} radars traités")
    print(f"Dossier racine: {args.output_base_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
