#!/usr/bin/env python3
"""
Générateur de fichiers WAV stéréo IQ pour simulation de signaux radar.
Compatible avec SDRangel (canal gauche = I, canal droit = Q).
Usage éducatif pour l'entraînement de modèles d'IA de détection radar.

Usage:
    python generate_radar_iq.py                    # Génère pour TOUS les radars
    python generate_radar_iq.py --num-files 10     # 10 fichiers par radar
    python generate_radar_iq.py --list             # Liste les radars disponibles
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple
import wave


class RadarIQGenerator:
    def __init__(self, sample_rate: float = 2e6):
        """
        Initialise le générateur IQ.

        Args:
            sample_rate: Taux d'échantillonnage en Hz (2 MHz par défaut pour SDR)
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

        if boat_name not in boats_data:
            raise ValueError(f"Bateau '{boat_name}' non trouvé dans {json_file}. Bateaux disponibles: {list(boats_data.keys())}")

        boat_info = boats_data[boat_name]
        for radar in boat_info['radar_systems']:
            if radar['name'] == radar_name:
                radar_info = radar.copy()
                radar_info['boat'] = boat_name
                return radar_info

        available_radars = [r['name'] for r in boat_info['radar_systems']]
        raise ValueError(f"Radar '{radar_name}' non trouvé sur le bateau '{boat_name}'. Radars disponibles: {available_radars}")

    def generate_iq_signal(self,
                          center_freq_hz: float,
                          bandwidth_hz: float,
                          duration_sec: float,
                          signal_power: float,
                          noise_power: float,
                          prf_hz: float = 1000.0,
                          duty_cycle: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère un signal IQ complexe simulant un radar pulsé.

        Args:
            center_freq_hz: Fréquence centrale en Hz
            bandwidth_hz: Largeur de bande en Hz
            duration_sec: Durée du signal en secondes
            signal_power: Puissance du signal (0.0-1.0)
            noise_power: Puissance du bruit (0.0-1.0)
            prf_hz: Pulse Repetition Frequency en Hz
            duty_cycle: Rapport cyclique (0.0-1.0)

        Returns:
            Tuple (I, Q) avec les composantes In-phase et Quadrature
        """
        num_samples = int(self.sample_rate * duration_sec)
        t = np.arange(num_samples) / self.sample_rate

        # Calcul des paramètres de pulse
        pulse_period = 1.0 / prf_hz
        pulse_duration = pulse_period * duty_cycle

        # Génération de l'enveloppe pulsée
        pulse_train = np.zeros(num_samples)
        samples_per_period = int(self.sample_rate * pulse_period)
        samples_per_pulse = int(self.sample_rate * pulse_duration)

        for i in range(0, num_samples, samples_per_period):
            if i + samples_per_pulse <= num_samples:
                # Enveloppe gaussienne pour chaque pulse
                pulse_t = np.linspace(-3, 3, samples_per_pulse)
                gaussian_envelope = np.exp(-pulse_t**2)
                pulse_train[i:i+samples_per_pulse] = gaussian_envelope

        # Modulation LFM (Linear Frequency Modulation / Chirp)
        chirp_rate = bandwidth_hz / pulse_duration
        phase = 2 * np.pi * center_freq_hz * t + np.pi * chirp_rate * (t % pulse_period)**2

        # Signal complexe (I + jQ)
        complex_signal = pulse_train * np.exp(1j * phase)

        # Ajout de bruit complexe gaussien
        noise_i = np.random.normal(0, noise_power, num_samples)
        noise_q = np.random.normal(0, noise_power, num_samples)
        noise = noise_i + 1j * noise_q

        # Signal final avec puissance
        signal = signal_power * complex_signal + noise

        # Normalisation
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val

        # Séparation I et Q
        I = np.real(signal).astype(np.float32)
        Q = np.imag(signal).astype(np.float32)

        return I, Q

    def save_wav_stereo(self, I: np.ndarray, Q: np.ndarray, filename: str):
        """
        Sauvegarde les données IQ au format WAV stéréo (compatible SDRangel).

        Format: Canal gauche = I, Canal droit = Q
        Type: WAV 16-bit stéréo

        Args:
            I: Composante In-phase
            Q: Composante Quadrature
            filename: Chemin du fichier de sortie
        """
        # Conversion en int16 (WAV 16-bit)
        I_int16 = np.clip(I * 32767, -32768, 32767).astype(np.int16)
        Q_int16 = np.clip(Q * 32767, -32768, 32767).astype(np.int16)

        # Entrelacement stéréo [L0, R0, L1, R1, ...] = [I0, Q0, I1, Q1, ...]
        stereo_interleaved = np.empty(2 * len(I), dtype=np.int16)
        stereo_interleaved[0::2] = I_int16
        stereo_interleaved[1::2] = Q_int16

        # Sauvegarde WAV
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(2)  # Stéréo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(int(self.sample_rate))
            wav_file.writeframes(stereo_interleaved.tobytes())

    def calculate_snr(self, signal_power: float, noise_power: float) -> float:
        """Calcule le SNR en dB."""
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return float('inf')

    def generate_dataset(self,
                        boat_name: str,
                        radar_name: str,
                        output_dir: str = None,
                        num_files: int = 10,
                        duration_range: Tuple[float, float] = (1.0, 5.0),
                        power_range: Tuple[float, float] = (0.3, 1.0),
                        noise_range: Tuple[float, float] = (0.01, 0.3)):
        """
        Génère un dataset de fichiers IQ pour un radar spécifique.

        Args:
            boat_name: Nom du bateau
            radar_name: Nom du radar
            output_dir: Dossier de sortie
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

        # Affichage des informations
        print("=" * 70)
        print(f"Radar: {radar_name}")
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
        min_freq_mhz = freq_range['min_mhz']
        max_freq_mhz = freq_range['max_mhz']
        bandwidth_mhz = max_freq_mhz - min_freq_mhz

        # Conversion en Hz pour la génération IQ
        min_freq_hz = min_freq_mhz * 1e6
        max_freq_hz = max_freq_mhz * 1e6
        bandwidth_hz = bandwidth_mhz * 1e6

        # Récupération des paramètres de signature du radar (DI et PRI)
        di_ms = radar_info.get('DI_ms', 50)
        pri_ms = radar_info.get('PRI_ms', 1000)

        # Calcul du PRF et duty cycle
        prf_hz = 1000.0 / pri_ms
        pulse_duration_sec = di_ms / 1000.0
        pulse_period_sec = pri_ms / 1000.0
        duty_cycle = pulse_duration_sec / pulse_period_sec

        print(f"\nCaractéristiques radar:")
        print(f"  Bande: {band}")
        print(f"  Fréquence: {min_freq_mhz} - {max_freq_mhz} MHz")
        print(f"  Bande passante: {bandwidth_mhz} MHz")
        print(f"  DI (Durée d'Impulsion): {di_ms} ms")
        print(f"  PRI (Période de Répétition): {pri_ms} ms")
        print(f"  PRF (Pulse Repetition Frequency): {prf_hz:.2f} Hz")
        print(f"  Duty Cycle: {duty_cycle*100:.2f}%")
        print(f"\nParamètres de génération:")
        print(f"  Taux d'échantillonnage: {self.sample_rate/1e6:.1f} MHz")
        print(f"  Fréquence: Aléatoire entre {min_freq_mhz}-{max_freq_mhz} MHz")
        print(f"  Durée: {duration_range[0]}-{duration_range[1]} secondes")
        print(f"  Puissance signal: {power_range[0]}-{power_range[1]}")
        print(f"  Puissance bruit: {noise_range[0]}-{noise_range[1]}")
        print()

        # Métadonnées du dataset
        metadata = {
            "radar_name": radar_name,
            "boat": radar_info['boat'],
            "band": band,
            "min_freq_mhz": min_freq_mhz,
            "max_freq_mhz": max_freq_mhz,
            "bandwidth_mhz": bandwidth_mhz,
            "DI_ms": di_ms,
            "PRI_ms": pri_ms,
            "PRF_hz": prf_hz,
            "duty_cycle": duty_cycle,
            "sample_rate_hz": self.sample_rate,
            "format": "WAV 16-bit stereo (L=I, R=Q)",
            "num_files": num_files,
            "files": []
        }

        # Génération des fichiers
        for i in range(num_files):
            # Paramètres aléatoires pour ce fichier
            duration = np.random.uniform(*duration_range)
            signal_power = np.random.uniform(*power_range)
            noise_power = np.random.uniform(*noise_range)

            # Fréquence centrale aléatoire entre min et max
            actual_center_freq_hz = np.random.uniform(min_freq_hz, max_freq_hz)
            actual_center_freq_mhz = actual_center_freq_hz / 1e6

            # Calcul du SNR
            snr_db = self.calculate_snr(signal_power, noise_power)

            # Génération du signal IQ
            I, Q = self.generate_iq_signal(
                center_freq_hz=actual_center_freq_hz,
                bandwidth_hz=bandwidth_hz,
                duration_sec=duration,
                signal_power=signal_power,
                noise_power=noise_power,
                prf_hz=prf_hz,
                duty_cycle=duty_cycle
            )

            # Nom du fichier avec fréquence centrale
            filename = f"{radar_name}_sample_{i:03d}_{actual_center_freq_mhz:.1f}MHz_snr{snr_db:.1f}dB.wav"
            filepath = output_path / filename

            # Sauvegarde
            self.save_wav_stereo(I, Q, str(filepath))

            # Métadonnées de ce fichier
            file_metadata = {
                "filename": filename,
                "duration_sec": duration,
                "signal_power": signal_power,
                "noise_power": noise_power,
                "snr_db": snr_db,
                "center_freq_mhz": actual_center_freq_mhz,
                "num_samples": len(I)
            }
            metadata["files"].append(file_metadata)

            # Affichage de la progression
            print(f"[{i+1:2d}/{num_files}] {filename:60s} | "
                  f"Durée: {duration:4.2f}s | SNR: {snr_db:5.1f} dB | {len(I)} échantillons")

        # Sauvegarde des métadonnées
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 70)
        print(f"Génération terminée!")
        print(f"Fichiers: {output_path}/*.wav")
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
        description='Générateur de fichiers WAV stéréo IQ de signaux radar simulés (compatible SDRangel)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python generate_radar_iq.py
  python generate_radar_iq.py --num-files 10
  python generate_radar_iq.py --sample-rate 5000000
  python generate_radar_iq.py --list
        """
    )

    parser.add_argument('--num-files', '-n', type=int, default=10,
                       help='Nombre de fichiers à générer par radar (défaut: 10)')

    parser.add_argument('--duration-min', type=float, default=1.0,
                       help='Durée minimale en secondes (défaut: 1.0)')

    parser.add_argument('--duration-max', type=float, default=5.0,
                       help='Durée maximale en secondes (défaut: 5.0)')

    parser.add_argument('--sample-rate', type=float, default=2e6,
                       help='Taux d\'échantillonnage IQ en Hz (défaut: 2000000 = 2 MHz)')

    parser.add_argument('--list', '-l', action='store_true',
                       help='Liste tous les radars disponibles')

    parser.add_argument('--output-base-dir', '-o', type=str, default='iq_files',
                       help='Dossier de base pour la sortie (défaut: iq_files)')

    args = parser.parse_args()

    # Création du générateur
    generator = RadarIQGenerator(sample_rate=args.sample_rate)

    # Liste des radars disponibles
    if args.list:
        generator.list_available_radars()
        return

    # Chargement du fichier JSON
    with open('boats.json', 'r') as f:
        boats_data = json.load(f)

    print("=" * 70)
    print("Génération automatique pour TOUS les radars - Format WAV stéréo IQ")
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
    print(f"\nFormat: WAV 16-bit stéréo (Canal gauche = I, Canal droit = Q)")
    print(f"Compatible avec SDRangel")
    print(f"Pour ouvrir dans SDRangel:")
    print(f"  File -> Open IQ file -> Sélectionner un fichier .wav")
    print(f"  Sample rate: {args.sample_rate/1e6:.1f} MHz")
    print("=" * 70)


if __name__ == "__main__":
    main()
