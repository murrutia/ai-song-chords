#!/usr/bin/env python3
"""Transcrit les accords d'une chanson à partir d'un fichier audio.

Utilise le modèle BTC (Bi-directional Transformer for Chord recognition,
Park et al., ISMIR 2019) avec ses poids pré-entraînés, entièrement en local.

Usage :
    uv run chords.py "songs/ma chanson.m4a"
    uv run chords.py songs/*.wav --simple --spelling flats --out out/
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BTC_DIR = ROOT / "vendor" / "BTC-ISMIR19"

sys.path.insert(0, str(BTC_DIR))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import torch
import yaml

# Le code BTC date de 2019 et utilise l'alias np.float, supprimé depuis numpy 1.24.
np.float = float  # type: ignore[attr-defined]
from btc_model import BTC_model

SAMPLE_RATE = 22050
INST_LEN = 10.0  # secondes par bloc de CQT
N_BINS = 144
BINS_PER_OCTAVE = 24
HOP_LENGTH = 2048

ROOTS_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOTS_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
QUALITIES = [
    "min",
    "maj",
    "dim",
    "aug",
    "min6",
    "maj6",
    "min7",
    "minmaj7",
    "maj7",
    "7",
    "dim7",
    "hdim7",
    "sus2",
    "sus4",
]
# Notation "lead sheet" usuelle
QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "min6": "m6",
    "maj6": "6",
    "min7": "m7",
    "minmaj7": "m(maj7)",
    "maj7": "maj7",
    "7": "7",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "sus2": "sus2",
    "sus4": "sus4",
}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Décode n'importe quel format (m4a, mp3, wav, flac...) via ffmpeg, en mono."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def compute_features(wav: np.ndarray) -> np.ndarray:
    """Log-CQT par blocs de 10 s, identique au prétraitement d'entraînement de BTC."""
    block = int(SAMPLE_RATE * INST_LEN)
    chunks = []
    for start in range(0, len(wav), block):
        chunks.append(
            np.abs(
                librosa.cqt(
                    wav[start : start + block],
                    sr=SAMPLE_RATE,
                    n_bins=N_BINS,
                    bins_per_octave=BINS_PER_OCTAVE,
                    hop_length=HOP_LENGTH,
                )
            )
        )
    return np.log(np.concatenate(chunks, axis=1) + 1e-6).T  # (frames, 144)


class ChordRecognizer:
    def __init__(self, large_vocab: bool, device: torch.device):
        with open(BTC_DIR / "run_config.yaml") as f:
            config = yaml.safe_load(f)["model"]
        if large_vocab:
            config["num_chords"] = 170
            weights = BTC_DIR / "test" / "btc_model_large_voca.pt"
        else:
            weights = BTC_DIR / "test" / "btc_model.pt"
        self.large_vocab = large_vocab
        self.timestep = config["timestep"]
        self.frame_sec = INST_LEN / self.timestep
        self.device = device

        # weights_only=False : le checkpoint contient des tableaux numpy (mean/std).
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.mean, self.std = checkpoint["mean"], checkpoint["std"]
        self.model = BTC_model(config=config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()

    def label(self, idx: int) -> tuple[int, str] | None:
        """Retourne (racine 0-11, qualité) ou None pour 'pas d'accord'."""
        if self.large_vocab:
            if idx >= 168:
                return None
            return idx // 14, QUALITIES[idx % 14]
        if idx == 24:
            return None
        return idx // 2, "min" if idx % 2 else "maj"

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Indice d'accord prédit pour chaque trame."""
        n_frames = features.shape[0]
        x = (features - self.mean) / self.std
        pad = (-n_frames) % self.timestep
        x = np.pad(x, ((0, pad), (0, 0)))
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = x.view(-1, self.timestep, x.shape[-1])  # un batch de fenêtres de 108 trames
        hidden, _ = self.model.self_attn_layers(x)
        prediction, _ = self.model.output_layer(hidden)
        return prediction.reshape(-1)[:n_frames].cpu().numpy()


def to_segments(frames: np.ndarray, frame_sec: float, min_dur: float):
    """Regroupe les trames en segments (début, fin, indice) et absorbe les segments trop courts."""
    segs = []
    start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] != frames[start]:
            segs.append([start * frame_sec, i * frame_sec, int(frames[start])])
            start = i

    # Fusionne itérativement le segment le plus court avec son voisin le plus long.
    while len(segs) > 1:
        durations = [e - s for s, e, _ in segs]
        k = int(np.argmin(durations))
        if durations[k] >= min_dur:
            break
        if k == 0:
            j = 1
        elif k == len(segs) - 1:
            j = k - 1
        else:
            j = k - 1 if durations[k - 1] >= durations[k + 1] else k + 1
        lo, hi = min(j, k), max(j, k)
        segs[lo] = [segs[lo][0], segs[hi][1], segs[j][2]]
        del segs[hi]
        # Recolle les voisins devenus identiques
        merged = [segs[0]]
        for s in segs[1:]:
            if s[2] == merged[-1][2]:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        segs = merged
    return segs


MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
FLAT_KEYS = {
    5,
    10,
    3,
    8,
    1,
    6,
}  # F, Bb, Eb, Ab, Db, Gb majeurs (et leurs relatifs mineurs)


def estimate_key(durations: dict) -> int:
    """Tonalité majeure (0-11) dont la gamme couvre le plus de durée d'accords.

    Une tonalité mineure est représentée par son relatif majeur (Do mineur -> Mib).
    """

    def triad(root, quality):
        third = 3 if quality.startswith(("min", "dim", "hdim")) else 4
        fifth = (
            6 if quality in ("dim", "dim7", "hdim7") else 8 if quality == "aug" else 7
        )
        if quality.startswith("sus"):
            third = 2 if quality == "sus2" else 5
        return {root, (root + third) % 12, (root + fifth) % 12}

    def score(key):
        scale = {(key + i) % 12 for i in MAJOR_SCALE}
        return sum(d * len(triad(*label) & scale) for label, d in durations.items())

    return max(range(12), key=score)


def chord_name(label, flats: bool) -> str:
    if label is None:
        return "N.C."
    root, quality = label
    return (ROOTS_FLAT if flats else ROOTS_SHARP)[root] + QUALITY_SUFFIX[quality]


def harte_name(label) -> str:
    """Notation Harte (format .lab standard en MIR)."""
    if label is None:
        return "N"
    root, quality = label
    return ROOTS_SHARP[root] + ("" if quality == "maj" else ":" + quality)


def fmt_time(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def analyze(
    recognizer: ChordRecognizer, wav: np.ndarray, min_dur: float, spelling: str
):
    """Segments d'accords [(début, fin, label, nom)], tonalité majeure estimée et choix bémols."""
    frames = recognizer.predict(compute_features(wav))
    segs = to_segments(frames, recognizer.frame_sec, min_dur)
    segs[-1][1] = min(segs[-1][1], len(wav) / SAMPLE_RATE)

    labels = [recognizer.label(idx) for _, _, idx in segs]
    durations = defaultdict(float)
    for (start, end, _), label in zip(segs, labels):
        if label is not None:
            durations[label] += end - start
    key = estimate_key(durations)
    flats = key in FLAT_KEYS if spelling == "auto" else spelling == "flats"
    chords = [
        (start, end, label, chord_name(label, flats))
        for (start, end, _), label in zip(segs, labels)
    ]
    return chords, key, flats


def key_name(key: int, flats: bool) -> str:
    roots = ROOTS_FLAT if flats else ROOTS_SHARP
    return f"{roots[key]} majeur / {roots[(key + 9) % 12]} mineur"


def add_chord_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--simple",
        action="store_true",
        help="vocabulaire réduit majeur/mineur (25 classes) au lieu de 170 accords",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="durée minimale d'un accord en secondes (défaut 0.5)",
    )
    parser.add_argument(
        "--spelling",
        choices=["auto", "sharps", "flats"],
        default="auto",
        help="altérations : auto (selon la tonalité estimée), dièses ou bémols",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, mps ou cuda (modèle d'accords)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("audio", nargs="+", type=Path, help="fichier(s) audio")
    add_chord_args(parser)
    parser.add_argument(
        "--out", type=Path, help="dossier où écrire les fichiers .lab et .txt"
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    recognizer = ChordRecognizer(large_vocab=not args.simple, device=device)
    print(
        f"Modèle : {'majeur/mineur' if args.simple else '170 accords'} sur {device}",
        file=sys.stderr,
    )

    for path in args.audio:
        print(f"\n=== {path.name} ===")
        chords, key, flats = analyze(
            recognizer, load_audio(path), args.min_duration, args.spelling
        )
        print(f"Tonalité estimée : {key_name(key, flats)}\n")

        lines, lab = [], []
        totals = defaultdict(float)
        for start, end, label, name in chords:
            lines.append(f"{fmt_time(start)}  {name}")
            lab.append(f"{start:.3f} {end:.3f} {harte_name(label)}")
            if label is not None:
                totals[name] += end - start

        print("\n".join(lines))
        print("\nAccords utilisés (par durée totale) :")
        print(
            "  "
            + "  ".join(
                f"{n} ({d:.0f}s)"
                for n, d in sorted(totals.items(), key=lambda x: -x[1])
            )
        )

        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{path.stem}.lab").write_text("\n".join(lab) + "\n")
            (args.out / f"{path.stem}.txt").write_text("\n".join(lines) + "\n")
            print(f"\n-> {args.out / path.stem}.{{lab,txt}}", file=sys.stderr)


if __name__ == "__main__":
    main()
