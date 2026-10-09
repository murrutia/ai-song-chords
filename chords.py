#!/usr/bin/env python3
"""Transcrit les accords d'une chanson à partir d'un fichier audio.

Deux modèles pré-entraînés, entièrement en local :
- btc : Bi-directional Transformer for Chord recognition (Park et al., ISMIR 2019)
- cnn : Chord-CNN-LSTM (Jiang et al., ISMIR 2019), avec renversements (C/E)
  et accords enrichis (9, 11, 13)

Option --separate : Demucs retire d'abord la batterie (--separate) ou la
batterie et la voix (--separate instruments) avant l'analyse des accords.

Usage :
    uv run chords.py "songs/ma chanson.m4a"
    uv run chords.py songs/*.wav --model cnn --separate --out out/
    uv run chords.py songs/*.wav --separate instruments
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BTC_DIR = ROOT / "vendor" / "BTC-ISMIR19"
CNN_DIR = ROOT / "vendor" / "chord-cnn-lstm"
STEMS_CACHE = ROOT / ".cache" / "stems"

sys.path.insert(0, str(BTC_DIR))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import torch
import yaml

# Le code de ces modèles date de 2019 et utilise des alias supprimés depuis numpy 1.24.
np.float = float  # type: ignore[attr-defined]
np.int = int  # type: ignore[attr-defined]
from btc_model import BTC_model

SAMPLE_RATE = 22050  # attendu par les deux modèles d'accords
INST_LEN = 10.0  # secondes par bloc de CQT (BTC)
N_BINS = 144
BINS_PER_OCTAVE = 24
HOP_LENGTH = 2048

ROOTS_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOTS_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DEGREE_SEMITONES = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11, "9": 14}
BTC_QUALITIES = [
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
    "sus4(b7)": "7sus4",
    "9": "9",
    "maj9": "maj9",
    "min9": "m9",
    "11": "11",
    "13": "13",
}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_audio(path: Path, sr: int = SAMPLE_RATE, channels: int = 1) -> np.ndarray:
    """Décode n'importe quel format (m4a, mp3, wav, flac...) via ffmpeg.

    Retourne un tableau (échantillons,) en mono, (canaux, échantillons) sinon.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le"]
    cmd += ["-ac", str(channels), "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    wav = np.frombuffer(raw, dtype=np.float32)
    return wav if channels == 1 else wav.reshape(-1, channels).T


# --- Séparation de sources ---------------------------------------------------


def separate(path: Path, device: torch.device, model: str = "htdemucs") -> dict:
    """Pistes Demucs en mono à leur fréquence d'origine : {"accomp", "vocals", "sr"}.

    "accomp" = basse + autres instruments (ni voix ni batterie). Mis en cache dans .cache/stems.
    """
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime}|{model}".encode())
    cache = STEMS_CACHE / f"{path.stem[:60]}.{key.hexdigest()[:10]}.npz"
    if cache.exists():
        data = np.load(cache)
        return {"accomp": data["accomp"], "vocals": data["vocals"], "sr": int(data["sr"])}

    from demucs.api import Separator

    separator = Separator(model=model, device=str(device), progress=True)
    wav = load_audio(path, sr=separator.samplerate, channels=2)
    _, stems = separator.separate_tensor(torch.from_numpy(wav.copy()))
    accomp = (stems["bass"] + stems["other"]).mean(0).cpu().numpy()
    vocals = stems["vocals"].mean(0).cpu().numpy()
    result = {"accomp": accomp, "vocals": vocals, "sr": separator.samplerate}
    STEMS_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **result)
    return result


def resample(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    return librosa.resample(wav, orig_sr=sr_from, target_sr=sr_to).astype(np.float32)


# --- Modèles d'accords -------------------------------------------------------
# Les deux exposent recognize(wav) -> [(début, fin, accord Harte)], wav en mono 22.05 kHz.


class BTCRecognizer:
    def __init__(self, simple: bool, device: torch.device):
        with open(BTC_DIR / "run_config.yaml") as f:
            config = yaml.safe_load(f)["model"]
        if simple:
            weights = BTC_DIR / "test" / "btc_model.pt"
        else:
            config["num_chords"] = 170
            weights = BTC_DIR / "test" / "btc_model_large_voca.pt"
        self.simple = simple
        self.timestep = config["timestep"]
        self.frame_sec = INST_LEN / self.timestep
        self.device = device

        # weights_only=False : le checkpoint contient des tableaux numpy (mean/std).
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.mean, self.std = checkpoint["mean"], checkpoint["std"]
        self.model = BTC_model(config=config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()

    def harte(self, idx: int) -> str:
        if self.simple:
            if idx == 24:
                return "N"
            return ROOTS_SHARP[idx // 2] + (":min" if idx % 2 else ":maj")
        if idx >= 168:
            return "N"
        return f"{ROOTS_SHARP[idx // 14]}:{BTC_QUALITIES[idx % 14]}"

    @staticmethod
    def features(wav: np.ndarray) -> np.ndarray:
        """Log-CQT par blocs de 10 s, identique au prétraitement d'entraînement de BTC."""
        block = int(SAMPLE_RATE * INST_LEN)
        chunks = []
        for start in range(0, len(wav), block):
            cqt = librosa.cqt(
                wav[start : start + block],
                sr=SAMPLE_RATE,
                n_bins=N_BINS,
                bins_per_octave=BINS_PER_OCTAVE,
                hop_length=HOP_LENGTH,
            )
            chunks.append(np.abs(cqt))
        return np.log(np.concatenate(chunks, axis=1) + 1e-6).T  # (trames, 144)

    @torch.no_grad()
    def recognize(self, wav: np.ndarray) -> list:
        features = self.features(wav)
        n_frames = features.shape[0]
        x = (features - self.mean) / self.std
        x = np.pad(x, ((0, (-n_frames) % self.timestep), (0, 0)))
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = x.view(-1, self.timestep, x.shape[-1])  # un batch de fenêtres de 108 trames
        hidden, _ = self.model.self_attn_layers(x)
        prediction, _ = self.model.output_layer(hidden)
        frames = prediction.reshape(-1)[:n_frames].cpu().numpy()

        segs, start = [], 0
        for i in range(1, n_frames + 1):
            if i == n_frames or frames[i] != frames[start]:
                segs.append([start * self.frame_sec, i * self.frame_sec, self.harte(frames[start])])
                start = i
        return segs


class CnnLstmRecognizer:
    """Ensemble de 5 réseaux CNN-LSTM + décodage HMM sur un dictionnaire d'accords."""

    def __init__(self, simple: bool):
        # Le code d'origine résout ses chemins (cache_data/, data/) depuis le dossier courant.
        cwd = os.getcwd()
        sys.path.insert(0, str(CNN_DIR))
        os.chdir(CNN_DIR)
        try:
            from chord_recognition import MODEL_NAMES
            from chordnet_ismir_naive import ChordNet
            from extractors.cqt import CQTV2
            from extractors.xhmm_ismir import XHMMDecoder
            from mir import DataEntry, io
            from mir.nn.train import NetworkInterface

            self.nets = [NetworkInterface(ChordNet(None), name) for name in MODEL_NAMES]
            dictionary = "ismir2017" if simple else "submission"
            self.hmm = XHMMDecoder(template_file=str(CNN_DIR / "data" / f"{dictionary}_chord_list.txt"))
        finally:
            os.chdir(cwd)
        self.DataEntry, self.MusicIO, self.CQTV2 = DataEntry, io.MusicIO, CQTV2

    def recognize(self, wav: np.ndarray) -> list:
        entry = self.DataEntry()
        entry.prop.set("sr", SAMPLE_RATE)
        entry.prop.set("hop_length", 512)
        entry.append_data(wav, self.MusicIO, "music")
        entry.append_extractor(self.CQTV2, "cqt", cache_enabled=False)
        outputs = [net.inference(entry.cqt) for net in self.nets]
        probs = [np.mean([out[i] for out in outputs], axis=0) for i in range(len(outputs[0]))]
        return [list(seg) for seg in self.hmm.decode_to_chordlab(entry, probs, False)]


def make_recognizer(model: str, simple: bool, device: torch.device):
    if model == "cnn":
        return CnnLstmRecognizer(simple)
    return BTCRecognizer(simple, device)


# --- Notation ----------------------------------------------------------------


def parse_harte(chord: str):
    """'Bb:maj/3' -> (10, 'maj', 2) ; (racine, qualité, basse ou None). None pour N/X."""
    if chord in ("N", "X"):
        return None
    m = re.fullmatch(r"([A-G])([b#]*)(?::([^/]+))?(?:/([b#]*)(\d+))?", chord)
    if not m:
        raise ValueError(f"accord Harte non reconnu : {chord}")
    note, acc, quality, bass_acc, degree = m.groups()
    root = (NOTE_PC[note] + acc.count("#") - acc.count("b")) % 12
    bass = None
    if degree:
        shift = DEGREE_SEMITONES[degree] + bass_acc.count("#") - bass_acc.count("b")
        bass = (root + shift) % 12
    return root, quality or "maj", bass


def chord_name(chord: str, flats: bool) -> str:
    parsed = parse_harte(chord)
    if parsed is None:
        return "N.C."
    root, quality, bass = parsed
    roots = ROOTS_FLAT if flats else ROOTS_SHARP
    name = roots[root] + QUALITY_SUFFIX.get(quality, quality)
    return name + (f"/{roots[bass]}" if bass is not None and bass != root else "")


def simplify(chord: str) -> str:
    """Réduit à majeur/mineur sans basse (option --simple)."""
    parsed = parse_harte(chord)
    if parsed is None:
        return "N"
    root, quality, _ = parsed
    minor = quality.startswith(("min", "dim", "hdim"))
    return ROOTS_SHARP[root] + (":min" if minor else ":maj")


def merge_short(segs: list, min_dur: float) -> list:
    """Fusionne itérativement le segment le plus court avec son voisin le plus long."""
    segs = [list(s) for s in segs]
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
        segs = merge_equal(segs)
    return merge_equal(segs)


def merge_equal(segs: list) -> list:
    """Recolle les segments voisins portant le même accord."""
    merged = [segs[0]]
    for s in segs[1:]:
        if s[2] == merged[-1][2]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
FLAT_KEYS = {5, 10, 3, 8, 1, 6}  # F, Bb, Eb, Ab, Db, Gb majeurs (et leurs relatifs mineurs)


def estimate_key(durations: dict) -> int:
    """Tonalité majeure (0-11) dont la gamme couvre le plus de durée d'accords.

    Une tonalité mineure est représentée par son relatif majeur (Do mineur -> Mib).
    """

    def triad(root, quality):
        third = 3 if quality.startswith(("min", "dim", "hdim")) else 4
        fifth = 6 if quality in ("dim", "dim7", "hdim7") else 8 if quality == "aug" else 7
        if quality.startswith("sus"):
            third = 2 if quality == "sus2" else 5
        return {root, (root + third) % 12, (root + fifth) % 12}

    def score(key):
        scale = {(key + i) % 12 for i in MAJOR_SCALE}
        return sum(d * len(triad(*chord) & scale) for chord, d in durations.items())

    return max(range(12), key=score)


def key_name(key: int, flats: bool) -> str:
    roots = ROOTS_FLAT if flats else ROOTS_SHARP
    return f"{roots[key]} majeur / {roots[(key + 9) % 12]} mineur"


def fmt_time(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


# --- Pipeline ----------------------------------------------------------------


def prepare_audio(path: Path, separation: str | None, device: torch.device) -> dict:
    """Signal pour les accords (22.05 kHz) et, si séparé, voix isolée avec sa fréquence.

    separation : None (mix d'origine), "nodrums" (sans batterie) ou "instruments" (ni voix ni batterie).
    """
    if not separation:
        return {"chords": load_audio(path), "vocals": None}
    stems = separate(path, device)
    harmony = stems["accomp"] if separation == "instruments" else stems["accomp"] + stems["vocals"]
    harmony = resample(harmony, stems["sr"], SAMPLE_RATE)
    # La piste séparée est moins forte que le mix sur lequel BTC a été entraîné, qui la prendrait
    # en partie pour un silence ("N") : on la ramène au niveau du mix d'origine.
    mix = load_audio(path)
    harmony *= np.sqrt(np.mean(mix**2) / (np.mean(harmony**2) + 1e-12))
    return {"chords": harmony, "vocals": (stems["vocals"], stems["sr"])}


def analyze(recognizer, wav: np.ndarray, min_dur: float, spelling: str, simple: bool):
    """Segments [(début, fin, accord Harte, nom affiché)], tonalité majeure estimée, choix bémols."""
    segs = recognizer.recognize(wav)
    if simple:
        segs = merge_equal([[s, e, simplify(c)] for s, e, c in segs])
    segs = merge_short(segs, min_dur)
    segs[-1][1] = min(segs[-1][1], len(wav) / SAMPLE_RATE)

    durations = defaultdict(float)
    for start, end, chord in segs:
        parsed = parse_harte(chord)
        if parsed is not None:
            durations[parsed[:2]] += end - start
    key = estimate_key(durations)
    flats = key in FLAT_KEYS if spelling == "auto" else spelling == "flats"
    chords = [(start, end, chord, chord_name(chord, flats)) for start, end, chord in segs]
    return chords, key, flats


def add_chord_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        choices=["btc", "cnn"],
        default="btc",
        help="modèle d'accords : btc (170 accords) ou cnn (renversements, 9/11/13)",
    )
    parser.add_argument(
        "--separate",
        nargs="?",
        const="nodrums",
        choices=["nodrums", "instruments"],
        help="séparer les sources avec Demucs : sans batterie (défaut) ou instruments seuls "
        "(ni voix ni batterie) pour les accords ; voix isolée pour Whisper",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="accords majeurs/mineurs uniquement, sans renversement",
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
        "--device", default="auto", help="auto, cpu, mps ou cuda (BTC et Demucs)"
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
    recognizer = make_recognizer(args.model, args.simple, device)
    print(f"Modèle : {args.model} sur {device}", file=sys.stderr)

    for path in args.audio:
        print(f"\n=== {path.name} ===")
        audio = prepare_audio(path, args.separate, device)
        chords, key, flats = analyze(
            recognizer, audio["chords"], args.min_duration, args.spelling, args.simple
        )
        print(f"Tonalité estimée : {key_name(key, flats)}\n")

        lines, lab = [], []
        totals = defaultdict(float)
        for start, end, chord, name in chords:
            lines.append(f"{fmt_time(start)}  {name}")
            lab.append(f"{start:.3f} {end:.3f} {chord}")
            if chord != "N":
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
