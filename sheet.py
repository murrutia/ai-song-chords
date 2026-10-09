#!/usr/bin/env python3
"""Génère une grille paroles + accords à partir d'un fichier audio.

Les paroles sont transcrites par Whisper (horodatage mot à mot), les accords
par BTC (voir chords.py). Chaque changement d'accord est placé au-dessus du
mot chanté à ce moment-là.

Usage :
    uv run sheet.py "songs/ma chanson.m4a"
    uv run sheet.py songs/*.m4a --language en --out out/
"""

import argparse
import html
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

from chords import (
    add_chord_args,
    analyze,
    key_name,
    load_audio,
    make_recognizer,
    pick_device,
    prepare_audio,
    resample,
)

WHISPER_SR = 16000
LINE_GAP = 0.8  # silence (s) au-delà duquel on coupe une ligne de paroles
INSTRUMENTAL_GAP = (
    2.0  # silence (s) au-delà duquel les accords forment une ligne à part
)
MAX_LINE_WORDS = 10  # au-delà, une ligne est recoupée


@dataclass
class Word:
    text: str
    start: float
    end: float
    chords: list[str] = field(default_factory=list)


@dataclass
class Line:
    words: list[Word]  # vide pour une ligne instrumentale
    start: float
    end: float
    chords: list[str] = field(default_factory=list)  # accords d'une ligne instrumentale


def default_backend() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "faster"


def transcribe(
    audio, backend: str, model: str, language: str | None
) -> list[list[Word]]:
    """Segments Whisper sous forme de listes de mots horodatés."""
    # condition_on_previous_text=False limite les boucles de répétition fréquentes sur la musique.
    if backend == "mlx":
        import mlx_whisper

        repo = model if "/" in model else f"mlx-community/whisper-{model}"
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=repo,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        segments = [
            (
                s["no_speech_prob"],
                s["avg_logprob"],
                [
                    Word(w["word"].strip(), w["start"], w["end"])
                    for w in s.get("words", [])
                ],
            )
            for s in result["segments"]
        ]
    else:
        import ctranslate2
        from faster_whisper import WhisperModel

        cuda = ctranslate2.get_cuda_device_count() > 0
        whisper = WhisperModel(
            model,
            device="cuda" if cuda else "cpu",
            compute_type="float16" if cuda else "int8",
        )
        segs, _ = whisper.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        segments = [
            (
                s.no_speech_prob,
                s.avg_logprob,
                [Word(w.word.strip(), w.start, w.end) for w in (s.words or [])],
            )
            for s in segs
        ]
    # Écarte les segments que Whisper lui-même juge douteux (hallucinations sur l'instrumental).
    return [
        words
        for no_speech, logprob, words in segments
        if words and not (no_speech > 0.6 and logprob < -1.0)
    ]


def split_lines(segments: list[list[Word]]) -> list[Line]:
    """Découpe les segments Whisper en lignes aux pauses de chant."""
    lines = []
    for words in segments:
        current = [words[0]]
        for prev, word in zip(words, words[1:]):
            if word.start - prev.end > LINE_GAP:
                lines.append(current)
                current = []
            current.append(word)
        lines.append(current)
    lines = [part for words in lines if words for part in split_long(words)]
    return [Line(words, words[0].start, words[-1].end) for words in lines]


def split_long(words: list[Word]) -> list[list[Word]]:
    """Recoupe récursivement une ligne trop longue, vers le milieu, sur la ponctuation ou la plus longue pause."""
    if len(words) <= MAX_LINE_WORDS:
        return [words]
    n = len(words)

    def score(i):  # coupure entre words[i-1] et words[i]
        punct = 1.0 if words[i - 1].text[-1:] in ".,;:!?" else 0.0
        gap = words[i].start - words[i - 1].end
        centered = 1 - abs(i / n - 0.5)  # favorise les moitiés équilibrées
        return punct + 2 * gap + centered

    cut = max(range(max(2, n * 3 // 10), min(n - 1, n * 7 // 10) + 1), key=score)
    return split_long(words[:cut]) + split_long(words[cut:])


def place_chords(lines: list[Line], chords) -> list[Line]:
    """Rattache chaque changement d'accord à un mot, ou à une ligne instrumentale."""
    chords = [
        (start, end, name) for start, end, chord, name in chords if chord != "N"
    ]
    result = []
    prev_end = 0.0
    shown = None  # dernier accord affiché
    i = 0
    for line in lines:
        # Accords qui commencent dans le silence avant la ligne
        gap = []
        while i < len(chords) and chords[i][0] < line.start:
            gap.append(chords[i])
            i += 1
        if gap and line.start - prev_end >= INSTRUMENTAL_GAP:
            names = [name for start, _, name in gap if start < line.start - 0.5]
            if names:
                result.append(Line([], gap[0][0], line.start, names))
                shown = names[-1]
        # Accord en cours au début de la ligne : affiché sauf s'il est le dernier déjà écrit
        active = gap[-1][2] if gap else shown
        if active and active != shown:
            line.words[0].chords.append(active)
            shown = active
        # Changements pendant la ligne : sur le premier mot encore en train de sonner
        while i < len(chords) and chords[i][0] < line.end:
            start, _, name = chords[i]
            word = next((w for w in line.words if w.end > start + 0.1), line.words[-1])
            if name != shown:
                if word is line.words[0] and word.chords == [shown]:
                    word.chords = []
                word.chords.append(name)
                shown = name
            i += 1
        result.append(line)
        prev_end = line.end
    # Accords après la dernière ligne (outro)
    outro = [name for start, _, name in chords[i:]]
    if outro:
        result.append(Line([], chords[i][0], chords[-1][1], outro))
    return result


def render_text(lines: list[Line]) -> str:
    """Deux lignes par ligne de paroles : accords alignés au-dessus des mots (police à chasse fixe)."""
    out = []
    for line in lines:
        if not line.words:
            out.append("\n" + "   ".join(line.chords) + "\n")
            continue
        chord_row, lyric_row = "", ""
        for word in line.words:
            if word.chords:
                if chord_row and len(lyric_row) < len(chord_row) + 1:
                    lyric_row = lyric_row.ljust(len(chord_row) + 1)
                chord_row = chord_row.ljust(len(lyric_row)) + " ".join(word.chords)
            lyric_row += word.text + " "
        if chord_row:
            out.append(chord_row.rstrip())
        out.append(lyric_row.rstrip())
    return "\n".join(out).strip() + "\n"


def render_html(lines: list[Line], title: str, key: str, audio_src: str | None) -> str:
    esc = html.escape
    body = []
    for line in lines:
        attrs = f'data-start="{line.start:.2f}" data-end="{line.end:.2f}"'
        if not line.words:
            chips = "".join(f'<span class="c">{esc(c)}</span>' for c in line.chords)
            body.append(f'<div class="line inst" {attrs}>{chips}</div>')
            continue
        words = "".join(
            f'<span class="w"><span class="c">{esc(" ".join(w.chords)) or "&nbsp;"}</span>'
            f'<span class="t">{esc(w.text)}</span></span>'
            for w in line.words
        )
        body.append(f'<div class="line" {attrs}>{words}</div>')
    audio = (
        f'<audio id="player" controls src="{esc(audio_src)}"></audio>'
        if audio_src
        else ""
    )
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  :root {{ --bg: #fdfcf9; --fg: #1d1d1f; --muted: #6b6b70; --chord: #b4441c; --hl: #fff1c9; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #17171a; --fg: #ececee; --muted: #9a9aa2; --chord: #ff9d6c; --hl: #3a3320; }}
  }}
  body {{ background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 52rem;
         padding: 1.5rem 1rem 6rem; font: 18px/1.4 system-ui, sans-serif; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; }}
  .meta {{ color: var(--muted); margin-bottom: 1.5rem; }}
  .line {{ display: flex; flex-wrap: wrap; column-gap: .3em; padding: .15rem .4rem;
          margin: 0 -.4rem .35rem; border-radius: 6px; cursor: pointer; }}
  .line.inst {{ margin: 1rem -.4rem; column-gap: 1.2em; }}
  .line.on {{ background: var(--hl); }}
  .w {{ display: inline-flex; flex-direction: column; }}
  .c {{ color: var(--chord); font-weight: 700; font-size: .85em; min-height: 1.2em; padding-right: .4em;
       white-space: nowrap; }}
  audio {{ position: fixed; bottom: 1rem; left: 50%; transform: translateX(-50%);
          width: min(40rem, calc(100% - 2rem)); }}
</style>
</head>
<body>
<h1>{esc(title)}</h1>
<div class="meta">Tonalité estimée : {esc(key)}</div>
{chr(10).join(body)}
{audio}
<script>
  const player = document.getElementById("player");
  const lines = [...document.querySelectorAll(".line")];
  if (player) {{
    lines.forEach(l => l.addEventListener("click", () => {{
      player.currentTime = +l.dataset.start; player.play();
    }}));
    player.addEventListener("timeupdate", () => {{
      const t = player.currentTime;
      lines.forEach(l => l.classList.toggle("on", t >= +l.dataset.start && t < +l.dataset.end));
    }});
  }}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("audio", nargs="+", type=Path, help="fichier(s) audio")
    parser.add_argument(
        "--out", type=Path, default=Path("out"), help="dossier de sortie (défaut out/)"
    )
    parser.add_argument(
        "--language", help="langue des paroles (en, fr...) ; détectée si absente"
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3-turbo",
        help="modèle Whisper (défaut large-v3-turbo ; small, medium... plus rapides)",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "faster"],
        default=default_backend(),
        help="moteur Whisper : mlx (Mac Apple Silicon) ou faster (faster-whisper)",
    )
    add_chord_args(parser)
    args = parser.parse_args()

    device = pick_device(args.device)
    recognizer = make_recognizer(args.model, args.simple, device)
    args.out.mkdir(parents=True, exist_ok=True)

    for path in args.audio:
        print(f"=== {path.name}", file=sys.stderr)
        if args.separate:
            print("  séparation des sources (Demucs)...", file=sys.stderr)
        audio = prepare_audio(path, args.separate, device)
        print(f"  accords ({args.model})...", file=sys.stderr)
        chords, key, flats = analyze(
            recognizer, audio["chords"], args.min_duration, args.spelling, args.simple
        )
        print(f"  paroles ({args.backend}, {args.whisper_model})...", file=sys.stderr)
        if audio["vocals"] is not None:
            vocals = resample(*audio["vocals"], WHISPER_SR)
        else:
            vocals = load_audio(path, WHISPER_SR)
        segments = transcribe(
            vocals,
            args.backend,
            args.whisper_model,
            args.language,
        )
        lines = place_chords(split_lines(segments), chords)

        title = path.stem
        key_label = key_name(key, flats)
        txt = args.out / f"{path.stem}.sheet.txt"
        txt.write_text(
            f"{title}\nTonalité estimée : {key_label}\n\n" + render_text(lines)
        )
        page = args.out / f"{path.stem}.html"
        audio_src = os.path.relpath(path.resolve(), args.out.resolve())
        page.write_text(render_html(lines, title, key_label, audio_src))
        (args.out / f"{path.stem}.sheet.json").write_text(
            json.dumps(
                [
                    {
                        "start": l.start,
                        "end": l.end,
                        "chords": l.chords,
                        "words": [vars(w) for w in l.words],
                    }
                    for l in lines
                ],
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"  -> {page}\n  -> {txt}", file=sys.stderr)


if __name__ == "__main__":
    main()
