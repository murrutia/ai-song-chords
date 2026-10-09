# ai-song-chords

Transcription locale des accords d'une chanson à partir d'un fichier audio.

Deux scripts :

- `chords.py` : accords seuls, avec leurs horodatages.
- `sheet.py` : grille paroles + accords, chaque accord placé au-dessus du mot où il change.

Les accords viennent de [BTC](https://github.com/jayg996/BTC-ISMIR19) (Bi-directional Transformer for Chord recognition, ISMIR 2019, licence MIT) et ses poids pré-entraînés. Tout tourne en local, sur CPU, Apple MPS ou CUDA.

## Prérequis

- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` dans le PATH (décode m4a, mp3, wav, flac...)

## Installation

```sh
git clone --depth 1 https://github.com/jayg996/BTC-ISMIR19.git vendor/BTC-ISMIR19
uv sync
```

## Utilisation

```sh
uv run chords.py "songs/ma chanson.m4a"
uv run chords.py songs/*.m4a --out out/      # écrit aussi out/<titre>.lab et .txt
```

Options :

- `--simple` : vocabulaire majeur/mineur (25 classes) au lieu des 170 accords (7, maj7, m7, sus, dim...)
- `--min-duration 0.5` : absorbe les accords plus courts que cette durée en secondes
- `--spelling auto|sharps|flats` : altérations ; `auto` choisit selon la tonalité estimée
- `--device auto|cpu|mps|cuda`

Sortie : une ligne par changement d'accord (`m:ss.cc  Accord`), la tonalité estimée et les accords classés par durée totale. Le `.lab` utilise la notation Harte standard (`C:min`, `G:7`...), lisible par Sonic Visualiser ou mir_eval.

## Grille paroles + accords

```sh
uv run sheet.py "songs/ma chanson.m4a" --language en
```

Sorties dans `out/` (modifiable avec `--out`) :

- `<titre>.html` : page avec lecteur audio. Cliquer une ligne lance la lecture à cet endroit, et la ligne en cours est surlignée.
- `<titre>.sheet.txt` : accords au-dessus des paroles, à lire en police à chasse fixe.
- `<titre>.sheet.json` : mots horodatés et accords rattachés, pour d'autres traitements.

Les paroles sont transcrites par Whisper `large-v3-turbo`, téléchargé au premier lancement (~1,6 Go). Moteur : `mlx-whisper` sur Mac Apple Silicon, `faster-whisper` ailleurs (CUDA si disponible, sinon CPU). Options :

- `--language en` : langue des paroles ; sans elle, Whisper la détecte, parfois mal sur de la musique
- `--whisper-model small` : modèle plus léger et plus rapide, moins précis
- `--backend mlx|faster` : forcer le moteur
- plus les options d'accords de `chords.py` (`--simple`, `--min-duration`, `--spelling`, `--device`)

## Limites

- Pas de basse ni de renversement (`C/E` donne `C`).
- Timeline en secondes, pas calée sur les temps ou les mesures.
- Whisper peut se tromper de mots ou en inventer sur les passages instrumentaux ou très chargés. Isoler la voix au préalable (Demucs, par exemple) aiderait.
- Les accords sont rattachés au début des mots, jamais au milieu d'un mot.
- Modèle entraîné surtout sur de la pop/rock ; les arrangements denses ou le chant seul passent moins bien.
