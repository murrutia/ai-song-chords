# ai-song-chords

Transcription locale des accords et des paroles d'une chanson à partir d'un fichier audio.

Deux scripts :

- `chords.py` : accords seuls, avec leurs horodatages.
- `sheet.py` : grille paroles + accords, chaque accord placé au-dessus du mot où il change.

Deux modèles d'accords pré-entraînés, au choix (`--model`) :

- `btc` (défaut) : [BTC](https://github.com/jayg996/BTC-ISMIR19), Bi-directional Transformer for Chord recognition (ISMIR 2019, licence MIT). 170 accords : maj, min, 7, maj7, m7, sus, dim...
- `cnn` : [Chord-CNN-LSTM](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) (ISMIR 2019, licence MIT). Ensemble de 5 réseaux + décodage HMM. Ajoute les renversements (`Bb/D`) et les accords enrichis (9, 11, 13, 7sus4).

Option `--separate` : [Demucs](https://github.com/adefossez/demucs) sépare d'abord les sources. Tout tourne en local, sur CPU, Apple MPS ou CUDA.

## Prérequis

- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` dans le PATH (décode m4a, mp3, wav, flac...)

## Installation

```sh
git clone --depth 1 https://github.com/jayg996/BTC-ISMIR19.git vendor/BTC-ISMIR19
git clone --depth 1 https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition.git vendor/chord-cnn-lstm
uv sync
```

## Utilisation

```sh
uv run chords.py "songs/ma chanson.m4a"
uv run chords.py songs/*.m4a --out out/      # écrit aussi out/<titre>.lab et .txt
```

Options :

- `--model btc|cnn` : modèle d'accords (voir plus haut)
- `--separate` : retire la batterie avec Demucs avant l'analyse des accords
- `--separate instruments` : retire batterie et voix, ne garde que basse et instruments. Utile quand les instruments portent l'harmonie ; nuisible quand ce sont les voix (chant a cappella, chœurs)
- `--simple` : accords majeurs/mineurs uniquement, sans renversement
- `--min-duration 0.5` : absorbe les accords plus courts que cette durée en secondes
- `--spelling auto|sharps|flats` : altérations ; `auto` choisit selon la tonalité estimée
- `--device auto|cpu|mps|cuda` : pour BTC et Demucs (CNN tourne sur CUDA si disponible, sinon CPU)

Les pistes séparées par Demucs (modèle `htdemucs`, téléchargé au premier lancement) sont mises en cache dans `.cache/stems/` : relancer avec un autre modèle d'accords ne refait pas la séparation.

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
- plus les options d'accords de `chords.py` (`--model`, `--separate`, `--simple`, `--min-duration`, `--spelling`, `--device`)

Avec `--separate`, Whisper reçoit la voix isolée au lieu du mix complet.

## Limites

- Renversements (`C/E`) seulement avec `--model cnn`.
- Timeline en secondes, pas calée sur les temps ou les mesures.
- Whisper peut se tromper de mots ou en inventer sur les passages instrumentaux ou très chargés. `--separate` peut aider.
- Les accords sont rattachés au début des mots, jamais au milieu d'un mot.
- Modèles entraînés surtout sur de la pop/rock ; les arrangements denses ou le chant seul passent moins bien.
