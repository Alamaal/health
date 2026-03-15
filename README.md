# Sports AI ⚽

A Python toolkit for AI-powered sports video analysis. Currently focused on soccer,
with modules for player detection, tracking, team classification, and pitch
visualisation.

## 📦 Installation

**Python ≥ 3.8** is required. Install the core package directly from source:

```bash
pip install git+https://github.com/Alamaal/health.git
```

For the full soccer example (includes `torch` and `ultralytics`):

```bash
pip install "git+https://github.com/Alamaal/health.git#egg=sports[soccer]"
cd examples/soccer
pip install -r requirements.txt
./setup.sh   # downloads pre-trained model weights and a sample video
```

## 🗂️ Project Structure

```
sports/
  annotators/    – pitch and player drawing helpers
  common/        – shared utilities (ball tracker, team classifier, view transformer)
  configs/       – pitch configuration for each sport
examples/
  soccer/        – end-to-end soccer analysis pipeline (main.py)
tests/           – pytest test-suite
```

## 🚀 Quick Start – Soccer Analysis

```bash
python examples/soccer/main.py \
  --source_video_path examples/soccer/data/2e57b9_0.mp4 \
  --target_video_path output.mp4 \
  --device cpu \
  --mode RADAR
```

Available `--mode` values:

| Mode | Description |
|------|-------------|
| `PITCH_DETECTION` | Detect and overlay pitch key-points |
| `PLAYER_DETECTION` | Detect players, goalkeepers, referees and ball |
| `BALL_DETECTION` | Detect and track the ball |
| `PLAYER_TRACKING` | Track players across frames with consistent IDs |
| `TEAM_CLASSIFICATION` | Classify each player into one of two teams |
| `RADAR` | Bird's-eye radar overlay combining all of the above |

## 🤖 Models Used

| Model | Purpose |
|-------|---------|
| YOLOv8 (player) | Detect players, goalkeepers, referees, and ball |
| YOLOv8 (pitch) | Detect soccer-field boundaries and key-points |
| SigLIP | Extract visual features from player crops |
| UMAP + KMeans | Reduce and cluster features for team assignment |

## 🧪 Running Tests

```bash
pip install ".[tests]"
pytest tests/
```

## ⚽ Datasets

Training data originates from the
[DFL – Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout)
Kaggle competition. Processed datasets are available on
[Roboflow Universe](https://universe.roboflow.com/).

## © License

- **ultralytics** (YOLOv8): [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **sports toolkit**: [MIT](LICENSE)
