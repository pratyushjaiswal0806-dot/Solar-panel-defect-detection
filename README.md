# Solar Panel Defect Detection

A YOLOv8-based defect detection system for solar panels. The model identifies and classifies various conditions on solar panel surfaces including bird droppings, dust, electrical damage, and snow coverage.

## Project Structure

```
├── models/          # Trained YOLO model weights (.pt)
├── data/
│   └── test/        # Test images for inference
├── src/
│   └── live.py      # Interactive image viewer with inference
├── .gitignore
├── requirements.txt
└── README.md
```

## Detected Classes

| Class             | Description                        |
|-------------------|------------------------------------|
| Bird Drop         | Bird droppings on panel surface    |
| Clean             | Panel in good condition            |
| Dusty             | Dust accumulation on panel         |
| Electrical Damage | Electrical faults or burn marks    |
| Snow              | Snow coverage on panel             |

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your trained model weights (`best.pt`) in the `models/` directory.
2. Add test images to `data/test/`.
3. Run the interactive viewer:

```bash
python src/live.py
```

### Controls

| Key | Action         |
|-----|----------------|
| `D` | Next image     |
| `A` | Previous image |
| `Q` | Quit           |

## Configuration

Edit the config section in `src/live.py`:

- **CONF** — Confidence threshold (default: `0.50`)
- **SCALE** — Display scale factor (default: `1.8`)
