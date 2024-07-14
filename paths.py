from pathlib import Path
import os

DATA_DIR =  'data'
MODELS_DIR = 'models'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)