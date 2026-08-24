import os
from pathlib import Path

"""Dataset constants for CogBench v1-1."""

COR_LABELS = [
    "0_E",
    "1_STR",
    "2_LR",
    "3_CR",
    "4_CRR",
    "5_ER",
    "6_ERR",
    "7_NMER",
    "8_MSR",
]

IMAGE_EXT = "jpg"
HEATMAP_EXT = "npy"

IMAGE_NUM_START = 1
IMAGE_NUM_END = 251
NUM_IMAGES = IMAGE_NUM_END - IMAGE_NUM_START + 1
NUM_CORS = len(COR_LABELS)
TOTAL_COMBINATIONS = NUM_IMAGES * NUM_CORS

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


"""Project paths and compatibility switches."""
ROOT = Path(os.environ.get("GAZE_VLM_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()

# The checked-in source used the strict LLaVA-NeXT batching rule by default.
# Set this to 0 only in environments known to support other batch sizes.
STRICT_LLAVA_NEXT_BATCH = os.environ.get("GAZE_VLM_STRICT_LLAVA_NEXT_BATCH", "1") != "0"
