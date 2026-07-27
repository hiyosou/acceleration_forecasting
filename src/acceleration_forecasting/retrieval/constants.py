from pathlib import Path


ACCELERATION_COLUMN = "acc_z[m/s2]"
VELOCITY_COLUMN = "velocity[km/h]"
DISTANCE_COLUMN = "distance_corrected[m]"

DEFAULT_WAVEFORM_DIR = Path(r"D:\railwaydata\方向別振動データ_位置補正_0.2m")
DEFAULT_TREND_DIR = Path(
    r"D:\railwaydata\方向別振動データ_位置補正_0.2m_100m最大値_datasets"
)
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "retrieval"

START_M = 2000.0
END_M = 33000.0
BIN_WIDTH_M = 100.0
STEP_M = 0.2
SAMPLES_PER_BIN = 500

MIN_MEAN_SPEED_KMH = 50.0
MAX_MEAN_SPEED_KMH = 75.0
FUTURE_MONTHS = 18
SEARCH_DAYS = 15
EMBEDDING_DIM = 256
DATABASE_RATIO = 0.8
RANDOM_SEED = 42
