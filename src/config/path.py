from pathlib import Path

#project root path
PROJECT_DIR = Path(__file__).resolve().parents[2]

#dataset path
DATA_DIR = PROJECT_DIR / 'Data' / 'Raw'

#artifacts path
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'

#reports path
REPORTS_DIR = ARTIFACTS_DIR / 'reports'

#charts path
CHART_DIR = ARTIFACTS_DIR / 'charts'

MODEL_DIR = ARTIFACTS_DIR / 'models'

# print("Project directory: ", PROJECT_DIR)
# print("Data directory: ", DATA_DIR)
# print("Artifacts directory: ", ARTIFACTS_DIR)
# print("Reports directory: ", REPORTS_DIR)
# print("Chart directory: ", CHART_DIR)