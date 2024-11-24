import json
import warnings

# Suppress warnings (optional)
warnings.filterwarnings("ignore")


# -------------------------------
# Utility Functions
# -------------------------------


with open('../../commons/config.json', 'r') as config_file:
    config = json.load(config_file)


