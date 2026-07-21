class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}" 
    
###### Fill the following paths with the appropriate directories on your system ######
RESULTS_DATA_HOME = ""

DATA_DIR = f"{RESULTS_DATA_HOME}/data/unstructured/"
BASLELINE_RESULTS_DIR = "{data_home}/results/analysis/{seed}/baseline_models/".format_map(SafeDict(data_home=RESULTS_DATA_HOME))
RESULTS_DIR = f"{RESULTS_DATA_HOME}/results/"
CACHE_DIR = f"{RESULTS_DATA_HOME}/.cache/"