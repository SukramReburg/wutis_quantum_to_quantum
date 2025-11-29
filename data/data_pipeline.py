import subprocess
import sys
import os 
import yaml 
import shutil

SCRIPTS = [
    "data/fetch.py",
    "data/preprocess.py",
    "data/datasets.py",
]


def run_script(script):
    print(f"\n=== Running: {script} ===")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"❌ Error in {script}")
        sys.exit(result.returncode)
    else:
        print(f"✅ Finished: {script}")


if __name__ == "__main__": 
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = os.path.join(base_dir, paths['raw'])
    processed = os.path.join(base_dir, paths['processed'])

    try:
        shutil.rmtree(raw)
        print("\nRemoved raw data directory.")
    except OSError as error:
        pass
        print("Raw data directory not removed.")
    try:
        shutil.rmtree(processed)
        print("Removed processed data directory.")
    except OSError as error:
        pass
        print("Processed data directory not removed.")

    print("\n🚀 Starting Data Pipeline...\n")

    for script in SCRIPTS:
        run_script(script)

    print("\nFULL DATA PIPELINE COMPLETE\n")
