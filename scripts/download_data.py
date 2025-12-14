# Downloads FNSPID from Hugging Face and updates config.yaml to point to it
from huggingface_hub import snapshot_download
from pathlib import Path
import yaml

def main():
    print("Downloading FNSPID dataset from Hugging Face...")
    
    # Download dataset to local cache
    local_path = Path(snapshot_download(
        repo_id="Zihan1004/FNSPID",
        repo_type="dataset"
    ))
    
    print(f"Dataset downloaded to: {local_path}")
    
    # Update raw_dir in config.yaml to point to the downloaded location
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg["data"]["raw_dir"] = str(local_path)
    with open("config.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated config.yaml - raw_dir now points to: {local_path}")
    print("Ready to run the next scripts in the pipeline!")

if __name__ == "__main__":
    main()
