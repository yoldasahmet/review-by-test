"""Train against the exact model revision used for evaluation."""
import subprocess
import sys
from generate import model_path

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "-c", "train.yaml",
                    "--model", model_path()], check=True)
