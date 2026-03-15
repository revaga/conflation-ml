"""
Run phase3 SLM labeling with Ollama, then with Hugging Face.
Requires:
  - Ollama: install a model first, e.g.  ollama pull llama3.2
  - Hugging Face: set HF_TOKEN or HUGGINGFACE_API_KEY in env or api_keys.env
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "slm_attribute_labeler.py"

def main():
    # 1) Ollama → data/phase3_slm_labeled_ollama.parquet
    print("Running phase3 with Ollama...")
    r1 = subprocess.run(
        [sys.executable, str(SCRIPT), "--provider", "ollama", "--output", "data/phase3_slm_labeled_ollama.parquet"],
        cwd=PROJECT_ROOT,
    )
    if r1.returncode != 0:
        print("Ollama run failed. Pull a model first: ollama pull llama3.2")
        return r1.returncode

    # 2) Hugging Face → data/phase3_slm_labeled_hf.parquet
    print("Running phase3 with Hugging Face...")
    r2 = subprocess.run(
        [sys.executable, str(SCRIPT), "--provider", "huggingface", "--output", "data/phase3_slm_labeled_hf.parquet"],
        cwd=PROJECT_ROOT,
    )
    if r2.returncode != 0:
        print("Hugging Face run failed. Set HF_TOKEN or HUGGINGFACE_API_KEY.")
        return r2.returncode

    print("Both runs completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
