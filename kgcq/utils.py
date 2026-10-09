import json
import os
import random

import pandas as pd


def set_global_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def file_to_string(filename):
    with open(filename, "r", errors="ignore") as file:
        return file.read()


def load_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


def load_csv(filename):
    return pd.read_csv(filename)


def save_to_json(data, output_file, indent=4):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def save_results(results, output_file):
    """Save results with keys sorted."""
    save_to_json(dict(sorted(results.items())), output_file)


def load_existing_results(output_file):
    """Load existing results if the file exists (used for resumable runs)."""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            print(f"Warning: Could not load existing results from {output_file}")
    return {}
