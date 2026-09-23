from datasets import load_dataset

dataset = load_dataset(
    "leduckhai/MultiMed",
    "English",
    cache_dir="multimed",
)
