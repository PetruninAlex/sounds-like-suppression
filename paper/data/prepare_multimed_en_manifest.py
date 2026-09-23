from manifest_common import export_dataset


def main():
    export_dataset(
        dataset_name="leduckhai/MultiMed",
        config="English",
        cache_dir="multimed",
        out_dir="multimed_nemo",
        log_every=100,
    )


if __name__ == "__main__":
    main()
