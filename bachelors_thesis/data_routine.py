import hydra
from bachelors_thesis.data import (
    generate_video_data_procedure,
    generate_file_mapping,
    hr_run,
    generate_matching_video_samples,
    generate_sample_subsets,
    generate_sample_pairs,
)
import logging
import mne


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg):
    DATA_DIR = cfg.paths.data_dir_path  # "../data/raw/video/"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"CFG: {cfg}")
    mne.set_log_level("ERROR")

    # region Generate File Mapping for Video Data

    generate_file_mapping(f"{DATA_DIR}raw/video", "EO", filepath=f"{DATA_DIR}mappings/video_EO.csv")
    generate_video_data_procedure(
        f"{DATA_DIR}mappings/video_EO.csv",
        f"{DATA_DIR}processed/video_EO.pt",
        singular=cfg.data.singular,
        overwrite=cfg.data.overwrite,
        tqdm_disabled=cfg.base.tqdm_disabled,
    )
    # endregion

    # region Generate File Mapping for HR Data
    generate_file_mapping(f"{DATA_DIR}raw/hr", "EO", f"{DATA_DIR}mappings/hr_EO.csv")
    hr_run(f"{DATA_DIR}mappings/hr_EO.csv", f"{DATA_DIR}processed/hr", cfg.data.duration)

    generate_matching_video_samples(
        f"{DATA_DIR}mappings/hr_sampling_mapping_duration{cfg.data.duration}.csv",
        f"{DATA_DIR}mappings/video_EO_torch.csv",
        f"{DATA_DIR}processed/videos_samples",
        tqdm_disabled=cfg.base.tqdm_disabled,
        overwrite=cfg.data.overwrite,
        duration=cfg.data.duration,
    )
    generate_sample_subsets(
        f"{DATA_DIR}mappings/sample_duration_{cfg.data.duration}_mapping.csv",
        f"{DATA_DIR}mappings",
        cfg.data.training_set_size,
        overwrite=cfg.data.overwrite,
    )
    generate_sample_pairs(
        f"{DATA_DIR}mappings/train_mapping.csv",
        f"{DATA_DIR}mappings/train_pairs.csv",
        overwrite=cfg.data.overwrite,
    )
    generate_sample_pairs(
        f"{DATA_DIR}mappings/test_mapping.csv",
        f"{DATA_DIR}mappings/test_pairs.csv",
        overwrite=cfg.data.overwrite,
    )
    generate_sample_pairs(
        f"{DATA_DIR}mappings/val_mapping.csv",
        f"{DATA_DIR}mappings/val_pairs.csv",
        overwrite=cfg.data.overwrite,
    )

    # endregion


if __name__ == "__main__":
    main()
