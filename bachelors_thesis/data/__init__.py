import itertools
from matplotlib import pyplot as plt
import mne
import numpy as np
from sklearn.model_selection import train_test_split
import sleepecg
from tqdm.auto import tqdm
from bachelors_thesis.data.make_dataset import process_videos, file_mapping, data_filter_type
import torch
import logging
import os
from typing import Literal
import pandas as pd

def generate_file_mapping(data_dir : str, experiment_type : Literal['EO', 'EC', 'BACK'], filepath : str, overwrite : bool = False):
    if not overwrite and os.path.exists(filepath):
        logging.debug(f"File mapping already exists at {filepath}. Skipping.")
        return
                
    filenames = os.listdir(data_dir)
    df = file_mapping(filenames, data_dir)
    df = data_filter_type(experiment_type, df)
    if os.path.exists(filepath):
        logging.debug(f"File mapping already exists at {filepath}. Overwriting.")
    df.to_csv(filepath, index = False, encoding='utf-8')
    logging.debug(f"File mapping saved to {filepath}")
    
def get_video_data(mapping_path, tqdm_disabled : bool = True):
    df = pd.read_csv(mapping_path)
    videos = process_videos(df["filepath"], tqdm_disabled=tqdm_disabled)
    return df, videos

def generate_video_data_procedure(mapping_path, filepath : str, singular : bool = False, overwrite : bool = False, tqdm_disabled : bool = True):
    # Get Data
    if not overwrite and os.path.exists(filepath):
        logging.debug(f"Data file already exists at {filepath}. Skipping.")
        df_map = pd.read_csv(mapping_path, encoding='utf-8')
        videos = torch.load(filepath)
    else:
        df_map, videos = get_video_data(mapping_path, tqdm_disabled=tqdm_disabled)
        
        if os.path.exists(filepath):
            logging.debug(f"Data file already exists at {filepath}. Overwriting.")
            torch.save(videos, filepath)
            logging.debug(f"Data file saved to {filepath}")

            
    if not singular:
        return
    
    video_mapping = df_map.copy()
    
    dirpath = os.path.join(os.path.dirname(filepath), "videos")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    pbar = tqdm(range(videos.shape[0]), disable=tqdm_disabled)
    # pbar = tqdm(enumerate(zip(videos, df_map['person_id'])), total=videos.shape[0])
    for idx in pbar:
        video = videos[idx].clone()
        person_id = df_map['person_id'][idx]
        
        logging.debug(f"Saving video {idx} for person {person_id} with shape {video.shape}")
        
        videoname = f"video_EO_{person_id}.pt"
        videopath = os.path.join(dirpath, videoname)
        
        if not overwrite and os.path.exists(videopath):
            pbar.set_description(f"File exists: Skipping video {idx} for person {person_id}")
            logging.debug(f"Video file already exists at {videopath}. Overwriting.")
        else:
            pbar.set_description(f"Saving video {idx} for person {person_id}")
            torch.save(video, videopath)
            
        video_mapping.loc[idx, "filename"] = videoname
        video_mapping.loc[idx, "filepath"] = os.path.abspath(videopath)
    
    path_root, ext = os.path.splitext(mapping_path)
    
    video_map_path = f"{path_root}_torch{ext}"
    video_mapping.to_csv(video_map_path, index=False, encoding='utf-8')

    logging.debug(f"Video mapping saved to {video_map_path}")
    
    logging.debug(f"Data files saved to {dirpath}")  
    

    


def hr_extract_channels(raw : mne.io.Raw, start, end):
    channels = raw.ch_names
    if set(["EXG1", "EXG2", "EXG3"]).issubset(set(channels)):
        return raw[["EXG1", "EXG2", "EXG3"], start:end]
    elif set(["EXG1-0", "EXG2-0", "EXG3-0"]).issubset(set(channels)):
        return raw[["EXG1-0", "EXG2-0", "EXG3-0"], start:end]
    else:
        logging.error("No channels found")
        return None

def hr_extract_experiment(raw):
    events = mne.find_events(raw, initial_event=False)  
    events[events[:, 0].argsort()] # Sort events by time
    start = events[events[:, 2] == 2][0][0]
    end = events[events[:, 2] == 8][0][0]
    return start, end

def hr_extract_samples(experiment : tuple[list[np.array], np.array], duration : float = 5.0, frequency : int = 2048):
    data, times = experiment
    samples = []
    for i in range(0, len(data[0]) - int(frequency * duration), int(frequency * duration)):
        sample = [ch[i:i + int(frequency * duration)] for ch in data]
        sample = np.array(sample)
        samples.append((sample, times[i:i + int(frequency * duration)]))
    return samples
    

def plot_hr_channels(sample):
    data, times = sample
    for ch in data:
        plt.plot(times[:2048*5], ch[:2048*5])
    plt.legend(["EXG1", "EXG2", "EXG3"])
    plt.show()    
    
def extract_heartbeat(sample, channel : int = 2, fs = 2048):
    return sleepecg.detect_heartbeats(sample[channel], fs=fs)

def plot_heartbeats(sample, beats, fs = 2048):
    plt.plot([i / fs for i in range(len(sample[2]))],sample[2])
    plt.scatter(beats / fs, sample[2][beats], color='red')
    plt.show()
    
def extract_rr_intervals(beats, fs = 2048):
    return np.diff(beats) / fs * 1000
    

    


def hr_run(df_path : str, outdir : str, duration, overwrite : bool = False):
    df = pd.read_csv(df_path)
    
    outdir = os.path.join(outdir, f"duration{duration}")
    outdir = os.path.abspath(outdir)
    
    os.makedirs(outdir, exist_ok=True)
    
    sample_mapping = pd.DataFrame(columns=["experiment_type", "person_id", "filename", "sample_id", "sample_idx", "data_path", "times_path", "rr_interval_path1", "rr_interval_path2", "rr_interval_path3", "starttime", "endtime"])
    
    for idx, row in df.iterrows():
        filename, _ = os.path.splitext(row["filename"])
        filepath = row["filepath"]
        raw = mne.io.read_raw_bdf(filepath, preload=False, verbose='ERROR')
        
        start, end = hr_extract_experiment(raw)
        
        experiment = hr_extract_channels(raw, start, end)
        
        samples = hr_extract_samples(experiment, duration=duration)
        
        sampledir = os.path.join(outdir, filename)
        
        os.makedirs(sampledir, exist_ok=True)
        
        for sample_idx, sample in enumerate(samples):
            data, times = sample
            
            # Extract RR intervals for each channel
            
            rr_intervals = [extract_rr_intervals(extract_heartbeat(data, channel=i)) for i in range(3)]
            

            
                        
            datadir = os.path.join(sampledir, "data")
            timesdir = os.path.join(sampledir, "times")
            rr_interval_dirs = [os.path.join(sampledir, f"rr_intervals/{i}") for i in range(3)]

            os.makedirs(datadir, exist_ok=True)
                
            os.makedirs(timesdir, exist_ok=True)
            
            for path in rr_interval_dirs:
                os.makedirs(path, exist_ok=True)
            
            sample_name = f"{filename}-duration-{duration}-sample-{sample_idx}"
            
            
            data_path = os.path.join(datadir, f"{sample_name}.npy")
            times_path = os.path.join(timesdir, f"{sample_name}.npy")
            
            rr_interval_paths = [os.path.join(rr_interval_dirs[i], f"{sample_name}.npy") for i in range(3)]
            
            starttime = times[0] 
            endtime = times[-1] 
            
            sample_mapping.loc[len(sample_mapping)] = [row["experiment_type"], row["person_id"], row["filename"], sample_name, sample_idx, data_path, times_path, rr_interval_paths[0], rr_interval_paths[1], rr_interval_paths[2], starttime, endtime]
            
            for rr_interval_path, rr_intervals in zip(rr_interval_paths, rr_intervals):
                if os.path.exists(rr_interval_path) and not overwrite:
                    logging.debug(f"Existing RR interval found, skipping: {sample_name}")
                    continue
                np.save(rr_interval_path, rr_intervals)
            
            if os.path.exists(data_path) and os.path.exists(times_path) and not overwrite:
                logging.debug(f"Existing sample found, skipping {sample_name}")
                
                continue
            
            np.save(data_path, data)
            np.save(times_path, times)
            
    
    mapping_dir = os.path.dirname(df_path)
    sample_mapping_path = os.path.join(mapping_dir, f"hr_sampling_mapping_duration{duration}.csv")
    
    sample_mapping.to_csv(sample_mapping_path, index = False, encoding='utf-8')
        
        
def generate_matching_video_samples(hr_df_path, video_df_path, outdir : str, duration : float = 5.0, fps : int = 30, overwrite : bool = False, tqdm_disabled : bool = True):
    hr_df = pd.read_csv(hr_df_path)
    video_df = pd.read_csv(video_df_path)

    outdir = os.path.join(outdir, f"duration_{duration}")
    outdir = os.path.abspath(outdir)
    
    pbar = tqdm(hr_df.iterrows(), total=len(hr_df), disable=tqdm_disabled)
    
    columns = list(hr_df.columns) + ['video_path']
    
    mapping = pd.DataFrame(columns=columns)
    
    for idx, hr_metadata in pbar:
        pbar.set_description(f"Processing {hr_metadata['filename']}")
        
        samplename, ext = os.path.splitext(hr_metadata['data_path'])
        samplename = os.path.basename(samplename)
        filename, ext = os.path.splitext(hr_metadata['filename'])
        sampledir = os.path.join(outdir, filename)
        video_sample_path = f"{sampledir}/{samplename}.pt"
        
        if os.path.exists(video_sample_path) and not overwrite:
            mapping.loc[len(mapping)] = hr_metadata.tolist() + [video_sample_path]
            logging.debug(f"Existing video sample found, skipping {video_sample_path}")
            continue
        
        video_slice, video_metadata = find_video_slice(hr_metadata, video_df, duration, fps)
        if video_slice is None:
            continue
        
        os.makedirs(sampledir, exist_ok=True)
            
        mapping.loc[len(mapping)] = hr_metadata.tolist() + [video_sample_path]

        
        torch.save(video_slice, video_sample_path)
    mapping_path = f"{os.path.dirname(hr_df_path)}/sample_duration_{duration}_mapping.csv"
    mapping.to_csv(mapping_path, index = False, encoding='utf-8')


def find_video_slice(hr_metadata, video_df, duration : float = 5.0, fps : int = 30):
    st = hr_metadata['sample_idx'] * duration
    p_id = hr_metadata['person_id']

    # find the video file that corresponds to the hr file
    video_metadata = video_df[video_df['person_id'] == p_id].iloc[0]
    video_file = torch.load(video_metadata['filepath'])

    # Calculate the time in the video that corresponds to the start and end time of the hr file
    video_st = int(st * fps) # 30 fps
    video_et = int(video_st + duration * fps)
    
    if video_et <= len(video_file):
        video_slice = video_file[video_st:video_et].clone()
    else:
        logging.warning("Video slice out of bounds, padding with zeros")
        logging.warning(f"Video shape: {video_slice.shape}, Start: {video_st}, End: {video_et}")
        return None, None
        
    return video_slice, video_metadata

def generate_sample_subsets(df_path, outdir: str, train_size: float = 0.6, overwrite: bool = False, person_sample_amount : int = 0):
    df = pd.read_csv(df_path)
    outdir = os.path.abspath(outdir)

    person_ids = df["person_id"].unique()
    logging.info(f"Person ids: len({person_ids if len(person_ids) < 20 else len(person_ids)})")
    
    train_ids, rem_ids = train_test_split(person_ids, train_size=train_size, random_state=0)

    test_ids, val_ids = train_test_split(rem_ids, test_size=0.5, random_state=0)

    # print(f"Train: {len(train_ids)}")
    # print(f"Test: {len(test_ids)}")
    # print(f"Val: {len(val_ids)}")
    
    
    # Only allow x samples per person
    if person_sample_amount > 0:
        train_df = pd.DataFrame()
        for person_id in train_ids:
            person_samples = df[df["person_id"] == person_id]
            person_samples = person_samples.sample(n=person_sample_amount, random_state=0)
            train_df = pd.concat([train_df, person_samples])
    else:
        train_df = df[df["person_id"].isin(train_ids)]
        
    test_df = df[df["person_id"].isin(test_ids)]
    val_df = df[df["person_id"].isin(val_ids)]
    
    
    if person_sample_amount > 0:
        train_df_path = os.path.join(outdir, f"train_mapping_subsample_{person_sample_amount}.csv")
    else:
        train_df_path = os.path.join(outdir, "train_mapping.csv")
    test_df_path = os.path.join(outdir, "test_mapping.csv")
    val_df_path = os.path.join(outdir, "val_mapping.csv")

    train_df.to_csv(
        train_df_path, index=False, encoding="utf-8"
    )
    test_df.to_csv(
        test_df_path, index=False, encoding="utf-8"
    )
    val_df.to_csv(
        val_df_path, index=False, encoding="utf-8"
    )


def generate_sample_pairs(df_path, outpath: str, overwrite: bool = False):
    df = pd.read_csv(df_path)
    outpath = os.path.abspath(outpath)

    combinations = list(itertools.combinations([row for _, row in df.iterrows()], 2))

    mapping = pd.DataFrame(
        columns=[
            "sample_id1",
            "sample_id2",
            "video_path1",
            "video_path2",
            "hr_path1",
            "hr_path2",
            "rr_interval1_path1",
            "rr_interval2_path1",
            "rr_interval3_path1",
            "rr_interval1_path2",
            "rr_interval2_path2",
            "rr_interval3_path2",
            "times_path1",
            "times_path2",
        ]
    )

    for c in combinations:
        if c[0]["person_id"] != c[1]["person_id"]:
            mapping.loc[len(mapping)] = [
                c[0]["sample_id"], #"sample_id1",
                c[1]["sample_id"], #"sample_id2",
                c[0]["video_path"], #"video_path1",
                c[1]["video_path"], #"video_path2",
                c[0]["data_path"], #"hr_path1",
                c[1]["data_path"], #"hr_path2",
                c[0]["rr_interval_path1"], #"rr_interval1_path1",
                c[0]["rr_interval_path2"],  #"rr_interval2_path1",
                c[0]["rr_interval_path3"], #"rr_interval3_path1",
                c[1]["rr_interval_path1"], #"rr_interval1_path2",
                c[1]["rr_interval_path2"], #"rr_interval2_path2",
                c[1]["rr_interval_path3"], #"rr_interval3_path2",
                c[0]["times_path"], #"times_path1",
                c[1]["times_path"] #"times_path2",
            ]
            

    mapping.to_csv(outpath, index=False, encoding="utf-8")