
from typing import Literal
import os
import re
import cv2
import pandas as pd
import torch
from tqdm import tqdm


def file_mapping(filenames : list[str], data_dir : str) -> pd.DataFrame:
    #Dict of heart rate data for eyes closed, eyes open, and back facing to camera experiments.
    dict_ec = {
        "experiment_type" : [],
        "person_id" : [],
        "filename" : [],
        "filepath" : [],
    }
    
    dict_eo = {
        "experiment_type" : [],
        "person_id" : [],
        "filename" : [],
        "filepath" : [],
    }
    
    dict_back = {
        "experiment_type" : [],
        "person_id" : [],
        "filename" : [],
        "filepath" : [],
    }
    
    # Loop through the filenames and add them to the appropriate dictionary
    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        filepath = os.path.abspath(filepath)
        
        if "EC" in filename:
            dict_ec["experiment_type"].append("EC")
            dict_ec["person_id"].append(re.findall(r'\d+', filename)[0])
            dict_ec["filename"].append(filename)
            dict_ec["filepath"].append(filepath)
        
        elif "EO" in filename:
            dict_eo["experiment_type"].append("EO")
            dict_eo["person_id"].append(re.findall(r'\d+', filename)[0])
            dict_eo["filename"].append(filename)
            dict_eo["filepath"].append(filepath)
            
        elif "back" in filename:
            dict_back["experiment_type"].append("BACK")
            dict_back["person_id"].append(filename.split("_")[0])
            dict_back["filename"].append(filename)
            dict_back["filepath"].append(filepath)
        
        else:
            raise ValueError(f"Filename {filename} does not match any of the expected patterns.")
        
    # Create dataframes from the dictionaries
    df_ec = pd.DataFrame(dict_ec)
    df_eo = pd.DataFrame(dict_eo)
    df_back = pd.DataFrame(dict_back)
    
    #Allows for easy categorization of the data
    df = pd.concat([df_ec, df_eo, df_back], keys=["EC", "EO", "BACK"])
    return df

# Filter the data by experiment type categorization
def data_filter_type(experiment_type : Literal['EO', 'EC', 'BACK'], df : pd.DataFrame) -> pd.DataFrame:
    experiment_type = experiment_type.upper()
    return df.xs(experiment_type, level=0)
        

def process_video(filepath : str, grayscaled : bool = True) -> torch.tensor:
    cap = cv2.VideoCapture(filepath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if grayscaled:
        buf = torch.empty((frameCount, frameHeight, frameWidth), dtype = torch.uint8)
    else:
        buf = torch.empty((frameCount, frameHeight, frameWidth, 3), dtype = torch.uint8)

    frame_idx = 0
    while (frame_idx < frameCount):
        ret, frame = cap.read()
        if grayscaled:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            raise ValueError("Error reading frame, non-existing frame")
        buf[frame_idx] = torch.from_numpy(frame)
        frame_idx += 1

    cap.release()
    
    return buf

def process_videos(filepaths : list, tqdm_disabled : bool = True, grayscaled : bool = True) -> torch.tensor:
    shape = process_video(filepaths[0], grayscaled).shape
    n = len(filepaths)
    videos = torch.empty((n, *shape), dtype = torch.uint8)
    for idx, filepath in tqdm(enumerate(filepaths), disable=tqdm_disabled, total=n):
        v = process_video(filepath, grayscaled)
        videos[idx] = v
    return videos


    

