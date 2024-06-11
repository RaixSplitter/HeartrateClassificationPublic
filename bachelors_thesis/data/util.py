import rembg
import numpy as np
from tqdm import tqdm


"""
To pip install rembg, you need to navigate to your virtual environment folder > pyvenv.cfg 
and modify "include-system-site-packages = false" to "include-system-site-packages = true". 
Then you can "pip install rembg --user", which installs the package on the user level. 
Not the preferred way, but it works.
"""


def remove_bg_from_video(video: np.ndarray, tqdm_disabled: bool = True) -> np.ndarray:
    buf = np.empty(
        (video.shape[0], video.shape[1], video.shape[2], 3), np.dtype("uint8")
    )
    for frame_idx, frame in tqdm(
        enumerate(video), disable=tqdm_disabled, total=video.shape[0]
    ):
        buf[frame_idx] = rembg.remove(frame)[
            :, :, :3
        ]  # In order to avoid the binary mask, we only take the first 3 channels. Remove returns a 4 channel image.

    return buf
