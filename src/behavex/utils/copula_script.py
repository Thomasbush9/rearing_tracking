import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from pathlib import Path 
from tqdm import tqdm
from scipy.stats import rankdata
import copulas 
import re
from  argparse import ArgumentParser



FEATURE_SET = [
    "height", "nose_tail_distance", "facing_angle",
    "head_speed", "rel_bearing", "radial_vel", "trunk_speed","dist_scaled", "dist", "speed_head_fwd",
    "head_acc",
]
ANGULAR_FEATURES = ["facing_angle"]  # keep as-is (used inside your functions)

def to_copula(x):
    x = np.asarray(x)
    return rankdata(x, method="average") / (len(x) + 1.0)

def angle_to_linear(theta, cut=-np.pi):
    twopi = 2 * np.pi
    return (theta - cut) % twopi  # in [0, 2π)
# functions to load sessions 
#
def match_session_filename(filename, m=None, s=None, suffix="cricket"):
    """
    Checks if the filename matches the pattern 'm{m}_s{s}_{suffix}.xlsx'.
    If m, s, or suffix are None, allows any digits/word for that part.

    Args:
        filename (str): The filename to check (not full path).
        m (int or str, optional): The mouse/session number (int or None).
        s (int or str, optional): The session number (int or None).
        suffix (str, optional): Suffix (e.g. "cricket") before extension.

    Returns:
        bool: True if match, False otherwise.
    """
    mstr = rf"{int(m):03d}" if m is not None else r"\d{3}"
    sstr = rf"{int(s):03d}" if s is not None else r"\d{3}"
    suf = re.escape(suffix) if suffix is not None else r"\w+"
    pattern = rf"m{mstr}_s{sstr}_{suf}\.xlsx"
    return bool(re.fullmatch(pattern, filename))



def load_session(session_path:Path)-> pd.DataFrame:
    assert session_path.suffix == ".xlsx", "File needs to be in excel format"
    data = pd.read_excel(session_path)
    assert "dist_head" in data.columns, "Columns of dataset incomplete"

    # cut nan cols: 
    idx = data["dist_head"].first_valid_index()
    data = data.iloc[idx:]
    return data 

def load_sessions_mouse(dir_path:Path, m:int, c:str="cricket"):
    """Load all sessions for a specific mouse / condition"""

    files_to_process = [file for file in dir_path.iterdir() if match_session_filename(file.name, m=m, suffix=c)]
    sessions = {file.name: load_session(file) for file in files_to_process}
    return sessions 
    

def plot_empirical_copula(df, col1, col2, save_path=None):
    """
    Computes the empirical copula (ranks) for the two given columns,
    and plots the hex jointplot using seaborn.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): Name of the first column.
        col2 (str): Name of the second column.
        save_path (str or Path, optional): Path to save the plot. If None, the plot is not saved.
    """
    if col1 not in df.columns or col2 not in df.columns:
        print("Invalid column names. Please check and try again.")
        return

    vec1 = df[col1].interpolate(method="linear")
    vec2 = df[col2].interpolate(method="linear")
    if col1 in ANGULAR_FEATURES:
        vec1 = angle_to_linear(vec1)
    elif col2 in ANGULAR_FEATURES:
        vec2 = angle_to_linear(vec2)

    vec1_rank = rankdata(vec1, method="average") / (len(vec1) + 1.0)
    vec2_rank = rankdata(vec2, method="average") / (len(vec2) + 1.0)

    copula_df = pd.DataFrame({f'{col1}_rank': vec1_rank, f'{col2}_rank': vec2_rank})

    h = sns.jointplot(data=copula_df, x=f'{col1}_rank', y=f'{col2}_rank', kind='hex')
    h.set_axis_labels(f'{col1}_rank', f'{col2}_rank', fontsize=16)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--m", type=int, help="mouse number: 1, 2, 3")
    parser.add_argument("--c", type=str, help="Condition of the sessions: Cricket or Object")
    parser.add_argument("--output", type=str, default="~/Downloads/copula_outputs/")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dirs = Path("/Users/thomasbush/Downloads/shared WithTWB/")
    sessions_dict = load_sessions_mouse(dirs, m=args.m, c=args.c.lower())

    # unique pairs: (i < j)
    for fx in FEATURE_SET:
        anchor_dir = output_dir / f"{fx}_{args.m}"
        anchor_dir.mkdir(exist_ok=True, parents=True)

        for fy in FEATURE_SET:
            if fy == fx:
                continue
            pair_dir = anchor_dir / f"vs_{fy}"
            pair_dir.mkdir(exist_ok=True, parents=True)

            for session_name, session_df in sessions_dict.items():
                save_path = pair_dir / f"{session_name}.png"
                plot_empirical_copula(session_df, fx, fy, save_path)

