import sys,os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from pathlib import Path
from fire import Fire
from functools import partial
from multiprocessing import Pool
from tqdm.cli import tqdm

from scripts.evaluate_darkstore_tsp import evaluate_cheque_tsp
from scripts.create_dark_store_graph import create_darkstore_graph


def generate_random_map(basemap_df):
    basemap_df = basemap_df.copy()
    basemap_df["LAGERID"] = basemap_df["LAGERID"].sample(frac=1).values
    return basemap_df


def eval(cheques_df, map_df, n_proc):
    cheques_public = cheques_df
    darkstore_map = map_df

    G = create_darkstore_graph()

    total_total_time = 0
    cheques = [ch for _, ch in cheques_public.groupby("CHEQUEID")]

    with Pool(n_proc) as p:
        map_func = partial(
            evaluate_cheque_tsp,
            darkstore_map=darkstore_map,
            G=G,
        )

        total_total_time = sum(list(tqdm(p.imap(map_func, cheques),
                                         total=len(cheques),
                                         desc="process cheques")))

    print()
    print('Total time:', total_total_time,
          'Mean total time:', total_total_time / len(cheques_public.CHEQUEID.unique()))
    return total_total_time


def main(
    n_iters,
    save_folder,
    random_state=42,
    n_proc=8
):
    np.random.seed(random_state)

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    data_folder = Path("data")
    basemap_fn = data_folder / "darkstore_map.csv"
    basemap_df = pd.read_csv(basemap_fn, sep=";")
    cheques_fn = data_folder / "cheques_public.csv"
    cheques_df = pd.read_csv(cheques_fn, sep=";")

    for i in range(n_iters):
        print(f"iter: {i+1}/{n_iters}")
        random_map = generate_random_map(basemap_df)
        score = eval(cheques_df, random_map, n_proc)
        save_filename = save_folder / f"random-map-score={score}-iter={i}.csv"
        random_map.to_csv(save_filename, index=False, sep=';')


if __name__ == "__main__":
    Fire(main)

