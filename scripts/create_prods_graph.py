import sys,os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm.cli import tqdm
from pathlib import Path
from itertools import combinations
from fire import Fire


def create_graph(cheques_df):
    g = nx.Graph()

    for cheque_id, cheque_df in tqdm(cheques_df.groupby("CHEQUEID", sort=True)):
        for _, cheque_item in cheque_df.iterrows():
            if g.has_node(cheque_item.LAGERID):
                g.nodes[cheque_item.LAGERID]["num_of_cheques"] += 1
                g.nodes[cheque_item.LAGERID]["overall_amount"] += cheque_item.KOLVO
            else:
                node_info = {
                    "num_of_cheques": 1,
                    "overall_amount": cheque_item.KOLVO
                }
                g.add_node(cheque_item.LAGERID, **node_info)

        produc_ids =  cheque_df.LAGERID.unique()
        for prod_1, prod_2 in list(combinations(produc_ids, 2)):
            if g.has_edge(prod_1, prod_2):
                g.edges[prod_1, prod_2]['weight'] += 1
            else:
                g.add_edge(prod_1, prod_2, weight=1)
    return g


def main():
    data_folder = Path("data")
    cheques_df = pd.read_csv(data_folder / "cheques_public.csv", sep=";")
    darkstore_df = pd.read_csv(data_folder / "darkstore_map.csv", sep=";")

    g = create_graph(cheques_df)

    # add mean amount of prodcut per cheque
    for n in g.nodes:
        g.nodes[n]["mean_amount"]  = g.nodes[n]["overall_amount"] / g.nodes[n]["num_of_cheques"]

    # calculate most popular node
    most_popular_prod = max(list(g.nodes.data()), key=lambda n: n[1]["overall_amount"])
    print(f"most popular node:", most_popular_prod)


    print(list(g.edges.data())[:10])

    plt.figure(figsize=(25, 25))
    nx.draw(g, width=0.05, with_labels=True)
    plt.savefig("prod-graph", bbox_inches="tight")


if __name__ == "__main__":
    Fire(main)

