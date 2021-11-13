import sys,os
sys.path.append(os.getcwd())

import pandas as pd
import networkx as nx

from tqdm import tqdm
from itertools import combinations
from fire import Fire

from scripts.create_dark_store_graph import create_darkstore_graph
from scripts.create_prods_graph import create_graph


def get_closest_lagers_to_start(G):
        dist_to_zero = pd.DataFrame()
        all_nodes = []
        for i in range(1, 45):
              list_of_lagers = [i+level for level in [0.1, 0.2, 0.3]]
              all_nodes = all_nodes + list_of_lagers

        for node in all_nodes:
            shortest_path_l = nx.shortest_path_length(G, 0, node, weight='weight')
            dist_to_zero.loc[node, 'dist_to_zero'] = shortest_path_l

        dist_to_zero = dist_to_zero.reset_index()
        dist_to_zero = dist_to_zero.sort_values('dist_to_zero')
        dist_to_zero['SECTION'] = dist_to_zero['index'].apply(lambda x: int(str(x).split('.')[0]))
        dist_to_zero['LEVEL'] = dist_to_zero['index'].apply(lambda x: int(str(x).split('.')[1]))
        dist_to_zero = dist_to_zero.drop(['index', 'dist_to_zero'], axis=1)
        return dist_to_zero.reset_index(drop=True)

def main(
    save_filename
):
    cheques_df = pd.read_csv("data/cheques_public.csv", sep=";")
    darkstore_df = pd.read_csv("data/darkstore_map.csv", sep=";")

    g = create_graph(cheques_df)
    popular = sorted(list(g.nodes.data()), key=lambda n: n[1]["overall_amount"], reverse=True)
    popular = [int(x[0]) for x in popular]

    print('10 most popular products:', popular[:10])

    for n in g.nodes:
        g.nodes[n]["mean_amount"]  = g.nodes[n]["overall_amount"] / g.nodes[n]["num_of_cheques"]

    best_items = []

    n_popular = 6 # Тут різні варіанти пробувати

    for popular_item in popular:
      if popular_item not in best_items:
        best_items.append(popular_item)
        closest_prods = sorted(
            [(n, g.edges[popular_item, n]) for n in list(g.neighbors(popular_item))],
            key=lambda n: n[1]["weight"],
            reverse=True
        )
        n_popular_list = [int(x[0]) for x in closest_prods[:n_popular]]
        best_items += list(set(n_popular_list) - set(best_items))

    print()
    print('Likely to buy list:', best_items)

    G = create_darkstore_graph()

    dist_to_zero = get_closest_lagers_to_start(G)
    dist_to_zero['LAGERID'] = best_items

    dist_to_zero.to_csv(save_filename, index=False, sep=';')


if __name__ == "__main__":
    Fire(main)

