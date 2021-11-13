import sys,os
sys.path.append(os.getcwd())

import pandas as pd
import networkx as nx

from tqdm import tqdm
from fire import Fire
from functools import partial
from multiprocessing import Pool

from scripts.create_dark_store_graph import create_darkstore_graph


def evaluate_cheque_tsp(one_cheque, darkstore_map, G):
  multiple_items = []
  path = []
  for _, row in one_cheque.iterrows():
          lager_info = darkstore_map[darkstore_map.LAGERID == row['LAGERID']]
          section = lager_info.iloc[0]['SECTION']
          level = lager_info.iloc[0]['LEVEL']
          quantity = row.loc['KOLVO']
          graph_node = section + level / 10
          if quantity > 1:
            multiple_items.append((graph_node, int(quantity)))
          path.append(graph_node)

  total_time = 0
  shortest_path = nx.approximation.traveling_salesman_problem(G, nodes=[0] + path + [0])
  for first, second in zip(shortest_path, shortest_path[1:]):
    multiple_items_section = [x[0] for x in multiple_items]
    multiple_items_quantity = [x[1] for x in multiple_items]
    time = nx.shortest_path_length(G, source=first, target=second, weight='weight')
    quantity = 1
    if first in multiple_items_section:
      quantity = multiple_items_quantity[multiple_items_section.index(first)]
    if second in multiple_items_section:
      quantity = multiple_items_quantity[multiple_items_section.index(second)]
    total_time += time * quantity
  return total_time


def main(
    darkstore_map_fn,
    n_proc=8,
):
    cheques_public = pd.read_csv('data/cheques_public.csv', sep=';')
    darkstore_map = pd.read_csv(darkstore_map_fn, sep=';')


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


if __name__ == "__main__":
    Fire(main)

