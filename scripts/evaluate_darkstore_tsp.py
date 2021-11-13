import pandas as pd
import networkx as nx
from tqdm import tqdm
from create_dark_store_graph import create_darkstore_graph
cheques_public = pd.read_csv('data/cheques_public.csv', sep=';')
darkstore_map = pd.read_csv('data/darkstore_map.csv', sep=';')

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

G = create_darkstore_graph()

total_total_time = 0
for cheque in tqdm(list(cheques_public.CHEQUEID.unique())):
  total_total_time += evaluate_cheque_tsp(cheques_public[cheques_public.CHEQUEID == cheque], darkstore_map, G)
print()
print('Total time:', total_total_time, 'Mean total time:', total_total_time / len(cheques_public.CHEQUEID.unique()))