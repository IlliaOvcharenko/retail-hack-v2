import sys,os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import networkx as nx

from tqdm.cli import tqdm
from pathlib import Path
from fire import Fire

from scripts.create_dark_store_graph import create_darkstore_graph
from scripts.create_prods_graph import create_graph


def fill_in_nodes_distances(store_graph):
    store_graph.remove_nodes_from([n for n in store_graph.nodes if n not in range(1, 44+1)])
    shotest_pathes = nx.floyd_warshall(store_graph)

    g = nx.Graph()
    for node, nodes_to in shotest_pathes.items():
        del nodes_to[node]
        for node_to, edge_weight in nodes_to.items():
            g.add_edge(node, node_to, weight=edge_weight)
    return g


def get_shelf_order(store_graph, init_node=1, mode="v1"):
    assert mode in ["v1", "v2", "v3", "v4"], f"no such mode available: {mode}"
    if mode == "v1":
        store_graph = fill_in_nodes_distances(store_graph)

        order = [init_node]
        n_nodes = store_graph.number_of_nodes()

        cur_node = init_node
        while store_graph.number_of_nodes() > 1:
            # print(f"node: {cur_node}, neighbors: {list(store_graph.neighbors(cur_node))}")

            closest_node = min(
                [(n, store_graph.edges[cur_node, n]) \
                 for n in store_graph.neighbors(cur_node)],
                key=lambda n: n[1]["weight"],
            )[0]
            order.append(closest_node)
            store_graph.remove_node(cur_node)
            cur_node = closest_node

    elif mode == "v2":
        store_graph.remove_nodes_from([n for n in store_graph.nodes if n not in range(0, 44+1)])
        order = list(sorted(
            store_graph.nodes,
            key=lambda n: nx.shortest_path_length(store_graph, 0, n, weight='weight')
        ))[1:]

    elif mode == "v3":
        store_graph.remove_nodes_from([n for n in store_graph.nodes if n not in range(0, 44+1)])

        dist_to_zero = pd.DataFrame()
        for node in store_graph.nodes:
            shortest_path_l = nx.shortest_path_length(store_graph, 0, node, weight='weight')
            dist_to_zero.loc[node, 'dist_to_zero'] = shortest_path_l

        dist_to_zero = dist_to_zero.reset_index()
        dist_to_zero = dist_to_zero.sort_values('dist_to_zero')
        dist_to_zero = dist_to_zero.reset_index(drop=True)
        order = dist_to_zero.values[1:, 0].astype(int).tolist()

    elif mode == "v4":
        order = [1, 2, 15, 7, 16, 8, 17, 9, 18, 10, 19, 11, 20, 12, 21, 13, 22, 14, 40, 39, 41, \
                 42, 43, 44, 38, 30, 37, 29, 36, 28, 35, 27, 34, 26, 33, 25, 32, 24, 31, 23, 5, \
                 6, 4, 3]

    return order


def get_n_nodes(prod_graph, n_closest=3):
    most_popular_prod = max(list(prod_graph.nodes.data()), key=lambda n: n[1]["overall_amount"])

    prod_id = most_popular_prod[0]
    closest_prods = sorted(
        [(n, prod_graph.edges[prod_id, n]) for n in list(prod_graph.neighbors(prod_id))],
        key=lambda n: n[1]["weight"],
        reverse=True
    )

    closest_prods = closest_prods[:n_closest-1]
    selected = [most_popular_prod[0]] + [n[0] for n in closest_prods]
    not_selected = [n for n in prod_graph.nodes if n not in selected]
    return prod_graph.subgraph(selected), prod_graph.subgraph(not_selected)


def split_into_parts(prod_graph, random_state):
    if prod_graph.number_of_nodes() in [33, 15]:
        selected, prod_graph = get_n_nodes(prod_graph)
        split = nx.algorithms.community.kernighan_lin_bisection(
            prod_graph,
            seed=random_state-1,
            max_iter=1000
        )
        res1 = split_into_parts(prod_graph.subgraph(split[0]), random_state)
        res2 = split_into_parts(prod_graph.subgraph(split[1]), random_state)

        return res1 + res2 + [selected]

    elif prod_graph.number_of_nodes() <= 3:
        return [prod_graph]

    else:
        split = nx.algorithms.community.kernighan_lin_bisection(
            prod_graph,
            seed=random_state,
            max_iter=1000
        )
        res1 = split_into_parts(prod_graph.subgraph(split[0]), random_state)
        res2 = split_into_parts(prod_graph.subgraph(split[1]), random_state)

        return res1 + res2


def get_graph_totoal_amount(g):
    return sum([n[1]["overall_amount"] for n in g.nodes.data()])


def get_prod_order(g, sort_by="overall_amount"):
    items = list(sorted(g.nodes, key=lambda n: g.nodes[n][sort_by], reverse=True))
    return items


def create_df(product_groupds, shelf_order):
    df = pd.DataFrame()
    for pg, sh in zip(product_groupds, shelf_order):
        # print(pg, "total amount:", get_graph_totoal_amount(pg))
        prod_order = get_prod_order(pg)
        for level, p in enumerate(prod_order):
            df = df.append({
                "SECTION": sh,
                "LEVEL": level+1,
                "LAGERID": p,
            }, ignore_index=True)

    df = df.astype(int)
    return df


def main(
    save_filename,
    random_state=42,
    shelf_order_mode="v3",

):
    data_folder = Path("data")
    cheques_df = pd.read_csv(data_folder / "cheques_public.csv", sep=";")

    store_graph = create_darkstore_graph()
    prod_graph = create_graph(cheques_df)

    shelf_order = get_shelf_order(store_graph, mode=shelf_order_mode)

    parts = split_into_parts(prod_graph, random_state)
    parts = list(sorted(parts, key=get_graph_totoal_amount, reverse=True))

    # without split, on every iteration take 1 most popular and 2 related products
    # parts = []
    # while prod_graph.number_of_nodes() > 3:
    #     selected, prod_graph = get_n_nodes(prod_graph)
    #     parts.append(selected)
    # parts.append(prod_graph)

    df = create_df(parts, shelf_order)
    df.to_csv(save_filename, index=False, sep=';')

if __name__ == "__main__":
    Fire(main)

