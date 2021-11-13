import networkx as nx

def create_darkstore_graph():

    G = nx.Graph()
    G.add_edge(0, 1, weight=8)
    G.add_edge(0, 39, weight=18)

    # переходи на кутах
    G.add_edge(2, 15, weight=2)
    G.add_edge(2, 7, weight=2)

    G.add_edge(40, 22, weight=2)
    G.add_edge(40, 14, weight=2)

    G.add_edge(5, 23, weight=2)
    G.add_edge(5, 31, weight=2)

    G.add_edge(43, 30, weight=2)
    G.add_edge(43, 38, weight=2)
    # переходи на кутах

    # вертикальна ліва
    for i in [1, 2, 3, 4, 5]:
        G.add_edge(i, i+1, weight=2)

    # вертикальна права
    for i in [39, 40, 41, 42, 43]:
        G.add_edge(i, i+1, weight=2)

    # суміжня нижня
    down_list = [7, 8, 9, 10, 11, 12, 13, 14]
    for i in down_list[:-1]:
        G.add_edge(i, i+1, weight=2)

    up_list = [15, 16, 17, 18, 19, 20, 21, 22]
    for i in up_list[:-1]:
        G.add_edge(i, i+1, weight=2)

    for i, j in zip(down_list, up_list):
        G.add_edge(i, j, weight=0)
    # суміжня нижня

    # суміжня верхня
    down_list = [23, 24, 25, 26, 27, 28, 29, 30]
    for i in down_list[:-1]:
        G.add_edge(i, i+1, weight=2)

    up_list = [31, 32, 33, 34, 35, 36, 37, 38]
    for i in up_list[:-1]:
        G.add_edge(i, i+1, weight=2)

    for i, j in zip(down_list, up_list):
        G.add_edge(i, j, weight=0)
    # суміжня верхня
    
    for i in range(1, 45):
        G.add_edge(i, i+0.1, weight=1/2)
        G.add_edge(i, i+0.2, weight=2/2)
        G.add_edge(i, i+0.3, weight=3/2)
    
    return G