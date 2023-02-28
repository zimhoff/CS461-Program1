from collections import ChainMap
from functools import partial, reduce
from typing import Dict, List, Tuple
from graph import Graph, Edge, Node, manhattan_distance


def read_coordinates(filename: str) -> Dict[str, Tuple[float, float]]:
    with open(filename) as f:
        coordinates = reduce(
            lambda x, y: y(x),
            [
                partial(map, str.strip),
                partial(map, str.split),
                partial(map, lambda x: {x[0]: tuple(map(float, x[1:]))}),
                partial(lambda mapping: ChainMap(*mapping)),
                dict,
            ],
            f.readlines(),
        )
    return coordinates


def read_adjacencies(filename: str) -> List[str]:
    with open(filename) as f:
        adjacencies: List[str] = reduce(
            lambda x, y: y(x),
            [
                partial(map, partial(str.strip)),
                partial(map, partial(str.split)),
                partial(map, list),
                list,
            ],
            f.readlines(),
        )
    return adjacencies


if __name__ == "__main__":
    coordinates = read_coordinates("coordinates.txt")
    adjacencies = read_adjacencies("Adjacencies.txt")

    edges = []

    for city in adjacencies:
        for neighbor in city[1:]:
            src = Node(city[0], coordinates[city[0]])
            dest = Node(neighbor, coordinates.get(neighbor))
            edges.append(
                Edge(src, dest, manhattan_distance(src.location, dest.location))
            )

    nodes = [Node(city, coordinates[city]) for city in coordinates.keys()]

    graph = Graph(nodes, edges)

    while (start := input("Start: ")) not in coordinates.keys():
        print("Error: unrecognized city.")

    while (goal := input("Goal: ")) not in coordinates.keys():
        print("Error: unrecognized city.")

    graph.a_star(start, goal)