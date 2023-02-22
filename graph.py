from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import DefaultDict, Dict, List, Set, Tuple
import heapq


def manhattan_distance(src: Tuple, dest: Tuple):
    return sqrt(abs(src[0] - dest[0]) + abs(src[1] - dest[1]))


def backtrack(came_from: Dict[str, str], city: str):
    path: List[str] = [city]
    while came_from.get(city) is not None:
        city = came_from.get(city)
        path.append(city)
    return list(reversed(path))


class Node:
    def __init__(self, name: str, location: Tuple[float, float]):
        self.name = name
        self.location = location

        def __eq__(self, other: Node) -> bool:
            return self.name == other.name

        def __hash__(self) -> int:
            return hash(self.name)


@dataclass
class Edge:
    src: Node
    dest: Node
    weight: float

    def inverse(self) -> "Edge":
        return Edge(self.dest, self.src, self.weight)


