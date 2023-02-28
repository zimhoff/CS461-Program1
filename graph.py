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


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge], directed=False):
        self.nodes = nodes
        if directed:
            self.edges = edges
        else:
            self.edges = [
                edge for pair in [(e, e.inverse()) for e in edges] for edge in pair
            ]

    def neighbors(self, src: str) -> Set[Node]:
        return {edge.dest for edge in self.edges if edge.src.name == src}

    def get_node(self, name: str) -> Node:
        node_names = list(map(lambda x: x.name, self.nodes))
        if name in node_names:
            return self.nodes[node_names.index(name)]
        else:
            raise KeyError("Node not found.")

    def a_star(self, start: str, goal: str):
        """I used Wikipedia's article on A* as a major resource for this implementation!
        https://en.wikipedia.org/wiki/A*_search_algorithm"""

        start_node: Node = self.get_node(start)
        goal_node: Node = self.get_node(goal)

        # For each city we visit, we'll store the city just prior in `came_from` as a {city: prior} k/v pair.
        came_from: Dict[str, str] = dict()

        # Consider the distances to be infinite until we can show otherwise.
        g_score: DefaultDict[str, float] = defaultdict(lambda: float("inf"))
        f_score: DefaultDict[str, float] = defaultdict(lambda: float("inf"))

        heuristic = lambda node: manhattan_distance(node.location, goal_node.location)

        g_score[start] = 0
        f_score[start] = heuristic(start_node)

        # open_set is the foundation for a min-heap ordered by f_score.
        open_set = [(f_score[start], start)]

        # Continue looping until we can't pop a city off the open set anymore.
        while current := heapq.heappop(open_set):
            current_node = self.get_node(current[1])

            # If we've reached the goal, print out the path we took to get here and exit.
            if current_node.name == goal:
                print(" -> ".join(backtrack(came_from, current_node.name)))
                return

            for neighbor in self.neighbors(current_node.name):
                # For each neighbor, assign a tentative g_score as the current node's known
                # g_score plus the Manhattan distance from the node to our goal location.
                tentative_g_score = g_score[current_node.name] + manhattan_distance(
                    current_node.location, neighbor.location
                )

                # Since the default g_score is infinity, any previously unvisited neighbors will be visited here.
                if tentative_g_score < g_score[neighbor.name]:
                    came_from[neighbor.name] = current_node.name
                    g_score[neighbor.name] = tentative_g_score
                    f_score[neighbor.name] = tentative_g_score + heuristic(neighbor)

                    # If the current neighbor isn't in the open_set, go ahead and heap-push it.
                    if neighbor.name not in map(lambda x: x[1], open_set):
                        heapq.heappush(
                            open_set, (f_score[neighbor.name], neighbor.name)
                        )

        # Since we return early for successful traversals, we can
        # be sure a path doesn't exist once we've exited the loop.
        print("No paths found.")
        return False