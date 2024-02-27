import heapq

import matplotlib.pyplot as plt
import numpy as np


def distance(x, y):
    return (x[0] - y[0]) ** 2 + ((x[1] - y[1]) ** 2)


class Map:
    def __init__(self, m: np.ndarray) -> None:
        self.m = m

    def neighbors(self, cell):
        nrow, ncol = m.shape
        x, y = cell
        nb = []
        if x > 0:
            if m[x - 1, y] == 0:
                nb = nb + [(x - 1, y)]
        if x < (nrow - 1):
            if m[x + 1, y] == 0:
                nb = nb + [(x + 1, y)]
        if y > 0:
            if m[x, y - 1] == 0:
                nb = nb + [(x, y - 1)]
        if y < (ncol - 1):
            if m[x, y + 1] == 0:
                nb = nb + [(x, y + 1)]
        return nb


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class Stack:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.pop()


# Hilfsfunktion zur Rekonstruktion des Pfades
def make_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:  # Rückverfolgung vom Ziel zum Start
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()  # Umkehrung des Pfades, um ihn von Start zu Ziel zu ordnen
    return path


def dfs(map: Map, start, goal):
    stack = Stack()
    stack.put(start)
    came_from = {start: None}

    while not stack.empty():
        current = stack.get()

        # Ziel gefunden
        if current == goal:
            break

        for next in map.neighbors(current):
            # Wenn der Nachbarknoten noch nicht besucht wurde, füge es in dem Stack hinzu
            if next not in came_from:
                stack.put(next)
                came_from[next] = current  # Speichere den Vorgänger

    return make_path(came_from, start, goal)  # Gib den Weg zurück


def astar(map: Map, start, goal):
    def heuristic(a):
        return distance(a, goal)

    p_queue = PriorityQueue()  # Startpunkt hat Priorität 0
    p_queue.put(start, 0)
    came_from = {start: None}  # Vorgänger
    cost = {start: 0}  # Kosten von Start bis zu jedem Knoten

    while not p_queue.empty():
        current = p_queue.get()

        if current == goal:
            break  # Ziel gefunden

        for next in map.neighbors(current):
            new_cost = cost[current] + distance(current, next)
            if next not in cost or new_cost < cost[next]:
                cost[next] = new_cost
                priority = new_cost + heuristic(next)  # Priorität mit Heuristik aktualisieren
                p_queue.put(next, priority)
                came_from[next] = current  # Pfad aktualisieren

    return make_path(came_from, start, goal)


def draw_path(search_alg, map, start, goal):
    path = search_alg(map, start, goal)
    print(path)

    # Visualization of the map with the path
    fig, ax = plt.subplots()
    ax.imshow(map.m, cmap='Purples', interpolation='nearest')
    x_coords, y_coords = zip(*path)
    ax.plot(y_coords, x_coords, color='black', linewidth=3)  # Visualize path
    ax.scatter(y_coords[0], x_coords[0], color='red', label='Start', s=250)  # Mark start point
    ax.scatter(y_coords[-1], x_coords[-1], color='green', label='Goal', s=250)  # Mark goal point
    ax.legend()

    ax.set_title(search_alg.__name__, fontsize=24)
    plt.show()


m = np.array(
    [[0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 1],
     [0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 1, 1, 0, 0]])
mm = Map(m)
mm.neighbors((4, 1))

_start = (0, 0)
_goal = (5, 6)

draw_path(dfs, mm, _start, _goal)
draw_path(astar, mm, _start, _goal)
