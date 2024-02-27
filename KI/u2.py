import numpy as np
import matplotlib.pyplot as plt


def distance(x, y):
    return ((x[0] - y[0])) ** 2 + ((x[1] - y[1]) ** 2)


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


import heapq


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


def dfs(map: Map, start, goal):
    frontier = Stack()  # Der Stack für die noch zu besuchenden Knoten
    frontier.put(start)
    came_from = {start: None}  # Dictionary, um den Vorgänger jedes Knotens zu speichern

    while not frontier.empty():
        current = frontier.get()

        # Ziel gefunden
        if current == goal:
            break

        for next in map.neighbors(current):
            # Wenn der Nachbarknoten noch nicht besucht wurde, fügen Sie ihn dem Stack hinzu
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current  # Speichern Sie den Vorgänger

    return came_from


m = np.array(
    [[0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 1],
     [0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 1, 1, 0, 0]])
mm = Map(m)
mm.neighbors((4, 1))

start = (0, 0)
goal = (5, 6)
path = dfs(mm, start, goal)
print(path)

# Visualisierung der Karte
plt.imshow(m, cmap='Purples')
plt.colorbar()
plt.show()
