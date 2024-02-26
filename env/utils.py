import math


def distance(s1, s2, size):
    coord1 = state2coord(s1, size)
    coord2 = state2coord(s2, size)
    return math.sqrt(
        math.pow(coord1[0]-coord2[0], 2) +
        math.pow(coord1[1]-coord2[1], 2)
    )


def state2coord(state, size):
    x = state // size
    y = state % size
    return x, y


def coord2state(coord, size):
    return coord[0] * size + coord[1]
