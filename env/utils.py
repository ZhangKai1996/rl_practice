
def state2coord(state, size, reg=False):
    x = state // size
    y = state % size
    if reg:
        return [float(x)/size, float(y)/size]
    return [x, y]


def coord2state(coord, size):
    return coord[0] * size + coord[1]


def distance(p1, p2, size):
    p1 = state2coord(p1, size)
    p2 = state2coord(p2, size)
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
