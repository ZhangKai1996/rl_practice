
def state2coord(state, size):
    x = state // size
    y = state % size
    return x, y


def coord2state(coord, size):
    return coord[0] * size + coord[1]
