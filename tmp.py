# 提取可能的路径
def build_adjacency_list(edges):
    graph = {}
    for start, end in edges:
        if start not in graph:
            graph[start] = []
        graph[start].append(end)
    return graph


def find_root_node(edges):
    children = {end for _, end in edges}
    parents = {start for start, _ in edges}
    root_candidates = parents - children
    return list(root_candidates)


def find_all_paths_from_root(graph, root):
    def dfs(node, path, all_paths):
        if node not in graph:  # If the node has no children, it's a leaf node
            all_paths.append(path)
            return
        for neighbor in graph[node]:
            dfs(neighbor, path + [neighbor], all_paths)

    all_paths = []
    dfs(root, [root], all_paths)
    return all_paths


def extract_paths_from_root_to_leaves(edges):
    graph = build_adjacency_list(edges)
    root_nodes = find_root_node(edges)
    all_paths = []
    for root in root_nodes:
        all_paths.extend(find_all_paths_from_root(graph, root))
    return all_paths


def main():
    node_dict = {
        642: [{'a': 1, 's_prime': 612, 'prob': 1.0}],
        612: [{'a': 4, 's_prime': 611, 'prob': 1.0}],
        611: [{'a': 4, 's_prime': 610, 'prob': 1.0}],
        610: [{'a': 4, 's_prime': 609, 'prob': 1.0}],
        609: [{'a': 4, 's_prime': 608, 'prob': 1.0}],
        608: [{'a': 4, 's_prime': 607, 'prob': 1.0}],
        607: [{'a': 1, 's_prime': 577, 'prob': 0.5}, {'a': 4, 's_prime': 606, 'prob': 0.5}],
        577: [{'a': 4, 's_prime': 576, 'prob': 1.0}],
        576: [{'a': 4, 's_prime': 575, 'prob': 1.0}],
        575: [{'a': 4, 's_prime': 574, 'prob': 1.0}],
        574: [{'a': 1, 's_prime': 544, 'prob': 1.0}],
        544: [{'a': 4, 's_prime': 543, 'prob': 1.0}],
        543: [{'a': 1, 's_prime': 513, 'prob': 1.0}],
        513: [{'a': 4, 's_prime': 512, 'prob': 1.0}],
        512: [{'a': 1, 's_prime': 482, 'prob': 1.0}],
        482: [{'a': 1, 's_prime': 452, 'prob': 1.0}],
        452: [{'a': 1, 's_prime': 422, 'prob': 1.0}],
        422: [{'a': 0, 's_prime': 423, 'prob': 0.5}, {'a': 1, 's_prime': 392, 'prob': 0.5}],
        423: [{'a': 1, 's_prime': 393, 'prob': 1.0}],
        393: [{'a': 1, 's_prime': 363, 'prob': 1.0}],
        363: [{'a': 1, 's_prime': 333, 'prob': 1.0}],
        333: [{'a': 1, 's_prime': 303, 'prob': 1.0}],
        303: [{'a': 1, 's_prime': 273, 'prob': 1.0}],
        273: [{'a': 1, 's_prime': 243, 'prob': 1.0}],
        243: [{'a': 1, 's_prime': 213, 'prob': 1.0}],
        213: [{'a': 1, 's_prime': 183, 'prob': 1.0}],
        183: [{'a': 1, 's_prime': 153, 'prob': 1.0}],
        153: [{'a': 1, 's_prime': 123, 'prob': 1.0}],
        123: [{'a': 1, 's_prime': 93, 'prob': 1.0}],
        93: [{'a': 1, 's_prime': 63, 'prob': 1.0}],
        63: [{'a': 0, 's_prime': 64, 'prob': 0.5}, {'a': 1, 's_prime': 33, 'prob': 0.5}],
        64: [{'a': 1, 's_prime': 34, 'prob': 1.0}],
        34: [{'a': 1, 's_prime': 4, 'prob': 1.0}],
        33: [{'a': 0, 's_prime': 34, 'prob': 0.5}, {'a': 1, 's_prime': 3, 'prob': 0.5}],
        3: [{'a': 0, 's_prime': 4, 'prob': 1.0}],
        392: [{'a': 0, 's_prime': 393, 'prob': 0.5}, {'a': 1, 's_prime': 362, 'prob': 0.5}],
        362: [{'a': 0, 's_prime': 363, 'prob': 0.5}, {'a': 1, 's_prime': 332, 'prob': 0.5}],
        332: [{'a': 0, 's_prime': 333, 'prob': 0.5}, {'a': 1, 's_prime': 302, 'prob': 0.5}],
        302: [{'a': 0, 's_prime': 303, 'prob': 0.5}, {'a': 1, 's_prime': 272, 'prob': 0.5}],
        272: [{'a': 0, 's_prime': 273, 'prob': 0.5}, {'a': 1, 's_prime': 242, 'prob': 0.5}],
        242: [{'a': 0, 's_prime': 243, 'prob': 0.5}, {'a': 1, 's_prime': 212, 'prob': 0.5}],
        212: [{'a': 0, 's_prime': 213, 'prob': 0.5}, {'a': 1, 's_prime': 182, 'prob': 0.5}],
        182: [{'a': 0, 's_prime': 183, 'prob': 0.5}, {'a': 1, 's_prime': 152, 'prob': 0.5}],
        152: [{'a': 0, 's_prime': 153, 'prob': 0.5}, {'a': 1, 's_prime': 122, 'prob': 0.5}],
        122: [{'a': 0, 's_prime': 123, 'prob': 0.5}, {'a': 1, 's_prime': 92, 'prob': 0.5}],
        92: [{'a': 0, 's_prime': 93, 'prob': 0.5}, {'a': 1, 's_prime': 62, 'prob': 0.5}],
        62: [{'a': 0, 's_prime': 63, 'prob': 0.5}, {'a': 1, 's_prime': 32, 'prob': 0.5}],
        32: [{'a': 0, 's_prime': 33, 'prob': 0.5}, {'a': 1, 's_prime': 2, 'prob': 0.5}],
        2: [{'a': 0, 's_prime': 3, 'prob': 1.0}],
        606: [{'a': 1, 's_prime': 576, 'prob': 0.5}, {'a': 4, 's_prime': 605, 'prob': 0.5}],
        605: [{'a': 1, 's_prime': 575, 'prob': 0.5}, {'a': 4, 's_prime': 604, 'prob': 0.5}],
        604: [{'a': 1, 's_prime': 574, 'prob': 1.0}]
    }
    edges = []
    new_node_dict = {}
    for start, ends in node_dict.items():
        new_ends = {}
        for info in ends:
            end = info.pop('s_prime')
            edges.append((start, end))
            new_ends[end] = info
        new_node_dict[start] = new_ends

    all_paths = extract_paths_from_root_to_leaves(edges)
    probs = []
    for path in all_paths:
        prob = 1.0
        for i, p1 in enumerate(path[:-1]):
            p2 = path[i+1]
            prob *= new_node_dict[p1][p2]['prob']
        probs.append(prob)
        print('{:>2d}, {}'.format(len(path), prob), path)
    probs = list(sorted(probs))
    print(sum(probs), probs)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(probs)
    plt.show()


if __name__ == '__main__':
    main()

