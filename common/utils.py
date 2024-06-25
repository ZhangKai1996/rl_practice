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


def extract_all_paths(edges):
    graph = build_adjacency_list(edges)
    root_nodes = find_root_node(edges)
    all_paths = []
    for root in root_nodes:
        all_paths.extend(find_all_paths_from_root(graph, root))
    return all_paths
