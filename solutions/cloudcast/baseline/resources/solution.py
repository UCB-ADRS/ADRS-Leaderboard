import inspect
from typing import Any, Dict
import networkx as nx

def search_algorithm(src, dsts, G, num_partitions):
    from broadcast import BroadCastTopology
    
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology


class Solution:
    def __init__(self) -> None:
        pass

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        imports = "import networkx as nx\nfrom typing import List\n\n\n"
        function_code = inspect.getsource(search_algorithm)
        code = imports + function_code
        return {"code": code}
