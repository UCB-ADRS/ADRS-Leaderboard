from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from broadcast import BroadCastTopology

import networkx as nx
from typing import List
from collections import defaultdict
import heapq

def search_algorithm(src: str, dsts: List[str], G: nx.DiGraph, num_partitions: int) -> BroadCastTopology:
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    # Precompute shortest paths from source to all nodes
    dist_from_src = {}
    prev_from_src = {}
    dist_from_src[src] = 0
    pq = [(0, src)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist_from_src.get(u, float('inf')):
            continue
        for v in G[u]:
            edge_cost = G[u][v]['cost']
            new_dist = d + edge_cost
            if new_dist < dist_from_src.get(v, float('inf')):
                dist_from_src[v] = new_dist
                prev_from_src[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    # Build direct paths from source to each destination
    direct_paths = {}
    for dst in dsts:
        if dst in dist_from_src:
            path = []
            cur = dst
            while cur != src:
                path.append(cur)
                cur = prev_from_src[cur]
            path.append(src)
            direct_paths[dst] = path[::-1]
    
    # Group destinations by cloud provider
    cloud_dests = defaultdict(list)
    for dst in dsts:
        if dst.startswith("aws"):
            cloud_dests["aws"].append(dst)
        elif dst.startswith("azure"):
            cloud_dests["azure"].append(dst)
        elif dst.startswith("gcp"):
            cloud_dests["gcp"].append(dst)
    
    # For each cloud, find best hub to minimize total cost
    all_nodes = list(G.nodes())
    cloud_hubs = {}
    hub_to_dest_paths = {}
    
    for cloud, cloud_dsts in cloud_dests.items():
        if not cloud_dsts:
            continue
            
        cloud_nodes = [n for n in all_nodes if n.startswith(cloud)]
        best_hub = None
        best_total_cost = float('inf')
        best_paths = {}
        
        for hub in cloud_nodes:
            if hub not in dist_from_src:
                continue
            
            # Compute shortest paths from hub to all destinations in this cloud
            dist_from_hub = {}
            prev_from_hub = {}
            dist_from_hub[hub] = 0
            pq_hub = [(0, hub)]
            
            while pq_hub:
                d, u = heapq.heappop(pq_hub)
                if d > dist_from_hub.get(u, float('inf')):
                    continue
                for v in G[u]:
                    edge_cost = G[u][v]['cost']
                    new_dist = d + edge_cost
                    if new_dist < dist_from_hub.get(v, float('inf')):
                        dist_from_hub[v] = new_dist
                        prev_from_hub[v] = u
                        heapq.heappush(pq_hub, (new_dist, v))
            
            total_cost = dist_from_src[hub]  # Source to hub
            valid = True
            paths = {}
            
            for dst in cloud_dsts:
                if dst == hub:
                    continue
                if dst not in dist_from_hub:
                    valid = False
                    break
                total_cost += dist_from_hub[dst]
                
                # Reconstruct hub->dst path
                path = []
                cur = dst
                while cur != hub:
                    path.append(cur)
                    cur = prev_from_hub[cur]
                path.append(hub)
                paths[dst] = path[::-1]
            
            if valid and total_cost < best_total_cost:
                best_total_cost = total_cost
                best_hub = hub
                best_paths = paths
        
        if best_hub:
            cloud_hubs[cloud] = best_hub
            hub_to_dest_paths[best_hub] = best_paths
    
    # Calculate costs for direct vs hub routing
    direct_total_cost = sum(dist_from_src.get(d, float('inf')) for d in dsts)
    
    hub_routing_cost = 0
    hub_routing_details = defaultdict(list)
    
    for cloud, cloud_dsts in cloud_dests.items():
        if cloud in cloud_hubs:
            hub = cloud_hubs[cloud]
            hub_routing_cost += dist_from_src[hub]
            hub_routing_details[hub].extend(cloud_dsts)
        else:
            # No hub found for this cloud, use direct
            for dst in cloud_dsts:
                hub_routing_cost += dist_from_src.get(dst, float('inf'))
    
    # Determine strategy: use hub routing if cheaper overall
    use_hub_routing = hub_routing_cost < direct_total_cost
    
    # Build paths for all partitions
    for partition in range(num_partitions):
        if use_hub_routing:
            # Use hub-based routing
            processed_hubs = set()
            
            for hub, destinations in hub_routing_details.items():
                if hub in processed_hubs:
                    continue
                    
                # Add source-to-hub path (once per hub)
                if hub in direct_paths:
                    hub_path = direct_paths[hub]
                    # Assign to the first destination using this hub
                    if destinations:
                        first_dst = destinations[0]
                        for i in range(len(hub_path)-1):
                            u, v = hub_path[i], hub_path[i+1]
                            bc_topology.append_dst_partition_path(first_dst, partition, [u, v, G[u][v]])
                
                # Add hub-to-destination paths
                for dst in destinations:
                    if dst == hub:
                        # Destination is the hub itself
                        continue
                    
                    if hub in hub_to_dest_paths and dst in hub_to_dest_paths[hub]:
                        path = hub_to_dest_paths[hub][dst]
                        for i in range(len(path)-1):
                            u, v = path[i], path[i+1]
                            bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
                    else:
                        # Fallback: compute path from hub to dst
                        try:
                            path = nx.dijkstra_path(G, hub, dst, weight='cost')
                            for i in range(len(path)-1):
                                u, v = path[i], path[i+1]
                                bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
                        except:
                            # Ultimate fallback: direct from source
                            if dst in direct_paths:
                                path = direct_paths[dst]
                                for i in range(len(path)-1):
                                    u, v = path[i], path[i+1]
                                    bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
                
                processed_hubs.add(hub)
            
            # Handle destinations not covered by hub routing
            all_hub_dests = set()
            for dest_list in hub_routing_details.values():
                all_hub_dests.update(dest_list)
            
            for dst in dsts:
                if dst not in all_hub_dests:
                    if dst in direct_paths:
                        path = direct_paths[dst]
                        for i in range(len(path)-1):
                            u, v = path[i], path[i+1]
                            bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
        else:
            # Use direct routing for all destinations
            for dst in dsts:
                if dst in direct_paths:
                    path = direct_paths[dst]
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
    
    # Ensure all destinations have paths for all partitions
    for dst in dsts:
        for partition in range(num_partitions):
            if not bc_topology.paths.get(dst, {}).get(str(partition), []):
                if dst in direct_paths:
                    path = direct_paths[dst]
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
    
    return bc_topology


class Solution:
    """LEVI submission for cloudcast."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split("\n")
        end_idx = next(i for i, line in enumerate(lines) if line.startswith("class Solution:"))
        return {"code": "\n".join(lines[:end_idx]).rstrip()}
