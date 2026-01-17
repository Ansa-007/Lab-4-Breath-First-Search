"""
Breadth-First Search (BFS) Algorithm Implementation
Industry-Grade Graph Traversal Solution

Author: Khansa Younas
Version: 2.0

This module provides a comprehensive BFS implementation with advanced features
including error handling, performance optimization, and industry-standard practices.
"""

from collections import deque
from typing import Dict, List, Set, Any, Optional, Union
import logging
import time
from dataclasses import dataclass
from enum import Enum


class TraversalResult(Enum):
    """Enumeration for traversal result types"""
    SUCCESS = "success"
    INVALID_START_NODE = "invalid_start_node"
    EMPTY_GRAPH = "empty_graph"
    CYCLE_DETECTED = "cycle_detected"


@dataclass
class BFSMetrics:
    """Data class to store BFS execution metrics"""
    nodes_visited: int
    edges_explored: int
    execution_time_ms: float
    max_queue_size: int
    path_length: int


class GraphValidator:
    """Utility class for graph validation and integrity checks"""
    
    @staticmethod
    def validate_graph(graph: Dict[Any, List[Any]]) -> tuple[bool, Optional[str]]:
        """
        Validate graph structure and integrity
        
        Args:
            graph: Graph represented as adjacency list
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(graph, dict):
            return False, "Graph must be a dictionary"
        
        if not graph:
            return False, "Graph cannot be empty"
        
        for node, neighbors in graph.items():
            if not isinstance(neighbors, list):
                return False, f"Neighbors of node {node} must be a list"
            
            for neighbor in neighbors:
                if neighbor not in graph:
                    return False, f"Neighbor {neighbor} of node {node} not found in graph"
        
        return True, None
    
    @staticmethod
    def detect_cycles(graph: Dict[Any, List[Any]]) -> List[List[Any]]:
        """
        Detect cycles in the graph using DFS
        
        Args:
            graph: Graph represented as adjacency list
            
        Returns:
            List of cycles found in the graph
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Cycle detected
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles


class IndustryBFS:
    """
    Industry-grade Breadth-First Search implementation with advanced features
    """
    
    def __init__(self, enable_logging: bool = True, log_level: int = logging.INFO):
        """
        Initialize BFS with optional logging
        
        Args:
            enable_logging: Whether to enable logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = self._setup_logger(enable_logging, log_level)
        self.metrics = None
    
    def _setup_logger(self, enable_logging: bool, log_level: int) -> logging.Logger:
        """Setup and configure logger"""
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        
        if enable_logging and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def bfs(self, 
            graph: Dict[Any, List[Any]], 
            start: Any, 
            return_metrics: bool = False,
            detect_cycles: bool = False) -> Union[List[Any], tuple[List[Any], BFSMetrics]]:
        """
        Perform Breadth-First Search on the given graph
        
        Args:
            graph: Graph represented as adjacency list
            start: Starting node for traversal
            return_metrics: Whether to return execution metrics
            detect_cycles: Whether to perform cycle detection
            
        Returns:
            List of nodes in BFS order, optionally with metrics
            
        Raises:
            ValueError: If graph is invalid or start node not found
        """
        start_time = time.time()
        
        # Validate graph
        is_valid, error_msg = GraphValidator.validate_graph(graph)
        if not is_valid:
            self.logger.error(f"Graph validation failed: {error_msg}")
            raise ValueError(f"Invalid graph: {error_msg}")
        
        # Check if start node exists
        if start not in graph:
            self.logger.error(f"Start node {start} not found in graph")
            raise ValueError(f"Start node {start} not found in graph")
        
        # Cycle detection if requested
        if detect_cycles:
            cycles = GraphValidator.detect_cycles(graph)
            if cycles:
                self.logger.warning(f"Cycles detected: {cycles}")
        
        # Initialize BFS data structures
        visited: Set[Any] = set()
        queue: deque = deque([start])
        visited.add(start)
        
        path: List[Any] = []
        edges_explored = 0
        max_queue_size = 1
        
        self.logger.info(f"Starting BFS traversal from node: {start}")
        
        # Main BFS loop
        while queue:
            # Track maximum queue size for metrics
            max_queue_size = max(max_queue_size, len(queue))
            
            node = queue.popleft()
            path.append(node)
            
            self.logger.debug(f"Processing node: {node}")
            
            # Explore neighbors
            for neighbor in graph[node]:
                edges_explored += 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    self.logger.debug(f"Added neighbor {neighbor} to queue")
        
        # Calculate metrics
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.metrics = BFSMetrics(
            nodes_visited=len(path),
            edges_explored=edges_explored,
            execution_time_ms=execution_time,
            max_queue_size=max_queue_size,
            path_length=len(path)
        )
        
        self.logger.info(f"BFS completed. Nodes visited: {len(path)}, "
                        f"Execution time: {execution_time:.2f}ms")
        
        if return_metrics:
            return path, self.metrics
        return path
    
    def shortest_path(self, 
                     graph: Dict[Any, List[Any]], 
                     start: Any, 
                     end: Any) -> Optional[List[Any]]:
        """
        Find shortest path between two nodes using BFS
        
        Args:
            graph: Graph represented as adjacency list
            start: Starting node
            end: Target node
            
        Returns:
            Shortest path as list of nodes, or None if no path exists
        """
        if start not in graph or end not in graph:
            self.logger.error("Start or end node not found in graph")
            return None
        
        if start == end:
            return [start]
        
        visited: Set[Any] = set()
        queue: deque = deque([start])
        visited.add(start)
        
        # Store parent pointers for path reconstruction
        parent: Dict[Any, Any] = {start: None}
        
        self.logger.info(f"Finding shortest path from {start} to {end}")
        
        while queue:
            node = queue.popleft()
            
            if node == end:
                # Reconstruct path
                path = []
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                self.logger.info(f"Shortest path found: {path}")
                return path
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        self.logger.warning(f"No path found from {start} to {end}")
        return None
    
    def find_connected_components(self, graph: Dict[Any, List[Any]]) -> List[List[Any]]:
        """
        Find all connected components in the graph
        
        Args:
            graph: Graph represented as adjacency list
            
        Returns:
            List of connected components, each as a list of nodes
        """
        visited: Set[Any] = set()
        components: List[List[Any]] = []
        
        self.logger.info("Finding connected components")
        
        for node in graph:
            if node not in visited:
                component = self.bfs(graph, node)
                components.append(component)
                visited.update(component)
        
        self.logger.info(f"Found {len(components)} connected components")
        return components
    
    def level_order_traversal(self, graph: Dict[Any, List[Any]], start: Any) -> Dict[int, List[Any]]:
        """
        Perform level-order traversal and group nodes by their level
        
        Args:
            graph: Graph represented as adjacency list
            start: Starting node
            
        Returns:
            Dictionary mapping levels to lists of nodes at that level
        """
        if start not in graph:
            raise ValueError(f"Start node {start} not found in graph")
        
        visited: Set[Any] = set()
        queue: deque = deque([(start, 0)])  # (node, level)
        visited.add(start)
        
        levels: Dict[int, List[Any]] = {}
        
        self.logger.info(f"Starting level-order traversal from {start}")
        
        while queue:
            node, level = queue.popleft()
            
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, level + 1))
        
        self.logger.info(f"Level-order traversal completed. Max level: {max(levels.keys())}")
        return levels


def create_sample_graph() -> Dict[str, List[str]]:
    """
    Create a sample graph for demonstration
    
    Returns:
        Sample graph represented as adjacency list
    """
    return {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F', 'G'],
        'D': ['H'],
        'E': [],
        'F': ['I'],
        'G': [],
        'H': [],
        'I': []
    }


def main():
    """
    Main function demonstrating industry-grade BFS features
    """
    print("=" * 80)
    print("INDUSTRY-GRADE BREADTH-FIRST SEARCH (BFS) ALGORITHM")
    print("=" * 80)
    
    # Initialize BFS with logging
    bfs = IndustryBFS(enable_logging=True, log_level=logging.INFO)
    
    # Create sample graph
    graph = create_sample_graph()
    
    print("\nGraph Structure:")
    for node, neighbors in graph.items():
        print(f"  {node} -> {neighbors}")
    
    # Basic BFS traversal
    print("\n" + "=" * 50)
    print("1. BASIC BFS TRAVERSAL")
    print("=" * 50)
    
    try:
        path, metrics = bfs.bfs(graph, 'A', return_metrics=True)
        print(f"BFS Path: {path}")
        print(f"Execution Metrics:")
        print(f"  - Nodes visited: {metrics.nodes_visited}")
        print(f"  - Edges explored: {metrics.edges_explored}")
        print(f"  - Execution time: {metrics.execution_time_ms:.2f} ms")
        print(f"  - Max queue size: {metrics.max_queue_size}")
        print(f"  - Path length: {metrics.path_length}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Shortest path finding
    print("\n" + "=" * 50)
    print("2. SHORTEST PATH FINDING")
    print("=" * 50)
    
    shortest_path = bfs.shortest_path(graph, 'A', 'I')
    if shortest_path:
        print(f"Shortest path from A to I: {shortest_path}")
        print(f"Path length: {len(shortest_path) - 1} edges")
    else:
        print("No path found from A to I")
    
    # Connected components
    print("\n" + "=" * 50)
    print("3. CONNECTED COMPONENTS")
    print("=" * 50)
    
    components = bfs.find_connected_components(graph)
    print(f"Connected components: {components}")
    print(f"Number of components: {len(components)}")
    
    # Level-order traversal
    print("\n" + "=" * 50)
    print("4. LEVEL-ORDER TRAVERSAL")
    print("=" * 50)
    
    levels = bfs.level_order_traversal(graph, 'A')
    print("Nodes by level:")
    for level, nodes in levels.items():
        print(f"  Level {level}: {nodes}")
    
    # Error handling demonstration
    print("\n" + "=" * 50)
    print("5. ERROR HANDLING DEMONSTRATION")
    print("=" * 50)
    
    try:
        bfs.bfs(graph, 'X')  # Invalid start node
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    try:
        bfs.bfs({}, 'A')  # Empty graph
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\n" + "=" * 80)
    print("BFS LAB DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

