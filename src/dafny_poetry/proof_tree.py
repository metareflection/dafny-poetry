"""
Data structures for POETRY proof search tree.

Implements the ProofNode and SorryEdge structures described in the POETRY paper.
"""

import pathlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class NodeStatus(Enum):
    """Status of a proof node in the search tree."""
    OPEN = "OPEN"              # Not yet explored or partially explored
    PROVED = "PROVED"          # Complete proof found
    HALF_PROVED = "HALF_PROVED"  # Sketch found with sorry edges
    FAILED = "FAILED"          # All attempts exhausted


class SorryStatus(Enum):
    """Status of a sorry edge (sub-goal)."""
    OPEN = "OPEN"
    PROVED = "PROVED"
    FAILED = "FAILED"


@dataclass
class SorryEdge:
    """
    Represents a sub-goal that needs to be proved (from an Admit call).
    Links a parent node to a child node, with a new root for recursive search.
    """
    parent_node: 'ProofNode'
    child_node: 'ProofNode'
    admit_tag: str  # The tag from Admit("tag", ...)
    admit_line: int  # Line number of the Admit
    sub_goal_root: Optional['ProofNode'] = None  # Root of next level's search
    sub_goal_status: SorryStatus = SorryStatus.OPEN


@dataclass
class ProofNode:
    """
    Node in the proof search tree.
    Each node represents a state of the Dafny file with some Admits remaining.
    """
    # File state
    file_path: pathlib.Path  # Path to .dfy file at this state
    admits: int              # Number of Admit(...) remaining
    
    # Tree structure
    parent: Optional['ProofNode'] = None
    children: List['ProofNode'] = field(default_factory=list)
    
    # Search metadata
    status: NodeStatus = NodeStatus.OPEN
    action_taken: Optional[str] = None  # "induction", "llm", "seed"
    score: float = 0.0  # Cumulative log probability (higher is better)
    depth: int = 0      # Recursion level (0 = root theorem)
    
    # Sorry edges for sub-goals
    sorry_edges: List[SorryEdge] = field(default_factory=list)
    
    # Method focused on (for debugging/logging)
    focused_method: Optional[str] = None
    
    def __repr__(self):
        return (f"ProofNode(admits={self.admits}, status={self.status.value}, "
                f"score={self.score:.2f}, depth={self.depth}, action={self.action_taken})")
    
    def is_proved(self) -> bool:
        """Check if this node represents a complete proof (no admits)."""
        return self.admits == 0
    
    def has_unproved_sorry_edges(self) -> bool:
        """Check if this node has sorry edges that aren't yet proved."""
        return any(edge.sub_goal_status != SorryStatus.PROVED 
                   for edge in self.sorry_edges)
    
    def get_path_to_root(self) -> List['ProofNode']:
        """Return path from this node to root (inclusive)."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))


@dataclass
class SearchTree:
    """
    Manages the proof search tree at a single level.
    Provides operations for best-first search and status tracking.
    """
    root: ProofNode
    nodes: List[ProofNode] = field(default_factory=list)
    
    def __post_init__(self):
        self.nodes = [self.root]
    
    def add_node(self, node: ProofNode):
        """Add a node to the search tree."""
        self.nodes.append(node)
    
    def get_open_nodes(self) -> List[ProofNode]:
        """Get all nodes with OPEN status."""
        return [n for n in self.nodes if n.status == NodeStatus.OPEN]
    
    def get_best_open_node(self) -> Optional[ProofNode]:
        """
        Select the best OPEN node using best-first search.
        Returns node with highest cumulative score.
        """
        open_nodes = self.get_open_nodes()
        if not open_nodes:
            return None
        # Best-first: highest score
        return max(open_nodes, key=lambda n: n.score)
    
    def has_half_proved_path(self) -> bool:
        """
        Check if there's any node in HALF_PROVED status.
        This indicates a sketch has been found.
        """
        return any(n.status == NodeStatus.HALF_PROVED for n in self.nodes)
    
    def get_half_proved_nodes(self) -> List[ProofNode]:
        """Get all HALF_PROVED nodes (sketches with sorry edges)."""
        return [n for n in self.nodes if n.status == NodeStatus.HALF_PROVED]
    
    def get_first_unproved_sorry_edge(self) -> Optional[SorryEdge]:
        """
        Find the first sorry edge that needs to be proved.
        Used for greedy recursion - recurse on first unproved sub-goal found.
        """
        # Look through HALF_PROVED nodes and their paths to root
        for node in self.get_half_proved_nodes():
            path = node.get_path_to_root()
            for path_node in path:
                for edge in path_node.sorry_edges:
                    if edge.sub_goal_status == SorryStatus.OPEN:
                        return edge
        return None
    
    def get_best_node(self) -> ProofNode:
        """Get the best node overall (highest score, preferring PROVED)."""
        proved_nodes = [n for n in self.nodes if n.status == NodeStatus.PROVED]
        if proved_nodes:
            return max(proved_nodes, key=lambda n: n.score)
        
        half_proved_nodes = [n for n in self.nodes 
                            if n.status == NodeStatus.HALF_PROVED]
        if half_proved_nodes:
            return max(half_proved_nodes, key=lambda n: n.score)
        
        # Return best node by score
        return max(self.nodes, key=lambda n: n.score)

