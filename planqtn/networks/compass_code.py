"""The `compass_code` module.

It contains the `CompassCodeDualSurfaceCodeLayoutTN` class, which implements a tensor network
representation of compass codes using dual surface code layout.
"""

from typing import Callable, Dict, Optional, Set, Tuple
from galois import GF2
import numpy as np
from planqtn.legos import Legos
from planqtn.networks.surface_code import SurfaceCodeTN
from planqtn.stabilizer_tensor_enumerator import (
    StabilizerCodeTensorEnumerator,
    TensorLeg,
)
from planqtn.tensor_network import TensorId, TensorNetwork


class CompassCodeDualSurfaceCodeLayoutTN(SurfaceCodeTN):
    """A tensor network representation of compass codes using dual surface code layout.

    This class implements a compass code using the dual doubled surface code equivalence
    described by Cao & Lackey in the expansion pack paper. The compass code is constructed
    by applying gauge operations to a surface code based on a coloring pattern.

    Args:
        coloring: Array specifying the coloring pattern for the compass code.
        lego: Function that returns the lego tensor for each node.
        coset_error: Optional coset error for weight enumerator calculations.
        truncate_length: Optional maximum weight for truncating enumerators.
    """

    def __init__(
        self,
        coloring: np.ndarray,
        *,
        lego: Callable[[TensorId], GF2] = lambda node: Legos.encoding_tensor_512,
        coset_error: Optional[GF2] = None,
        truncate_length: Optional[int] = None,
    ):
        """Create a square compass code based on the coloring.

        Creates a compass code using the dual doubled surface code equivalence
        described by Cao & Lackey in the expansion pack paper.

        Args:
            coloring: Array specifying the coloring pattern for the compass code.
            lego: Function that returns the lego tensor for each node.
            coset_error: Optional coset error for weight enumerator calculations.
            truncate_length: Optional maximum weight for truncating enumerators.
        """
        # See d3_compass_code_numbering.png for numbering - for an (r,c) qubit in the compass code,
        # the (2r, 2c) is the coordinate of the lego in the dual surface code.
        d = len(coloring) + 1
        super().__init__(d=d, lego=lego, truncate_length=truncate_length)
        gauge_idxs = [
            (r, c) for r in range(1, 2 * d - 1, 2) for c in range(1, 2 * d - 1, 2)
        ]
        for tensor_id, color in zip(gauge_idxs, np.reshape(coloring, (d - 1) ** 2)):
            self.nodes[tensor_id] = self.nodes[tensor_id].trace_with_stopper(
                Legos.stopper_z if color == 2 else Legos.stopper_x, 4
            )

        self._q_to_node = [(2 * r, 2 * c) for c in range(d) for r in range(d)]
        self.n = d * d
        self.coloring = coloring

        self.set_coset(
            coset_error if coset_error is not None else GF2.Zeros(2 * self.n)
        )


class CompassCodeConcatenateAndSparsifyTN(TensorNetwork):
    """A tensor network representation of compass codes using concatenate and sparsify method.

    This class implements a compass code using a concatenate and sparsify method described by
    Cao & Lackey in the following paper. The compass code is constructed by applying
    non-isometric tensors to "carve out" the desired stabilizers starting from a
    Bacon-Shor code.

    Cao, C., & Lackey, B. (2025). Growing sparse quantum codes from a seed.
    arXiv. https://arxiv.org/abs/2507.13496
    """

    def __init__(
        self,
        coloring: np.ndarray,
        *,
        coset_error: Optional[GF2] = None,
        truncate_length: Optional[int] = None,
    ):
        """Create a square compass code based on the coloring using the concatenate
        and sparsity method.

        Args:
            coloring: Array specifying the coloring pattern for the compass code.
            coset_error: Optional coset error for weight enumerator calculations.
            truncate_length: Optional maximum weight for truncating enumerators.
        """
        d = len(coloring) + 1
        nodes: Dict[TensorId, StabilizerCodeTensorEnumerator] = {}
        attachments: Dict[TensorId, Tuple[TensorId, int]] = {}

        # Start with the base layer of Z and X repetition codes which forms the Bacon-Shor code
        nodes[(0, 0)] = StabilizerCodeTensorEnumerator(
            Legos.x_rep_code(d + 1), tensor_id=(0, 0)
        )

        for c in range(d):
            nodes[(1, c)] = StabilizerCodeTensorEnumerator(
                Legos.z_rep_code(d + 1), tensor_id=(1, c)
            )
            for leg in range(d):
                attachments[(leg, c)] = ((1, c), leg)

        nodes[(0, 0)] = nodes[(0, 0)].trace_with_stopper(Legos.stopper_i, d)

        connections_to_trace: Set[Tuple[TensorId, TensorId, int, int]] = set()
        trace_with_stopper: Set[TensorId] = set()

        # Iterate over each column to apply non-isometries based on the coloring
        for col in range(len(coloring[0])):
            # Skip this column if there are no stabilizers to carve out
            if not any(coloring[row][col] == 1 for row in range(len(coloring))):
                continue

            row = 0
            while row < len(coloring):
                if coloring[row][col] == 2:
                    start_row = row
                    while row + 1 < len(coloring) and coloring[row + 1][col] == 2:
                        row += 1
                    end_row = row + 1
                    block_size = end_row - start_row + 1
                    last_zero_row = start_row - 1
                    next_one = next(
                        (
                            r
                            for r in range(end_row + 1, len(coloring))
                            if coloring[r][col] == 1
                        ),
                        len(coloring) + 1,
                    )

                    gap_above = max(0, start_row - (last_zero_row + 1))
                    gap_below = max(0, next_one - end_row - 1)
                    if gap_above <= gap_below:
                        # Merge upward (use rows from start_row to end_row)
                        z_merge_key = ("z_merge", start_row, col)
                        nodes[z_merge_key] = StabilizerCodeTensorEnumerator(
                            Legos.z_rep_code(block_size), tensor_id=z_merge_key
                        )

                        for offset, j in enumerate(range(start_row, end_row + 1)):
                            self._make_non_isometric_tensor(nodes, j, col)
                            self._connect_non_isometric_tensor(
                                j,
                                col,
                                z_merge_key,
                                offset,
                                attachments,
                                connections_to_trace,
                            )

                    else:
                        extra_rows = next_one - (end_row + 1)
                        z_merge_key = ("z_merge", end_row + 1, col)
                        nodes[z_merge_key] = StabilizerCodeTensorEnumerator(
                            Legos.z_rep_code(extra_rows), tensor_id=z_merge_key
                        )

                        for offset, j in enumerate(range(end_row + 1, col)):
                            self._make_non_isometric_tensor(nodes, j, col)
                            self._connect_non_isometric_tensor(
                                j,
                                col,
                                z_merge_key,
                                offset,
                                attachments,
                                connections_to_trace,
                            )

                row += 1

            top_rows, bottom_rows = [], []

            # Find contiguous top block of 1s
            row = 0
            while row < len(coloring) and coloring[row][col] == 1:
                top_rows.append(row)
                row += 1

            # Find contiguous bottom block of 1s
            row = len(coloring) - 1
            while row >= 0 and coloring[row][col] == 1:
                bottom_rows.append(row + 1)
                row -= 1

            bottom_rows = list(reversed(bottom_rows))  # ensure increasing order

            # Avoid duplication if full column is 1s
            full_column_ones = len(top_rows) + len(bottom_rows) > len(coloring)
            if full_column_ones:
                # Only apply from the top to avoid duplication
                bottom_rows = []
                if len(top_rows) > 1:
                    top_rows.append(top_rows[-1] + 1)

            # Apply non-isometry at top rows
            for label, rows in [("top", top_rows), ("bottom", bottom_rows)]:
                for r in rows:
                    print(f"adding non-isometry at col {col}, row {r} {label}")
                    self._make_non_isometric_tensor(nodes, r, col)
                    self._connect_non_isometric_tensor(
                        r, col, None, None, attachments, connections_to_trace
                    )
                    trace_with_stopper.add(("z", r, col))

        super().__init__(nodes, truncate_length=truncate_length)

        for leg in range(d):
            self.self_trace((0, 0), (1, leg), [leg], [d])

        for connection in connections_to_trace:
            self.self_trace(
                connection[0], connection[1], [connection[2]], [connection[3]]
            )

        for node in trace_with_stopper:
            self.nodes[node] = self.nodes[node].trace_with_stopper(Legos.stopper_x, 2)

        self.n = d * d
        self.d = d

        self.attachments = attachments
        self.set_coset(
            coset_error if coset_error is not None else GF2.Zeros(2 * self.n)
        )

    def _connect_non_isometric_tensor(
        self,
        row: int,
        col: int,
        z_merge_key: Optional[Tuple[str, int, int]],
        offset: Optional[int],
        attachments: Dict[TensorId, Tuple[TensorId, int]],
        connections_to_trace: Set[Tuple[TensorId, TensorId, int, int]],
    ) -> None:
        connections_to_trace.add((("x1", row, col), ("z", row, col), 0, 1))
        connections_to_trace.add((("z", row, col), ("x2", row, col), 0, 1))

        qubit1, leg1 = attachments[(row, col)]
        qubit2, leg2 = attachments[(row, col + 1)]

        connections_to_trace.add((qubit1, ("x1", row, col), leg1, 2))
        connections_to_trace.add((qubit2, ("x2", row, col), leg2, 2))

        attachments[(row, col)] = (("x1", row, col), 1)
        attachments[(row, col + 1)] = (("x2", row, col), 0)

        if z_merge_key is not None and offset is not None:
            connections_to_trace.add((("z", row, col), z_merge_key, 2, offset))

    def _make_non_isometric_tensor(
        self, nodes: Dict[TensorId, StabilizerCodeTensorEnumerator], row: int, col: int
    ) -> None:
        nodes[("x1", row, col)] = StabilizerCodeTensorEnumerator(
            Legos.x_rep_code(3), tensor_id=("x1", row, col)
        )
        nodes[("z", row, col)] = StabilizerCodeTensorEnumerator(
            Legos.z_rep_code(3), tensor_id=("z", row, col)
        )
        nodes[("x2", row, col)] = StabilizerCodeTensorEnumerator(
            Legos.x_rep_code(3), tensor_id=("x2", row, col)
        )

    def qubit_to_node_and_leg(self, q: int) -> Tuple[TensorId, TensorLeg]:
        idx_leg = q % self.d
        idx_node = q // self.d
        node, leg = self.attachments[(idx_leg, idx_node)]
        return node, (node, leg)

    def n_qubits(self) -> int:
        return self.n

Node = Tuple[int, int]
Connection = Tuple[Node, Node, int, int]
Stopper = Tuple[Node, int, str]

class CompassCodeQubitWiseTN(TensorNetwork):
    def __init__(
        self,
        coloring: list,
        *,
        lego=lambda node: Legos.encoding_tensor_512,
        coset_error: GF2 = None,
        truncate_length: int = None
    ):
        """Creates a square compass code based on the connections.

        Uses the rotated surface code layout.
        """
        self.coloring = coloring
        self.d = len(coloring) + 1
        self.n = self.d * self.d

        custom_connections, stoppers_to_add = self.make_custom_connections()

        nodes = {
            (r, c): StabilizerCodeTensorEnumerator(
                lego((r, c)),
                tensor_id=(r, c),
            )
            # col major ordering
            for r in range(self.d)
            for c in range(self.d)
        }

        last_row = self.d - 1
        last_col = self.d - 1

        for node, leg, stopper_type in stoppers_to_add:
            if(stopper_type == "x"):
                nodes[node] = nodes[node].trace_with_stopper(Legos.stopper_x, leg)
            elif(stopper_type == "z"):
                nodes[node] = nodes[node].trace_with_stopper(Legos.stopper_z, leg)

        # Apply stoppers to corners
        open_legs = self.find_open_legs(custom_connections, stoppers_to_add)
        top_left_open = open_legs.get((0,0))
        nodes[(0, 0)] = (
            nodes[(0, 0)]
            .trace_with_stopper(Legos.stopper_z, 2 if 2 in top_left_open else 3)
            .trace_with_stopper(Legos.stopper_x, 3 if 2 in top_left_open else 0)
        )


        top_right_open = open_legs.get((0, last_col))
        nodes[(0, last_col)] = (
            nodes[(0, last_col)]
            .trace_with_stopper(Legos.stopper_z, 1 if 1 in top_right_open else 0)
            .trace_with_stopper(Legos.stopper_x, 0 if 1 in top_right_open else 3)
        )

        bottom_left_open = open_legs.get((last_row, 0))
        nodes[(last_row, 0)] = (
            nodes[(last_row, 0)]
            .trace_with_stopper(Legos.stopper_z, 3 if 3 in bottom_left_open else 2)
            .trace_with_stopper(Legos.stopper_x, 2 if 3 in bottom_left_open else 1)
        )

        bottom_right_open = open_legs.get((last_row, last_col))
        nodes[(last_row, last_col)] = (
            nodes[(last_row, last_col)]
            .trace_with_stopper(Legos.stopper_z, 0 if 0 in bottom_right_open else 1)
            .trace_with_stopper(Legos.stopper_x, 1 if 0 in bottom_right_open else 2)
        )

        # Apply stoppers to sides
        for c in range(1, last_col):
            top_open = open_legs.get((0, c))
            for leg in top_open:
                nodes[(0, c)] = (
                    nodes[(0, c)]
                    .trace_with_stopper(Legos.stopper_x, leg)
                )
            bottom_open = open_legs.get((last_row, c))
            for leg in bottom_open:
                nodes[(last_row, c)] = (
                    nodes[(last_row, c)]
                    .trace_with_stopper(Legos.stopper_x, leg)
                )
        
        for r in range(1, last_row):
            left_open = open_legs.get((r, 0))
            for leg in left_open:
                nodes[(r, 0)] = (
                    nodes[(r, 0)]
                    .trace_with_stopper(Legos.stopper_z, leg)
                )

            right_open = open_legs.get((r, last_col))
            for leg in right_open:
                nodes[(r, last_col)] = (
                    nodes[(r, last_col)]
                    .trace_with_stopper(Legos.stopper_z, leg)
                )


        super().__init__(nodes, truncate_length=truncate_length)

        for node_a, node_b, leg_a, leg_b in custom_connections:
            self.self_trace(node_a, node_b, [leg_a], [leg_b])

        self.set_coset(coset_error=coset_error)
    
    def qubit_to_node_and_leg(self, q):
        # col major ordering
        node = (q % self.d, q // self.d)
        return node, (node, 4)
    
    def node_to_qubit(self, node):
        # inverse of qubit_to_node_and_leg's col-major node mapping
        row, col = node
        return row + self.d * col

    def n_qubits(self):
        return self.n
    
    def find_open_legs(self, connections, stoppers_to_add):
        used_legs = {}

        for coord1, coord2, leg1, leg2 in connections:
            if coord1 not in used_legs:
                used_legs[coord1] = set()
            if coord2 not in used_legs:
                used_legs[coord2] = set()
            
            used_legs[coord1].add(leg1)
            used_legs[coord2].add(leg2)

        for node, leg, stopper_type in stoppers_to_add:
            used_legs[node].add(leg)

        all_legs = set(range(4))
        open_legs = {coord: all_legs - legs for coord, legs in used_legs.items()}
        return open_legs

    def make_custom_connections(self) -> Tuple[Set[Connection], Set[Stopper]]:
        """Build the connection set for this coloring, plus the boundary stoppers."""
        connections = self._shor_connections()
        self._apply_coloring(connections)
        stoppers = self._cut_boundary_connections(connections)
        return connections, stoppers
    
    def _shor_connections(self) -> Set[Connection]:
        connections = set()

        for row in range(self.d):
            for col in range(self.d):
                if(col < self.d - 1):
                    if(row == 0):
                        connections.add(((row, col), (row, col + 1), 0, 3))
                    if(row == self.d - 1):
                        connections.add(((row, col), (row, col + 1), 1, 2))
                if(row < self.d - 1):
                    if(col < self.d - 1):
                        connections.add(((row, col), (row + 1, col), 1, 0))
                    if(col > 0):
                        connections.add(((row, col), (row + 1, col), 2, 3))
        return connections

    def _apply_coloring(self, connections) -> None:
        # now change connections to adjust for the specific coloring
        for row in range(self.d - 1):
            for col in range(self.d - 1):
                if(self.coloring[row][col] != 1):
                    continue
                top_left = (row, col)
                top_right = (row, col + 1)
                bottom_left = (row + 1, col)
                bottom_right = (row + 1, col + 1)

                connections.remove((top_left, bottom_left, 1, 0))
                connections.remove((top_right, bottom_right, 2, 3))

                if(col == self.d - 2):
                    connections.add((top_right, bottom_right, 1, 0))
                if(col == 0):
                    connections.add((top_left, bottom_left, 2, 3))
                if(row == 0):
                    connections.remove((top_left, top_right, 0, 3))
                if(row == self.d - 2):
                    connections.remove((bottom_left, bottom_right, 1, 2))

                connections.add((top_left, top_right, 1, 2))
                connections.add((bottom_left, bottom_right, 0, 3))

    def _cut_boundary_connections(self, connections) -> Set[Stopper]:
        stoppers = set()

        def is_col_all(col_idx, value):
            return all(row[col_idx] == value for row in self.coloring)

        def is_row_all(row_idx, value):
            return all(entry == value for entry in self.coloring[row_idx])

        def cut(node_a: Node, node_b: Node, stopper_type: str) -> None:
            """Remove any connection between these two legos, in either orientation."""
            for connection in list(connections):
                first, second, leg_first, leg_second = connection
                if {first, second} == {node_a, node_b}:
                    connections.remove(connection)
                    stoppers.add((first, leg_first, stopper_type))
                    stoppers.add((second, leg_second, stopper_type))

        for row in range(self.d - 1):
            for col in range(self.d - 1):
                color = self.coloring[row][col]
                top_left = (row, col)
                top_right = (row, col + 1)
                bottom_left = (row + 1, col)
                bottom_right = (row + 1, col + 1)
 
                if row == 0 and color == 2:
                    cut(top_left, top_right, "x")
                elif (
                    row == self.d - 2
                    and color == 2  
                    and not is_col_all(col, 2)
                ):
                    cut(bottom_left, bottom_right, "x")
 
                if col == 0 and color == 1:
                    cut(top_left, bottom_left, "z")
                if (
                    col == self.d - 2
                    and color == 1
                    and not is_row_all(row, 1)
                ):
                    cut(top_right, bottom_right, "z")
 
        return stoppers

    def with_coset_flipped_legs(
        self, coset_flipped_legs
    ) -> "CompassCodeRotatedTN":
        """Create a new tensor enumerator with coset-flipped legs."""
        return CompassCodeRotatedTN(
            self.coloring,
            coset_error=coset_flipped_legs
        )
