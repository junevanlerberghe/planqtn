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

        for col in range(len(coloring[0])):
            # blocks: qubit rows cut at every 1-plaquette in this column
            col_blocks, cur = [], [0]
            for r in range(len(coloring)):
                if coloring[r][col] == 1:
                    col_blocks.append(cur); cur = [r + 1]
                else:
                    cur.append(r + 1)
            col_blocks.append(cur)

            # a full-height block means the column-pair is uncarved (X check keeps
            # weight 2d). otherwise: k blocks -> k-1 carves. the remainder block's
            # X check is implied by the others times the original weight-2d check.
            col_blocks = [b for b in col_blocks if len(b) != d]
            col_blocks = col_blocks[:-1]        # [REASONING] Sec 3.2 says "carve out
                                                # of", but not which block is the
                                                # remainder. Choice is ours; documented.

            for block in col_blocks:
                block_size = len(block)
                if block_size > 1:
                    z_merge_key = ("z_merge", block[0], col)
                    nodes[z_merge_key] = StabilizerCodeTensorEnumerator(
                        Legos.z_rep_code(2 * block_size), tensor_id=z_merge_key
                    )
                for offset, j in enumerate(block):
                    # print(f"\t applying X non-isometry to qubit in row {j} in column {col}, {col+1}")
                    nodes[("x", j, col)] = StabilizerCodeTensorEnumerator(
                        Legos.x_rep_code(4), tensor_id=("x", j, col)
                    )
                    qubit1, leg1 = attachments[(j, col)]
                    qubit2, leg2 = attachments[(j, col + 1)]
                    connections_to_trace.add((qubit1, ("x", j, col), leg1, 2))
                    connections_to_trace.add((qubit2, ("x", j, col), leg2, 3))
                    if block_size > 1:
                        connections_to_trace.add((("x", j, col), z_merge_key, 0, offset))
                        attachments[(j, col + 1)] = (z_merge_key, offset + block_size)
                    else:
                        # print(f"\t m=1 X non-isometry so no z spider needed")
                        attachments[(j, col + 1)] = (("x", j, col), 0)
                    attachments[(j, col)] = (("x", j, col), 1)

        super().__init__(nodes, truncate_length=truncate_length)

        for leg in range(d):
            self.self_trace((0, 0), (1, leg), [leg], [d])

        for connection in connections_to_trace:
            self.self_trace(
                connection[0], connection[1], [connection[2]], [connection[3]]
            )

        # print("\t after construction, nodes are: ", self.nodes.keys())
        self.n = d * d
        self.d = d

        self.attachments = attachments
        self.set_coset(
            coset_error if coset_error is not None else GF2.Zeros(2 * self.n)
        )

    def qubit_to_node_and_leg(self, q: int) -> Tuple[TensorId, TensorLeg]:
        idx_leg = q % self.d
        idx_node = q // self.d
        node, leg = self.attachments[(idx_leg, idx_node)]
        return node, (node, leg)

    def n_qubits(self) -> int:
        return self.n

