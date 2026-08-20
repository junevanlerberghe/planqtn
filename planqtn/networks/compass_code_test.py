from galois import GF2
import numpy as np
import itertools
import pytest
from planqtn.networks.compass_code import (
    CompassCodeConcatenateAndSparsifyTN,
    CompassCodeDualSurfaceCodeLayoutTN,
    CompassCodeQubitWiseTN
)
from planqtn.legos import Legos
from planqtn.poly import UnivariatePoly
from planqtn.stabilizer_tensor_enumerator import StabilizerCodeTensorEnumerator

### Helper functions for testing ###
def rref(matrix: np.ndarray) -> np.ndarray:
    """Row-reduced form over GF(2), zero rows dropped."""
    m = np.array(matrix, dtype=np.int8) % 2
    row = 0
    for col in range(m.shape[1]):
        pivot = next((r for r in range(row, m.shape[0]) if m[r, col]), None)
        if pivot is None:
            continue
        m[[row, pivot]] = m[[pivot, row]]
        for r in range(m.shape[0]):
            if r != row and m[r, col]:
                m[r] ^= m[row]
        row += 1
        if row == m.shape[0]:
            break
    return m[:row]
 
 
def parity_check(tn) -> np.ndarray:
    """Conjoined check matrix with columns in qubit order q = row + d * col."""
    legs = [tn.qubit_to_node_and_leg(q)[1] for q in range(tn.n_qubits())]
    node = tn.conjoin_nodes()
    returned = list(node.legs)
    # ensure the legs are in the same order as the legs in the tn for consistency between layouts
    pos = {leg: i for i, leg in enumerate(returned)}
    n = len(returned)
    cols = [pos[leg] for leg in legs] + [n + pos[leg] for leg in legs]
    return rref(np.array(node.h)[:, cols])

def colorings(d: int):
    """All 2**((d-1)**2) colorings of a d x d compass code."""
    for bits in itertools.product([1, 2], repeat=(d - 1) ** 2):
        yield np.array(bits).reshape(d - 1, d - 1)

### Tests ###

@pytest.mark.parametrize("coloring", list(colorings(3)), ids=str)
def test_constructions_give_the_same_code_d3(coloring):
    """Test for same stabilizer group, not just the same enumerator, for every d=3 coloring."""
    expected = parity_check(CompassCodeDualSurfaceCodeLayoutTN(coloring))
    tn = CompassCodeConcatenateAndSparsifyTN(coloring)
    assert np.array_equal(parity_check(tn), expected), f"Concat & Sparsify differs for coloring {coloring}"
    assert np.array_equal(parity_check(CompassCodeQubitWiseTN(coloring)), expected)


@pytest.mark.parametrize(
    "coloring",
    [
        np.full((4, 4), 2),  # every X-check keeps weight 2d, no carves at all
        np.full((4, 4), 1),  # fully cut: every block is one row, so no Z-merges at all
        np.array([[2, 2, 2, 2], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]]),  # 3-row blocks
        np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]]),  # d=5 rotated-surface-like
    ],
    ids=["shor", "fully_cut", "tall_blocks", "rotated_surface"],
)
def test_d5_edge_colorings(coloring):
    """Test some distance-5 colorings"""
    expected = parity_check(CompassCodeDualSurfaceCodeLayoutTN(coloring))
    assert np.array_equal(parity_check(CompassCodeConcatenateAndSparsifyTN(coloring)), expected)
    assert np.array_equal(parity_check(CompassCodeQubitWiseTN(coloring)), expected)

@pytest.mark.parametrize(
    "TNClass", [CompassCodeDualSurfaceCodeLayoutTN, CompassCodeConcatenateAndSparsifyTN, CompassCodeQubitWiseTN]
)
def test_compass_code(TNClass):
    tn = TNClass(
        [
            [1, 1],
            [2, 1],
        ]
    )

    tn_wep = tn.stabilizer_enumerator_polynomial(cotengra=False)
    expected_wep = StabilizerCodeTensorEnumerator(
        GF2(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
            ]
        )
    ).stabilizer_enumerator_polynomial()

    assert tn_wep == expected_wep

    tn_shor = TNClass(
        [
            [2, 2],
            [2, 2],
        ]
    )

    tn_wep = tn_shor.stabilizer_enumerator_polynomial(cotengra=False)
    expected_wep = StabilizerCodeTensorEnumerator(
        GF2(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            ]
        )
    ).stabilizer_enumerator_polynomial()

    assert tn_wep == expected_wep

    tn_rsc = TNClass(
        [
            [1, 2],
            [2, 1],
        ]
    )
    tn_rsc_wep = tn_rsc.stabilizer_enumerator_polynomial(cotengra=False)
    expected_wep = StabilizerCodeTensorEnumerator(
        GF2(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
            ]
        )
    ).stabilizer_enumerator_polynomial()

    assert tn_rsc_wep == expected_wep


def test_compass_code_z_coset_weight_enumerator_weight1():
    coloring = np.array(
        [
            [1, 2],
            [2, 1],
        ]
    )
    tn = CompassCodeDualSurfaceCodeLayoutTN(
        coloring,
        lego=lambda i: Legos.encoding_tensor_512_z,
        coset_error=GF2([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    )
    wep = tn.stabilizer_enumerator_polynomial(cotengra=False)
    assert wep == UnivariatePoly({5: 9, 3: 4, 7: 2, 1: 1}), f"Not equal, got:\n{wep}"


def test_compass_code_z_coset_weight_enumerator_weight2():
    coloring = np.array(
        [
            [1, 2],
            [2, 1],
        ]
    )
    tn = CompassCodeDualSurfaceCodeLayoutTN(
        coloring,
        lego=lambda i: Legos.encoding_tensor_512_z,
        coset_error=GF2([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
    )
    wep = tn.stabilizer_enumerator_polynomial(cotengra=False)
    assert wep == UnivariatePoly({4: 10, 6: 5, 2: 1}), f"Not equal, got:\n{wep}"


def test_compass_d3_rsc_z_coset():
    coloring = np.array(
        [
            [1, 2],
            [2, 1],
        ]
    )
    tn = CompassCodeDualSurfaceCodeLayoutTN(
        coloring,
        lego=lambda i: Legos.encoding_tensor_512_z,
        coset_error=((), (0, 5)),
    )

    we = tn.stabilizer_enumerator_polynomial(cotengra=False)
    print(we)
    assert we == UnivariatePoly(
        {
            2: 1,
            4: 10,
            6: 5,
        }
    )


def test_compass_truncated_coset_wep():
    coloring = np.array(
        [
            [1, 2],
            [2, 1],
        ]
    )
    tn = CompassCodeDualSurfaceCodeLayoutTN(
        coloring,
        lego=lambda i: Legos.encoding_tensor_512_z,
        coset_error=((), (0, 8)),
        truncate_length=2,
    )

    wep = tn.stabilizer_enumerator_polynomial(cotengra=False)

    tn.set_truncate_length(None)

    wep_full = tn.stabilizer_enumerator_polynomial(cotengra=False)
    assert (
        wep_full.dict[2] == wep.dict[2]
    ), f"Not equal, got: {wep} vs expected {wep_full}"

    # pytest.fail(f"Debug, got:\n{wep} vs {wep_full}")

    tn = CompassCodeDualSurfaceCodeLayoutTN(
        coloring,
        lego=lambda i: Legos.encoding_tensor_512_z,
        coset_error=((), (4,)),
        truncate_length=1,
    )
    wep = tn.stabilizer_enumerator_polynomial(verbose=True, cotengra=False)
    assert wep == UnivariatePoly({1: 1}), f"Not equal, got:\n{wep}"
