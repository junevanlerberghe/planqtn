from planqtn.poly import BivariatePoly, UnivariatePoly


def test_normalizer_enumerator_polynomial_513():
    # the [[5,1,3]] code's normalizer enumerator polynomial
    n = 5
    k = 1
    polynomial = UnivariatePoly({0: 1, 4: 15})
    poly_b = polynomial.macwilliams_dual(n=n, k=k)

    assert poly_b == UnivariatePoly(
        {0: 1, 3: 30, 4: 15, 5: 18}
    ), f"{poly_b} is not equal to {UnivariatePoly({0: 1, 3: 30, 4: 15, 5: 18})}"

    poly_a = poly_b.macwilliams_dual(n=n, k=k, to_normalizer=False)
    assert poly_a == UnivariatePoly({0: 1, 4: 15}), f"{poly_a}"


def test_normalizer_enumerator_polynomial_422():
    # the [[4,2,2]] code's normalizer enumerator polynomial
    n = 4
    k = 2
    polynomial = UnivariatePoly({0: 1, 4: 3})
    poly_b = polynomial.macwilliams_dual(n=n, k=k)
    assert poly_b == UnivariatePoly(
        {0: 1, 2: 18, 3: 24, 4: 21}
    ), f"{poly_b} is not equal to {UnivariatePoly({0: 1, 2: 18, 3: 24, 4: 21})}"

    poly_a = poly_b.macwilliams_dual(n=n, k=k, to_normalizer=False)
    assert poly_a == UnivariatePoly({0: 1, 4: 3})


def test_normalizer_enumerator_polynomial_3x3RSC():
    n = 9
    k = 1
    stabilizer_polynomial = UnivariatePoly({0: 1, 2: 4, 4: 22, 6: 100, 8: 129})
    normalizer_polynomial = UnivariatePoly(
        {0: 1, 2: 4, 3: 24, 4: 22, 5: 192, 6: 100, 7: 408, 8: 129, 9: 144}
    )
    poly_b = stabilizer_polynomial.macwilliams_dual(n=n, k=k)
    assert (
        poly_b == normalizer_polynomial
    ), f"{poly_b} is not equal to {normalizer_polynomial}"

    poly_a = poly_b.macwilliams_dual(n=n, k=k, to_normalizer=False)
    assert (
        poly_a == stabilizer_polynomial
    ), f"{poly_a} is not equal to {stabilizer_polynomial}"


def test_bivariate_poly_edge_cases():
    """Test minw, leading_order_poly, and normalization for bivariate tuples."""
    # Polynomial with various (x, z) weights
    poly = BivariatePoly({
        (1, 2): 5,  # total 3, x=1
        (2, 1): 3,  # total 3, x=2
        (0, 4): 1,  # total 4, x=0
        (4, 0): 2   # total 4, x=4
    })
    
    # minw should evaluate total weight first (3), then tie-break with x-weight, 
    # so (1, 2) wins over (2, 1)
    assert poly.minw() == ((1, 2), 5)
    
    # leading_order_poly should return ALL terms sharing the minimum total weight
    leading = poly.leading_order_poly()
    assert leading == BivariatePoly({(1, 2): 5, (2, 1): 3})

    # Test normalization (should divide by the (0,0) coefficient if > 1)
    unnorm = BivariatePoly({(0, 0): 2, (2, 2): 4})
    assert unnorm.normalize() == BivariatePoly({(0, 0): 1, (2, 2): 2})
    
    # Test normalization ignoring polynomials with no (0,0) term
    no_identity = BivariatePoly({(2, 2): 4})
    assert no_identity.normalize() == no_identity