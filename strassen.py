import math
import numpy as np
import unittest
from numpy.testing import assert_array_equal

def strassen(A, B):
    """
    Calculates the product of two square matrices, A and B, using the Strassen
    algorithm - which obtains an asymptotic complexity of n ** log_7(n), where
    n is the side length, via divide-and-conquer.
    A and B should be two-dimensional numpy arrays.

    Note that the core Strassen algorithm requires n to be a power of 2; this
    implementation pads the operands with zeroes to ensure that n is such, and
    then strips the extra rows/columns before returning the result. In practice
    it's often better to extend to a side length that has some factor which is a
    power of two, run a few iterations of Strassen until the result has an odd
    side length, and then do a direct product on the result. However, since this
    is a learning exercise I'll sacrifice efficiency for algorithmic purity, and
    just extend up to a power of two (thus allowing Strassen to run all the way
    to the base case).

    StackOverflow has some interesting observations on extension heuristics
    as well as techniques for handling rectangular matrices.
    https://cs.stackexchange.com/questions/92666/strassen-algorithm-for-unusal-matrices
    """
    _assert_is_square_matrix(A)
    _assert_is_square_matrix(B)
    assert len(A) == len(B), "Inputs must be of the same dimension"
    n_orig = len(A)

    # Pad A and B such that their side length is a power of two.
    n_new = _next_power_of_two(n_orig)
    A = _pad(A, n_new)
    B = _pad(B, n_new)

    return _strassen(A, B)[:n_orig,:n_orig]


def _strassen(A, B):
    n = len(A)
    if n == 1:
        # Since by requirement, A and B have side lengths that are
        # perfect powers of two, and are equal and square, it follows
        # that all sides of A and B are length one.
        # So we can simply do an integer multiplication here.
        return np.array([[A[0][0] * B[0][0]]])

    # A and B have side length strictly greater than one.
    # We proceed to subdivide each into four even quadrants,
    # in preparation to recurse into them.
    a, b, c, d = _quadrants(A)
    e, f, g, h = _quadrants(B)

    # We recursively complete seven products from which we can
    # derive the final result.
    p1 = _strassen(a, f - h)
    p2 = _strassen(a + b, h)
    p3 = _strassen(c + d, e)
    p4 = _strassen(d, g - e)
    p5 = _strassen(a + d, e + h)
    p6 = _strassen(b - d, g + h)
    p7 = _strassen(a - c, e + f)

    # Through the mad genius of Volker Strassen, we can combine
    # our seven intermediary values to obtain the product of A and B.
    k = int(n / 2)
    result = np.zeros((n, n))
    result[:k,:k] = p5 + p4 - p2 + p6  # Top left
    result[:k,k:] = p1 + p2            # Top right
    result[k:,:k] = p3 + p4            # Bottom left
    result[k:,k:] = p1 + p5 - p3 - p7  # Bottom right
    return result


def _assert_is_square_matrix(X):
    """
    Asserts that the given numpy array is a square matrix.
    """
    assert X.ndim == 2, "Inputs must be matrices"
    height, width = X.shape
    assert height == width, "Inputs must be square"


def _next_power_of_two(n):
    """
    If n is a power of two, returns n.
    Otherwise, returns the first power of two greater than n.
    """
    return 2 ** math.ceil(math.log2(n))


def _pad(X, side_length, value=0):
    """
    Given a square matrix X, adds rows and columns to the bottom and right
    with element value equal to 'value', such that the resulting side
    length is equal to 'side_length'.
    Throws an assertion error if the existing side length of X is greater
    than that requested.
    """
    _assert_is_square_matrix(X)
    result = np.full((side_length, side_length), value)
    result[:X.shape[0], :X.shape[1]] = X
    return result


def _quadrants(X):
    """
    Given a square matrix X whose side length is a multiple of two, returns
    four submatrixes a, b, c, d of equal dimension len(X)/2, such that
    a is the top-left submatrix of X, b is the top-right, c is the
    bottom-left, and d is the bottom-right.
    """
    k = int(len(X) / 2)
    return (X[:k,:k], X[:k,k:], X[k:,:k], X[k:,k:])


class TestNextPowerOfTwo(unittest.TestCase):
    def test_1(self):
        self.assertEqual(1, _next_power_of_two(1))

    def test_2(self):
        self.assertEqual(2, _next_power_of_two(2))

    def test_3(self):
        self.assertEqual(4, _next_power_of_two(3))

    def test_4(self):
        self.assertEqual(4, _next_power_of_two(4))

    def test_6(self):
        self.assertEqual(8, _next_power_of_two(6))

    def test_8(self):
        self.assertEqual(8, _next_power_of_two(8))

    def test_575(self):
        self.assertEqual(1024, _next_power_of_two(575))


class MatrixTest(unittest.TestCase):
    def assertMatricesEqual(self, A, B):
        msg = 'Expected matrices to be equal:\n\t%s\n\n\t%s' % (str(A), str(B))
        self.assertIsNone(assert_array_equal(A, B), msg=msg)


class TestPad(MatrixTest):
    def test_none_needed(self):
        original = np.eye(8)
        self.assertMatricesEqual(original, _pad(original, 8))

    def test_two_rowcols_added(self):
        original = np.eye(4)
        expected = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
        ])
        self.assertMatricesEqual(expected, _pad(original, 6))


class TestQuadrants(MatrixTest):
    def test_2x2_eye(self):
        original = np.eye(2)
        a_e = np.array([[1]])
        b_e = np.array([[0]])
        c_e = b_e
        d_e = a_e
        a, b, c, d = _quadrants(original)
        self.assertMatricesEqual(a_e, a)
        self.assertMatricesEqual(b_e, b)
        self.assertMatricesEqual(c_e, c)
        self.assertMatricesEqual(d_e, d)


    def test_4x4_eye(self):
        original = np.eye(4)
        a_e = np.eye(2)
        b_e = np.zeros((2, 2))
        c_e = b_e
        d_e = a_e
        a, b, c, d = _quadrants(original)
        self.assertMatricesEqual(a_e, a)
        self.assertMatricesEqual(b_e, b)
        self.assertMatricesEqual(c_e, c)
        self.assertMatricesEqual(d_e, d)


    def test_4x4_arbitrary(self):
        original = np.array([
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
            [41, 42, 43, 44]
        ])
        a_e = np.array([[11, 12], [21, 22]])
        b_e = np.array([[13, 14], [23, 24]])
        c_e = np.array([[31, 32], [41, 42]])
        d_e = np.array([[33, 34], [43, 44]])
        a, b, c, d = _quadrants(original)
        self.assertMatricesEqual(a_e, a)
        self.assertMatricesEqual(b_e, b)
        self.assertMatricesEqual(c_e, c)
        self.assertMatricesEqual(d_e, d)


class TestStrassen(MatrixTest):
    def test_singletons(self):
        A = np.array([[3]])
        B = np.array([[5]])
        self.assertMatricesEqual(np.array([[15]]), strassen(A, B))

    def test_2x2_identities(self):
        A = np.eye(2)
        B = np.eye(2)
        self.assertMatricesEqual(np.eye(2), strassen(A, B))

    def test_2x2_arbitrary(self):
        A = np.array([[7, 11], [14, 2]])
        B = np.array([[12, 2], [19, 2]])
        self.assertMatricesEqual(np.matmul(A, B), strassen(A, B))

    def test_6x6_identities(self):
        A = np.eye(6)
        B = np.eye(6)
        self.assertMatricesEqual(np.eye(6), strassen(A, B))

    def test_8x8_identities(self):
        A = np.eye(8)
        B = np.eye(8)
        self.assertMatricesEqual(np.eye(8), strassen(A, B))

    def test_8x8_arbitrary(self):
        A = np.array([
            [11, 12, 13, 14, 15, 16, 17, 18],
            [21, 22, 23, 24, 25, 26, 27, 28],
            [31, 32, 33, 34, 35, 36, 37, 38],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [51, 52, 53, 54, 55, 56, 57, 58],
            [61, 62, 63, 64, 65, 66, 67, 68],
            [71, 72, 73, 74, 75, 76, 77, 78],
            [81, 82, 83, 84, 85, 86, 87, 88]
        ])
        B = np.array([
            [17, 53, 61, 11, 35, 00, 35, 44],
            [21, 27, 52, 34, 88, 35, 14,  9],
            [11,  3, 14, 41, 13, 74, 88, 16],
            [14, 73, 67, 22, 86, 24, 26, 15],
            [87, 46, 66, 11, 14, 52, 84, 34],
            [90, 17, 42, 53, 57, 11, 89, 77],
            [ 1,  0, 73, 48, 73, 89,  4, 14],
            [32, 84, 61, 91,  4, 13,  9, 12]
        ])
        self.assertMatricesEqual(np.matmul(A, B), strassen(A, B))

    def test_random_matrices(self):
        for _ in range(100):
            n = np.random.randint(1, 32)
            A = np.random.randint(0, 100, size=(n, n))
            B = np.random.randint(0, 100, size=(n, n))
            self.assertMatricesEqual(np.matmul(A, B), strassen(A, B))


if __name__ == '__main__':
    unittest.main()
