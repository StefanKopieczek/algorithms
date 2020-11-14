import random
import unittest


def karatsuba(x, y):
    """
    Multiplies nonnegative integers x and y using the Karatsuba multiplication algorithm,
    which uses a divide-and-conquer strategy, and has asymptotic complexity of n ** log_2(3),
    where n is the total number of digits in x and y.
    The approach is to split each operand in half, forming four groups: the uppermost and
    lowermost digits of each of x and y. Three products are computed recursively from these
    groups and then combined to form the final product.
    """
    assert x >= 0
    assert y >= 0
    assert x == int(x)
    assert y == int(y)
    return _karatsuba(x, y)


def _karatsuba(x, y):
    x_l = x.bit_length()
    y_l = y.bit_length()

    # Base case
    if x_l == 0 or y_l == 0:
        return 0
    if x_l == 1:
        return y
    if y_l == 1:
        return x

    # We're not in the base case, so we need to split x and y up and then recurse.
    # We treat x and y as being of equal length, padding with zeros as necessary.
    n = max(x_l, y_l)
    half_n = int(n/2)

    a, b = _split_int(x, half_n)
    c, d = _split_int(y, half_n)

    # Recursively compose the three terms we need to form the final product.
    ac = _karatsuba(a, c)
    bd = _karatsuba(b, d)
    zz = _karatsuba(a + b, c + d) - ac - bd

    return (ac << (half_n << 1)) + (zz << half_n) + bd


def _split_int(n, l):
    """
    Splits the given integer n into two integers a and b, such that b is composed
    of the lowermost l bits of n, and a is composed of the remaining uppermost bits,
    If l is greater than half of the length of n, a will be left-padded with zeroes.
    If l is greater than or equal to the length of n, a will be zero.
    """
    a = n >> l
    b = ((2 ** l) - 1) & n
    return (a, b)


class TestSplitInt(unittest.TestCase):
    def test_1(self):
        a,b = _split_int(0b10, 1)
        self.assertEqual(a, 0b1)
        self.assertEqual(b, 0b0)


    def test_2(self):
        a, b = _split_int(0b11110000, 4)
        self.assertEqual(a, 0b1111)
        self.assertEqual(b, 0b0000)

    def test_3(self):
        a, b = _split_int(0b101, 2)
        self.assertEqual(a, 0b1)
        self.assertEqual(b, 0b01)

    def test_4(self):
        a, b = _split_int(0b101, 1)
        self.assertEqual(a, 0b10)
        self.assertEqual(b, 0b1)

    def test_5(self):
        a, b = _split_int(0b101110001, 5)
        self.assertEqual(a, 0b1011)
        self.assertEqual(b, 0b10001)

    def test_6(self):
        a, b = _split_int(0b101110001, 4)
        self.assertEqual(a, 0b10111)
        self.assertEqual(b, 0b0001)

    def test_7(self):
        a, b = _split_int(0b11, 8)
        self.assertEqual(a, 0b0)
        self.assertEqual(b, 0b11)


class TestKaratsuba(unittest.TestCase):
    def test_both_one(self):
        self.assertEqual(karatsuba(1, 1), 1)

    def test_one_and_two(self):
        self.assertEqual(karatsuba(1, 2), 2)

    def test_two_and_one(self):
        self.assertEqual(karatsuba(2, 1), 2)

    def test_two_and_two(self):
        self.assertEqual(karatsuba(2, 2), 4)

    def test_three_and_four(self):
        self.assertEqual(karatsuba(3, 4), 12)

    def test_eleven_and_nine(self):
        self.assertEqual(karatsuba(11, 9), 99)

    def test_200_and_300(self):
        self.assertEqual(karatsuba(200, 300), 60000)

    def test_one_and_five(self):
        self.assertEqual(karatsuba(1, 5), 5)

    def test_many_random_cases(self):
        for _ in range(1000):
            x = random.randint(1, 10000)
            y = random.randint(1, 10000)
            expected = x * y
            actual = karatsuba(x, y)
            msg = 'Expected %d*%d=%d but got %d' % (x, y, expected, actual)
            self.assertEqual(expected, actual, msg=msg)


if __name__ == '__main__':
    unittest.main()

