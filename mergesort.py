import random
import unittest


def mergesort(l):
    """
    Sorts the given list in place using merge sort, returning a
    sorted copy of the list.

    Merge sort is a divide-and-conquer sorting algorithm whereby
    the given list is divided into two halves which are recursively
    mergesorted, and then combined together to form the final list.

    The mergesort algorithm has worst case asymptotic complexity of
    n * log_2(n), which is optimal for arbitrary sorting algorithms.

    However, the lower-order terms of mergesort are quite large in
    practice compared to algorithms such as quicksort, which in the
    average case tend to perform better.
    """

    if len(l) == 1:
        return l

    mid = int(len(l) / 2)
    a, b = l[:mid], l[mid:]
    a = mergesort(a)
    b = mergesort(b)
    return merge(a, b)


def merge(a, b):
    """
    Merges two sorted lists, returning a sorted list containing
    exactly the elements from both. This is done in linear
    time.
    """
    result = []
    while a and b:
        # Take the smaller of the heads of each list, and consume it.
        if a[0] <= b[0]:
            result.append(a[0])
            a = a[1:]
        else:
            result.append(b[0])
            b = b[1:]

    # If either list has elements left over, consume them.
    # At this stage, one list is guaranteed to be empty.
    if a:
        result.extend(a)
    else:
        result.extend(b)

    return result


class TestMerge(unittest.TestCase):
    def test_both_empty(self):
        self.assertEqual([], merge([], []))

    def test_left_empty(self):
        self.assertEqual([1, 2, 3], merge([], [1, 2, 3]))

    def test_right_empty(self):
        self.assertEqual([4, 5, 6], merge([4, 5, 6], []))

    def test_right_all_greater(self):
        self.assertEqual([1, 2, 3, 4, 5, 6], merge([1, 2, 3], [4, 5, 6]))

    def test_left_all_greater(self):
        self.assertEqual([1, 2, 3, 4, 5, 6], merge([4, 5, 6], [1, 2, 3]))

    def test_interleaved(self):
        self.assertEqual([1, 2, 3, 4, 5, 6], merge([1, 3, 5], [2, 4, 6]))

    def test_left_is_longer(self):
        self.assertEqual([1, 2, 3, 4, 5, 6], merge([1, 2, 3, 4, 6], [5]))

    def test_right_is_longer(self):
        self.assertEqual([1, 2, 3, 4, 5, 6], merge([4], [1, 2, 3, 5, 6]))

    def test_random_lists(self):
        for _ in range(1000):
            xs = list(range(random.randint(1, 100)))
            random.shuffle(xs)
            pivot = random.randint(1, 100)
            left = sorted(xs[:pivot])
            right = sorted(xs[pivot:])
            self.assertEqual(sorted(xs), merge(left, right))


class TestMergeSort(unittest.TestCase):
    def test_singleton(self):
        self.assertEqual([3], mergesort([3]))

    def test_sorted_pair(self):
        self.assertEqual([1, 2], mergesort([1, 2]))

    def test_reverse_pair(self):
        self.assertEqual([1, 2], mergesort([2, 1]))

    def test_sorted_triple(self):
        self.assertEqual([1, 2, 3], mergesort([1, 2, 3]))

    def test_reverse_triple(self):
        self.assertEqual([1, 2, 3], mergesort([3, 2, 1]))

    def test_skew_triple(self):
        self.assertEqual([1, 2, 3], mergesort([2, 3, 1]))

    def test_five_elements(self):
        self.assertEqual([1, 2, 3, 4, 5], mergesort([4, 1, 2, 5, 3]))

    def test_six_elements(self):
        self.assertEqual([1, 2, 3, 4, 5, 6], mergesort([1, 6, 4, 3, 5, 2]))

    def test_random_lists(self):
        for _ in range(1000):
            xs = list(range(random.randint(1, 100)))
            random.shuffle(xs)
            self.assertEqual(sorted(xs), mergesort(xs))


if __name__ == '__main__':
    unittest.main()
