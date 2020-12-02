import math
import random
import unittest


def closest_pair(points):
    """
    Finds the closest pair of points within the given list (using Euclidean distance).
    The complexity is O(nlogn) - so surprisingly this isn't asymptotically more complex
    than just sorting the points.
    The strategy is divide and conquer, with a slightly involved combination step
    which is explained inline.
    """
    assert len(points) >= 2, "Finding a closest pair requires 2 or more elements!"
    assert not any(p is None for p in points)

    p, q, _ = _closest_pair(sorted(points, key=lambda pt: pt[0]))
    return (p, q)


def _closest_pair(points):
    """
    Applies the closest pair algorithm assuming that its input is sorted by x coordinate.
    Returns three arguments:
    - The leftmost point of the closest pair
    - The rightmost poing of the closest pair
    - The input points, sorted by y-coordinate (this is useful for the 'combine' step).
    """
    if len(points) == 2:
        # Normal base case.
        p, q = tuple(points)
        return p, q, sorted(points, key=lambda pt: pt[1])
    elif len(points) == 1:
        # Pathological base case that can happen if we recurse into a three element
        # list (dividing it into a singleton and a duple).
        # Return sentinel values.
        return None, None, points

    # Split the inputs into left- and right- halves.
    # Recurse into each to find the closest pair in each half.
    midpoint = int(len(points) / 2)
    left_points = points[:midpoint]
    right_points = points[midpoint:]
    p_left, q_left, y_sorted_left = _closest_pair(left_points)
    p_right, q_right, y_sorted_right = _closest_pair(right_points)

    # Calculate the distances d_left and d_right between the closest points in
    # each half.
    if p_left is None:
        # Handle the case where left_points is a singleton - we handle this
        # as though the best distance in the left list was infinite, so that it
        # is disregarded.
        d_left = math.inf
    else:
        d_left = distance(p_left, q_left)
    if p_right is None:
        # As above
        d_right = math.inf
    else:
        d_right = distance(p_right, q_right)

    # Choose the best distance and pair from those two possibilities.
    if d_left <= d_right:
        best_d = d_left
        best_p, best_q = p_left, q_left
    else:
        best_d = d_right
        best_p, best_q = p_right, q_right

    y_sorted = merge(y_sorted_left, y_sorted_right)

    # We now need to check if there are any pairs whose leftmost element is in
    # left_points and whose rightmost element is in right_points, and which are
    # closer together than 'best_d'.
    # It is sufficient to do this in a strip of width d*2 centered about the median
    # of the list. We build that strip in ascending order of y coordinate.
    median = points[midpoint][0]
    strip = [p for p in y_sorted if median - best_d <= p[0] <= median + best_d]
    for idx, q1 in enumerate(strip):
        for q2 in strip[idx+1:idx+8]:
            d = distance(q1, q2)
            if d < best_d:
                best_d = d
                if q1[0] <= q2[0]:
                    best_p, best_q = q1, q2
                else:
                    best_p, best_q = q2, q1

    return best_p, best_q, y_sorted



def merge(a, b):
    """
    Merges two sorted lists into a combined sorted list, in linear time.
    Borrowed from mergesort.py in this same repo.
    This version assumes the lists are duples, sorted by the second element.
    """
    result = []
    while a and b:
        # Take the smaller of the heads of each list, and consume it.
        if a[0][1] <= b[0][1]:
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


def distance(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


class TestClosestPair(unittest.TestCase):
    def test_four_elements(self):
        points = [(-4, 76), (37, -15), (59, -4), (94, 88)]
        actual_p, actual_q = closest_pair(points)
        expected_p = (37, -15)
        expected_q = (59, -4)
        self.assertEqual((expected_p, expected_q), (actual_p, actual_q))

    def test_seven_elements(self):
        points = [(-82, -50), (-64, -53), (-37, 8), (19, -81), (69, -29), (91, 80), (98, -76)]
        actual_p, actual_q = closest_pair(points)
        expected_p = (-82, -50)
        expected_q = (-64, -53)
        self.assertEqual((expected_p, expected_q), (actual_p, actual_q))

    def test_random(self):
        for attempt in range(1000):
            n = random.randint(2, 50)
            points = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(n)]
            actual_p, actual_q =  closest_pair(points)
            expected_p, expected_q, best_d = None, None, math.inf
            points.sort()
            for i, p in enumerate(points):
                for j, q in enumerate(points[i+1:]):
                    if distance(p, q) < best_d:
                        best_d = distance(p, q)
            msg = 'Failed on attempt %d: Expected %s, %s but got %s, %s for input %s' % (
                    attempt + 1,
                    str(expected_p), str(expected_q),
                    str(actual_p), str(actual_q),
                    str(points))
            self.assertEqual(best_d, distance(actual_p, actual_q), msg=msg)


if __name__ == '__main__':
    unittest.main()
