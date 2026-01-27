"""
Binary Search Implementation
This module provides both iterative and recursive implementations of binary search algorithm.
"""

def binary_search_iterative(arr, target):
    """
    Performs binary search iteratively on a sorted array.

    Args:
        arr: A sorted list of elements
        target: The element to search for

    Returns:
        The index of target if found, otherwise -1

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Avoid potential overflow

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def binary_search_recursive(arr, target, left=None, right=None):
    """
    Performs binary search recursively on a sorted array.

    Args:
        arr: A sorted list of elements
        target: The element to search for
        left: Left boundary index (defaults to 0)
        right: Right boundary index (defaults to len(arr) - 1)

    Returns:
        The index of target if found, otherwise -1

    Time Complexity: O(log n)
    Space Complexity: O(log n) due to recursion stack
    """
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


def binary_search_leftmost(arr, target):
    """
    Finds the leftmost (first) occurrence of target in a sorted array.
    Useful when array contains duplicates.

    Args:
        arr: A sorted list of elements (may contain duplicates)
        target: The element to search for

    Returns:
        The index of the first occurrence of target if found, otherwise -1
    """
    left = 0
    right = len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching in left half
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result


def binary_search_rightmost(arr, target):
    """
    Finds the rightmost (last) occurrence of target in a sorted array.
    Useful when array contains duplicates.

    Args:
        arr: A sorted list of elements (may contain duplicates)
        target: The element to search for

    Returns:
        The index of the last occurrence of target if found, otherwise -1
    """
    left = 0
    right = len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching in right half
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result


def binary_search_sqrt(n, precision=1e-7):
    """
    使用二分查找计算非负数的平方根。

    Args:
        n: 非负数，要计算其平方根
        precision: 精度，默认为 1e-7

    Returns:
        n 的平方根（浮点数）

    Raises:
        ValueError: 如果 n 为负数

    Time Complexity: O(log(n / precision))
    Space Complexity: O(1)

    Example:
        >>> binary_search_sqrt(16)
        4.0
        >>> binary_search_sqrt(2)
        1.4142135...
    """
    if n < 0:
        raise ValueError("不能计算负数的平方根")

    if n == 0:
        return 0.0

    # 确定搜索范围
    # 当 n >= 1 时，平方根在 [1, n] 之间
    # 当 0 < n < 1 时，平方根在 [n, 1] 之间
    if n >= 1:
        left, right = 1.0, float(n)
    else:
        left, right = float(n), 1.0

    # 二分查找
    while right - left > precision:
        mid = left + (right - left) / 2

        if mid * mid < n:
            left = mid
        else:
            right = mid

    return left + (right - left) / 2


def binary_search_sqrt_integer(n):
    """
    使用二分查找计算非负整数的整数平方根（向下取整）。

    Args:
        n: 非负整数

    Returns:
        n 的整数平方根（向下取整）

    Raises:
        ValueError: 如果 n 为负数

    Time Complexity: O(log n)
    Space Complexity: O(1)

    Example:
        >>> binary_search_sqrt_integer(16)
        4
        >>> binary_search_sqrt_integer(10)
        3
    """
    if n < 0:
        raise ValueError("不能计算负数的平方根")

    if n == 0:
        return 0

    left, right = 1, n

    while left <= right:
        mid = left + (right - left) // 2

        if mid * mid == n:
            return mid
        elif mid * mid < n:
            left = mid + 1
        else:
            right = mid - 1

    # 返回向下取整的结果
    return right


if __name__ == "__main__":
    # Test cases
    test_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    test_array_with_duplicates = [1, 2, 2, 2, 3, 4, 5, 5, 5, 6]

    print("Binary Search Implementation Tests")
    print("=" * 50)

    # Test iterative binary search
    print("\nIterative Binary Search:")
    print(f"Array: {test_array}")
    print(f"Search for 7: Index = {binary_search_iterative(test_array, 7)}")
    print(f"Search for 15: Index = {binary_search_iterative(test_array, 15)}")
    print(f"Search for 1: Index = {binary_search_iterative(test_array, 1)}")
    print(f"Search for 19: Index = {binary_search_iterative(test_array, 19)}")
    print(f"Search for 10 (not found): Index = {binary_search_iterative(test_array, 10)}")

    # Test recursive binary search
    print("\nRecursive Binary Search:")
    print(f"Array: {test_array}")
    print(f"Search for 7: Index = {binary_search_recursive(test_array, 7)}")
    print(f"Search for 15: Index = {binary_search_recursive(test_array, 15)}")
    print(f"Search for 1: Index = {binary_search_recursive(test_array, 1)}")
    print(f"Search for 19: Index = {binary_search_recursive(test_array, 19)}")
    print(f"Search for 10 (not found): Index = {binary_search_recursive(test_array, 10)}")

    # Test with duplicates
    print("\nBinary Search with Duplicates:")
    print(f"Array: {test_array_with_duplicates}")
    print(f"Search for 2 (leftmost): Index = {binary_search_leftmost(test_array_with_duplicates, 2)}")
    print(f"Search for 2 (rightmost): Index = {binary_search_rightmost(test_array_with_duplicates, 2)}")
    print(f"Search for 5 (leftmost): Index = {binary_search_leftmost(test_array_with_duplicates, 5)}")
    print(f"Search for 5 (rightmost): Index = {binary_search_rightmost(test_array_with_duplicates, 5)}")

    # Edge cases
    print("\nEdge Cases:")
    print(f"Empty array: {binary_search_iterative([], 5)}")
    print(f"Single element (found): {binary_search_iterative([5], 5)}")
    print(f"Single element (not found): {binary_search_iterative([5], 3)}")

    # Test square root functions
    print("\n" + "=" * 50)
    print("Square Root Binary Search Tests")
    print("=" * 50)

    # Test floating point square root
    print("\nFloating Point Square Root:")
    print(f"sqrt(16) = {binary_search_sqrt(16)}")
    print(f"sqrt(2) = {binary_search_sqrt(2)}")
    print(f"sqrt(0.25) = {binary_search_sqrt(0.25)}")
    print(f"sqrt(0) = {binary_search_sqrt(0)}")
    print(f"sqrt(100) = {binary_search_sqrt(100)}")

    # Test integer square root
    print("\nInteger Square Root (floor):")
    print(f"isqrt(16) = {binary_search_sqrt_integer(16)}")
    print(f"isqrt(10) = {binary_search_sqrt_integer(10)}")
    print(f"isqrt(15) = {binary_search_sqrt_integer(15)}")
    print(f"isqrt(0) = {binary_search_sqrt_integer(0)}")
    print(f"isqrt(1) = {binary_search_sqrt_integer(1)}")
    print(f"isqrt(100) = {binary_search_sqrt_integer(100)}")
