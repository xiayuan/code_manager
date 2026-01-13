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
