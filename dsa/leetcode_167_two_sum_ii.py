"""
LeetCode 167: Two Sum II - Input Array Is Sorted (Medium)

Problem Statement:
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order,
find two numbers such that they add up to a specific target number.

Return the indices of the two numbers, index1 and index2, added by one as an integer array
[index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.
Your solution must use only constant extra space.

Complexity:
- Time Complexity: O(N) where N is the length of numbers
- Space Complexity: O(1) auxiliary space (two-pointer approach on sorted array)
"""


def two_sum(numbers: list[int], target: int) -> list[int]:
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]

        if current_sum == target:
            # 1-indexed result
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    raise ValueError("No two sum solution found")


if __name__ == "__main__":
    test_cases = [
        ([2, 7, 11, 15], 9, [1, 2]),
        ([2, 3, 4], 6, [1, 3]),
        ([-1, 0], -1, [1, 2]),
        ([1, 2, 3, 4, 4, 9, 56, 90], 8, [4, 5]),
    ]

    for nums, target, expected in test_cases:
        result = two_sum(nums, target)
        assert result == expected, (
            f"Failed for {nums} target {target}: expected {expected}, got {result}"
        )
        print(f"PASS: nums={nums}, target={target} -> {result}")
