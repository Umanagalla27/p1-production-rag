"""
LeetCode 125: Valid Palindrome (Easy)

Problem Statement:
A phrase is a palindrome if, after converting all uppercase letters into lowercase
letters and removing all non-alphanumeric characters, it reads the same forward and
backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Complexity:
- Time Complexity: O(N) where N is the length of s
- Space Complexity: O(1) auxiliary space (in-place two pointers)
"""


def is_palindrome(s: str) -> bool:
    left, right = 0, len(s) - 1

    while left < right:
        # Move left pointer past non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1

        # Move right pointer past non-alphanumeric characters
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters ignoring case
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True


if __name__ == "__main__":
    test_cases = [
        ("A man, a plan, a canal: Panama", True),
        ("race a car", False),
        (" ", True),
        ("0P", False),
        ("ab_a", True),
    ]

    for s, expected in test_cases:
        result = is_palindrome(s)
        assert result == expected, (
            f"Failed for '{s}': expected {expected}, got {result}"
        )
        print(f"PASS: '{s}' -> {result}")
