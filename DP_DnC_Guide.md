
# 📘 Dynamic Programming and Divide & Conquer Famous Problems

This guide provides detailed explanations and **line-by-line commented solutions** to 7 classic problems in algorithm design, covering:

- ✅ 4 Dynamic Programming Problems
- ✅ 3 Divide and Conquer Problems

---

## 🧠 Dynamic Programming Problems

---

### 1. 0/1 Knapsack Problem

**Problem:**  
Given `n` items with weights and values, find the maximum value that can fit in a knapsack of given `capacity`.

**Idea:**  
Use a bottom-up DP approach with a 1D array where `dp[w]` holds the max value for capacity `w`.

```python
def knapsack(weights, values, capacity):
    n = len(weights)  # Number of items
    dp = [0] * (capacity + 1)  # dp[w] = max value for weight w

    for i in range(n):  # Go through each item
        for w in range(capacity, weights[i] - 1, -1):  # Reverse loop to prevent overwrite
            # Choose the better between not taking and taking this item
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]  # Max value for full capacity

# Example:
weights = [1, 2, 3]
values = [10, 20, 30]
capacity = 4
print(f"Maximum Value: {knapsack(weights, values, capacity)}")
```

---

### 2. Travelling Salesman Problem (TSP)

**Problem:**  
Find the shortest possible route that visits each city once and returns to the origin city.

**Idea:**  
Use bitmasking and DP table `dp[mask][i]` = min cost to reach `i` having visited `mask`.

```python
import sys

def tsp(graph):
    n = len(graph)
    dp = [[sys.maxsize] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from city 0

    for mask in range(1, 1 << n):  # All subsets of cities
        for u in range(n):
            if not (mask & (1 << u)):  # u not in mask
                continue
            for v in range(n):
                if mask & (1 << v):  # v already visited
                    continue
                dp[mask | (1 << v)][v] = min(
                    dp[mask | (1 << v)][v],
                    dp[mask][u] + graph[u][v]
                )

    return min(dp[(1 << n) - 1][i] + graph[i][0] for i in range(1, n))

# Example:
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(f"Minimum Distance: {tsp(graph)}")
```

---

### 3. Coin Change Problem (Infinite Supply)

**Problem:**  
Given coin denominations and a target `amount`, return the **minimum number of coins** to make it.

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # No coins needed for amount = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Example:
coins = [1, 2, 5]
amount = 11
print(f"Minimum Coins: {coin_change(coins, amount)}")
```

---

### 4. Longest Increasing Subsequence (LIS)

**Problem:**  
Given an array, find the longest increasing subsequence length.

```python
def length_of_lis(nums):
    if not nums:
        return 0

    dp = [1] * len(nums)  # dp[i] = LIS ending at i

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Example:
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"Length of LIS: {length_of_lis(nums)}")
```

---

## 🧩 Divide and Conquer Problems

---

### 1. Merge Sort

**Problem:**  
Sort an array efficiently using divide and conquer.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # Sort left half
    right = merge_sort(arr[mid:])   # Sort right half

    return merge(left, right)       # Merge the two sorted halves

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

arr = [38, 27, 43, 3, 9, 82, 10]
print(f"Sorted Array: {merge_sort(arr)}")
```

---

### 2. Binary Search

**Problem:**  
Find index of an element in sorted array in O(log n).

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

arr = [1, 3, 5, 7, 9, 11]
target = 7
print(f"Element found at index: {binary_search(arr, target)}")
```

---

### 3. Matrix Chain Multiplication

**Problem:**  
Find optimal way to parenthesize matrix multiplications to minimize cost.

```python
def matrix_chain_order(dimensions):
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):  # chain length
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]

dimensions = [40, 20, 30, 10, 30]
print(f"Minimum number of multiplications: {matrix_chain_order(dimensions)}")
```

---


