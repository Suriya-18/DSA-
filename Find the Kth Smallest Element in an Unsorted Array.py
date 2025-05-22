import random

def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]
    
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    pivot_count = len(arr) - len(left) - len(right)
    
    if len(left) == k - 1:
        return pivot
    elif len(left) > k - 1:
        return quickselect(left, k)
    else:
        return quickselect(right, k - len(left) - pivot_count)

# Find the 3rd smallest element in the array
arr = [7, 10, 4, 3, 20, 15]
print(quickselect(arr, 3))  # Output: 7
