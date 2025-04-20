import heapq

def merge_k_sorted_arrays(arrays):
    min_heap = []
    result = []

    # Initialize the heap with the first element of each array
    for i, array in enumerate(arrays):
        if array:
            heapq.heappush(min_heap, (array[0], i, 0))  # (value, array index, element index)

    while min_heap:
        val, array_idx, element_idx = heapq.heappop(min_heap)
        result.append(val)

        # If the next element exists in the same array, push it to the heap
        if element_idx + 1 < len(arrays[array_idx]):
            next_val = arrays[array_idx][element_idx + 1]
            heapq.heappush(min_heap, (next_val, array_idx, element_idx + 1))

    return result

# Example usage:
arrays = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print(merge_k_sorted_arrays(arrays))
