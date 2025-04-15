class MinHeap:
    def __init__(self):
        self.heap = [] 

    def __len__(self):  # Get the size of the heap
        return len(self.heap)

    def __parent(self, i):  # Get the parent index
        return (i - 1) // 2

    def __left(self, i):  # Get the left child index
        return 2 * i + 1

    def __right(self, i):  # Get the right child index
        return 2 * i + 2

    def __swap(self, i, j):  # Swap two elements
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def __heapify_up(self, i):  # Restore min-heap property after insertion
        while i > 0 and self.heap[i] < self.heap[self.__parent(i)]:
            self.__swap(i, self.__parent(i))
            i = self.__parent(i)

    def __heapify_down(self, i):  # Restore min-heap property after extraction
        while True:
            smallest = i
            left = self.__left(i)
            right = self.__right(i)
            if left < len(self) and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < len(self) and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest != i:
                self.__swap(i, smallest)
                i = smallest
            else:
                break

    def insert(self, val):  # Insert a value into the heap
        self.heap.append(val)
        self.__heapify_up(len(self) - 1)

    def extract_min(self):  # Extract the minimum value from the heap
        if not self.heap:
            return None
        min_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self.__heapify_down(0)
        return min_val
