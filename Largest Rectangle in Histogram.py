def largestRectangleArea(heights):
    stack = []  # Monotonic stack to store indices
    max_area = 0
    heights.append(0)  # Append a 0 to handle remaining elements in stack

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area

# Example Usage
heights = [2, 1, 5, 6, 2, 3]
print(largestRectangleArea(heights))  # Output
