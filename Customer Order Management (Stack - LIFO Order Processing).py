"""
Scenario:
Manage an online shopping order system where orders are processed in Last-In-First-Out (LIFO) manner.

Problem:
Implement a stack-based order processing system.
"""

class OrderStack:
    def __init__(self):
        self.stack = []

    def add_order(self, order):
        self.stack.append(order)

    def process_order(self):
        return self.stack.pop() if self.stack else "No Orders"

# Test case
orders = OrderStack()
orders.add_order("Order 1")
orders.add_order("Order 2")
print(orders.process_order())  # "Order 2"
print(orders.process_order())  # "Order 1"
