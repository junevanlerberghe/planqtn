"""Singleton class that keeps track of number of operations during a 
tensor network contraction"""
class OperationTracker:
    def __init__(self):
        self.count = 0

    def increment(self, by=1):
        self.count += by

    def reset(self):
        self.count = 0

    def get(self):
        return self.count

# Create a single internal instance
operations_tracker = OperationTracker()

def get_tracker():
    return operations_tracker