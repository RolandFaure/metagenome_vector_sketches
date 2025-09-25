#!/usr/bin/env python3
"""
Advanced counting examples that demonstrate different loop patterns
that could be useful for genomic data processing and vector sketching.
"""

def simple_count_to_1000():
    """Simple for loop counting to 1000."""
    print("=== Simple count to 1000 ===")
    for i in range(1, 1001):
        if i <= 5 or i >= 996:  # Show first 5 and last 5
            print(f"Count: {i}")
        elif i == 6:
            print("... (counting continues) ...")
    print("Simple count completed!\n")

def while_loop_count():
    """While loop counting to 1000."""
    print("=== While loop count to 1000 ===")
    count = 1
    while count <= 1000:
        if count <= 5 or count >= 996:  # Show first 5 and last 5
            print(f"While count: {count}")
        elif count == 6:
            print("... (while loop continues) ...")
        count += 1
    print("While loop count completed!\n")

def step_counting():
    """Counting by steps (useful for processing genomic positions)."""
    print("=== Step counting (by 10s to 1000) ===")
    for i in range(10, 1001, 10):
        if i <= 50 or i >= 960:  # Show first 5 and last 5
            print(f"Step count: {i}")
        elif i == 60:
            print("... (step counting continues) ...")
    print("Step counting completed!\n")

def reverse_counting():
    """Reverse counting from 1000 to 1."""
    print("=== Reverse count from 1000 to 1 ===")
    for i in range(1000, 0, -1):
        if i >= 996 or i <= 5:  # Show first 5 and last 5
            print(f"Reverse count: {i}")
        elif i == 995:
            print("... (reverse counting continues) ...")
    print("Reverse counting completed!\n")

def enumerate_counting():
    """Using enumerate for counting (useful for processing sequences)."""
    print("=== Enumerate counting ===")
    data = list(range(1, 1001))
    for index, value in enumerate(data):
        if index < 5 or index >= 995:  # Show first 5 and last 5
            print(f"Index {index}: Value {value}")
        elif index == 5:
            print("... (enumerate continues) ...")
    print("Enumerate counting completed!\n")

if __name__ == "__main__":
    simple_count_to_1000()
    while_loop_count()
    step_counting()
    reverse_counting()
    enumerate_counting()
    print("All counting demonstrations completed!")