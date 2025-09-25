#!/usr/bin/env python3
"""
A simple script that demonstrates counting to 1000.
This could be useful for iterating over genomic data positions or sketches.
"""

def count_to_1000():
    """
    Count from 1 to 1000 and print each number.
    This demonstrates a basic loop structure that could be adapted
    for processing genomic data or vector sketches.
    """
    print("Starting count to 1000...")
    
    for i in range(1, 1001):
        print(f"Count: {i}")
    
    print("Finished counting to 1000!")

if __name__ == "__main__":
    count_to_1000()