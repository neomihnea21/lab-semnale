import numpy as np
from functools import cmp_to_key
import heapq

MAGIC = 8
# we precompute the pair list for zig-zag-vec
def compare_pixels(p1, p2):
    if p1[0]+p1[1] == p2[0] + p2[1]:
        parity = (p1[0] + p1[1]) %2
        if parity == 0:
            return p1[0] < p2[0]
        else:
           return p1[0] > p2[0]
    return p1[0]+p1[1] < p2[0] + p2[1]

#Python is weird, it wants functions to return more than a bool for sorting
def true_compare_pixels(p1, p2):
    if compare_pixels(p1, p2):
        return -1
    if compare_pixels(p2, p1):
        return 1
    return 0

def zigzag_pairs():
    pairs = [(i, j) for i in range(MAGIC) for j in range(MAGIC)]
    new_pairs = sorted(pairs, key=cmp_to_key(true_compare_pixels))
    return new_pairs

def encode_huffman(v):
    freq=dict()
    for x in v:
        if x in v:
            freq[x]+=1
        else:
            freq[x]=1
    sets = [(freq[x], x) for x in freq]