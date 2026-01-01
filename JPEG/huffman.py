import numpy as np
from functools import cmp_to_key
import heapq, pickle

MAGIC = 8
# first things first: let's build a Huffman Tree Structure
# this will be used to revive the bits
class HuffmanNode:
    def __init__(self, symbol, value):
        self.value = value 
        self.symbol  = symbol
        self.left  = None
        self.right = None
    def __lt__(self, other):
        return self.value < other.value
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

def get_codes(root, code, book):
    if root is None:
        return
    # the node is a "symbol"
    if root.left is None and root.right is None:
        book[root.symbol] = code
        return 
    # if not, there is no code
    get_codes(root.left, code+"0", book)
    get_codes(root.right, code+"1", book)

def encode_huffman(v):
    # basic preprocessing
    freq=dict()
    for x in v:
        if x in v:
            freq[x]+=1
        else:
            freq[x]=1
    sets = [HuffmanNode(x, freq[x]) for x in freq]
    n = len(sets)
    heap_sets = heapq.heapify(sets)
    
    for _ in range(n-1):
        e1 = heapq.heappop(heap_sets)
        e2 = heapq.heappop(heap_sets)
        # we push a "partial node", which has no symbol
        interNode = HuffmanNode(None, e1.value+e2.value)
        interNode.left = e1
        interNode.right = e2

        heapq.heappush(heap_sets, interNode)
    
    #now we have a Huffman Tree, let's commit it to disk - one way or another
    final_tree = heapq.heappop(heap_sets)
    

def save(path, tree):
    with open(path, 'wb') as out:
        pickle.dump(tree, out)

def load(path):
    with open(path, 'rb') as read:
        ans = pickle.load(path, read)
    return ans

def decode_huffman(bit_stream, root):
    index = 0
    ans = []
    N = len(bit_stream)
    current_node = root
    while(index < N):
        bit = bit_stream[index]
        if current_node.left is None and current_node.right is None:
            ans.append(current_node.symbol)
            current_node = root
        elif bit == "0":
            current_node = current_node.left
        elif bit == "1":
            current_node = current_node.right
    return ans 
