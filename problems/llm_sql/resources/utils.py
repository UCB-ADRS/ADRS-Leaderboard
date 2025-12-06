from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import sys
from typing import List, Tuple

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_word = True

    def longest_common_prefix(self, word):
        node = self.root
        common_prefix_length = 0
        for char in word:
            if char in node.children:
                common_prefix_length += len(char)
                node = node.children[char]
            else:
                break
        return common_prefix_length

def calculate_length(value):
    val = 0
    if isinstance(value, bool):
        val = 4  # length of 'True' or 'False'
    elif isinstance(value, (int, float)):
        val = len(str(value))
    elif isinstance(value, str):
        val = len(value)
    else:
        val = 0
    return val**2

def evaluate_df_prefix_hit_cnt(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Function to evaluate the prefix hit count of a DataFrame
    """

    def max_overlap(trie, row_string):
        return min(len(row_string), trie.longest_common_prefix(row_string))


    trie = Trie()
    total_prefix_hit_count = 0
    total_string_length = 0
    
    num_rows = len(df)
    # Progress indicator for large datasets
    progress_interval = max(1, num_rows // 20)  # Show progress every 5%

    def process_row(index, row):
        nonlocal total_string_length, total_prefix_hit_count
        row_string = "".join(row.fillna("").astype(str).values)  # No spaces between columns
        total_string_length += len(row_string)
        row_prefix_hit_count = max_overlap(trie, row_string)
        trie.insert(row_string)
        total_prefix_hit_count += row_prefix_hit_count

    # Actually iterate through the DataFrame rows
    for idx, (index, row) in enumerate(df.iterrows(), 1):
        process_row(index, row)
        # Show progress for large datasets
        if num_rows > 10000 and idx % progress_interval == 0:
            progress_pct = (idx / num_rows) * 100
            print(f"[utils] Progress: {idx}/{num_rows} rows ({progress_pct:.1f}%)", file=sys.stderr, flush=True)

    total_prefix_hit_rate = total_prefix_hit_count / total_string_length if total_string_length > 0 else 0
    assert total_prefix_hit_count <= total_string_length
    return total_prefix_hit_count, total_prefix_hit_rate * 100