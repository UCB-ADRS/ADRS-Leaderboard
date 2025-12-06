import pandas as pd
from typing import List, Tuple


import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import networkx as nx
import numpy as np

class Evolved(Algorithm):
    """
    GGR algorithm
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.val_len = {}
        self.base = 5000
        self.row_stop = None
        self.col_stop = None

    def find_max_group_value(self, value_counts: Dict, early_stop: int = 0) -> str:
        best_val, best_score = None, -1
        for val, count in value_counts.items():
            if count <= 1: continue
            vl = self.val_len.get(val)
            if vl is None:
                vl = self.calculate_length(val)
                self.val_len[val] = vl
            score = vl * (count - 1)
            if score > best_score:
                best_score, best_val = score, val
        return best_val if best_score >= early_stop else None

    def column_recursion(self, max_value, grouped_rows, row_stop, col_stop, early_stop):
        values = grouped_rows.values
        mask = (values == max_value)
        idx = np.argsort(~mask, axis=1, kind='stable')
        sorted_values = values[np.arange(len(grouped_rows))[:, None], idx]
        
        remainder_values = sorted_values[:, 1:]
        remainder_counts = Counter(remainder_values.ravel())
        
        reordered_remainder, _ = self.recursive_reorder(
            pd.DataFrame(remainder_values), remainder_counts, early_stop, row_stop=row_stop, col_stop=col_stop + 1
        )
        
        result_df = pd.DataFrame(np.hstack([np.full((len(grouped_rows), 1), max_value, dtype=object), reordered_remainder.values]))
        return result_df, remainder_counts

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        cols = [c for c in df.columns if c != "original_index"]
        if not cols: return df, []
        
        n_rows = len(df)
        sample = df.sample(min(n_rows, 1000), random_state=42) if n_rows > 1000 else df
        
        scores = {}
        for col in cols:
            try:
                vc = sample[col].value_counts(normalize=True)
                p = (vc ** 2).sum()
                lens = sample[col].astype(str).str.len()
                avg_sq_len = (lens ** 2).mean()
                if pd.isna(avg_sq_len): avg_sq_len = 0
                scores[col] = avg_sq_len * p / (1 - p + 1e-9)
            except:
                scores[col] = 0
                
        sorted_cols = sorted(cols, key=lambda x: scores.get(x, 0), reverse=True)
        if "original_index" in df.columns:
            sorted_cols.append("original_index")
            
        reordered_df = df[sorted_cols]
        if row_sort:
            sort_cols = [c for c in sorted_cols if c != "original_index"]
            if sort_cols: reordered_df = reordered_df.sort_values(by=sort_cols)
        return reordered_df, []

    def recursive_reorder(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0, row_stop: int = 0, col_stop: int = 0) -> Tuple[pd.DataFrame, List[List[str]]]:
        if df.empty: return df, []
        if (self.row_stop and row_stop >= self.row_stop) or (self.col_stop and col_stop >= self.col_stop) or (df.shape[1] <= 1):
            return self.fixed_reorder(df)

        max_value = self.find_max_group_value(value_counts, early_stop)
        if max_value is None: return self.fixed_reorder(df)

        mask = (df.values == max_value).any(axis=1)
        if not mask.any(): return self.fixed_reorder(df)
        
        if mask.all():
            grouped_rows, remaining_rows = df, pd.DataFrame()
        else:
            grouped_rows, remaining_rows = df[mask], df[~mask]
            
        reordered_grouped, grouped_counts = self.column_recursion(max_value, grouped_rows, row_stop, col_stop, early_stop)
        
        if not remaining_rows.empty:
            remaining_counts = value_counts.copy()
            remaining_counts[max_value] -= len(grouped_rows)
            if remaining_counts[max_value] <= 0: del remaining_counts[max_value]
            for k, v in grouped_counts.items():
                remaining_counts[k] -= v
                if remaining_counts[k] <= 0: del remaining_counts[k]
            
            reordered_remaining, _ = self.recursive_reorder(remaining_rows, remaining_counts, early_stop, row_stop=row_stop + 1, col_stop=col_stop)
            final_df = pd.DataFrame(np.vstack([reordered_grouped.values, reordered_remaining.values]))
        else:
            final_df = reordered_grouped
            
        return final_df, []

    def recursive_split_and_reorder(self, df: pd.DataFrame, early_stop: int = 0):
        if len(df) <= self.base:
            return self.recursive_reorder(df, Counter(df.values.ravel()), early_stop, row_stop=0, col_stop=0)[0]
        mid = len(df) // 2
        with ThreadPoolExecutor(max_workers=2) as exc:
            f1, f2 = exc.submit(self.recursive_split_and_reorder, df.iloc[:mid], early_stop), exc.submit(self.recursive_split_and_reorder, df.iloc[mid:], early_stop)
            return pd.concat([f1.result(), f2.result()], axis=0, ignore_index=True)

    @lru_cache(maxsize=None)
    def calculate_length(self, value):
        if isinstance(value, str): return len(value) ** 2
        if isinstance(value, (int, float)): return len(str(value)) ** 2
        return 16

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        if col_merge:
            for group in col_merge:
                valid = [c for c in group if c in df.columns]
                if valid: df = self.merging_columns(df, valid, prepended=False)

        self.row_stop = row_stop if row_stop else len(df)
        self.col_stop = col_stop if col_stop else len(df.columns)
        self.val_len = {}
        
        work_df = df.copy()
        if "original_index" not in work_df.columns: work_df["original_index"] = range(len(work_df))
        
        threshold = len(work_df) * distinct_value_threshold
        cols_to_keep, cols_to_discard = [], []
        for col in work_df.columns:
            if col == "original_index": continue
            if work_df[col].nunique() > threshold: cols_to_discard.append(col)
            else: cols_to_keep.append(col)
            
        df_recurse = work_df[cols_to_keep + ["original_index"]]
        df_discard = work_df[cols_to_discard + ["original_index"]]
        
        try:
             vals = pd.unique(df_recurse.values.ravel())
             for v in vals:
                 if v not in self.val_len: self.val_len[v] = self.calculate_length(v)
        except:
             pass
            
        if parallel:
            reordered_core = self.recursive_split_and_reorder(df_recurse, early_stop)
        else:
            reordered_core, _ = self.recursive_reorder(df_recurse, Counter(df_recurse.values.ravel()), early_stop)
            
        reordered_core.columns = list(range(len(reordered_core.columns) - 1)) + ["original_index"]
        
        if cols_to_discard:
            final_df = pd.merge(reordered_core, df_discard, on="original_index", how="left")
        else:
            final_df = reordered_core
            
        final_df = final_df.drop(columns=["original_index"])
        return final_df.sort_values(by=list(final_df.columns)), []

class Solution:
    def __init__(self):
        self.alg = Evolved()

    def solve(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> pd.DataFrame:
        reordered_df, _ = self.alg.reorder(
            df,
            early_stop=early_stop,
            row_stop=row_stop,
            col_stop=col_stop,
            col_merge=col_merge,
            one_way_dep=one_way_dep,
            distinct_value_threshold=distinct_value_threshold,
            parallel=parallel,
        )
        return reordered_df


