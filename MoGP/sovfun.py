from typing import List, Sequence, Any, Dict, Tuple, Optional, Union
import numpy as np

def _to_segments(seq: Sequence[Any], ignore_label: Optional[Any] = None) -> List[Tuple[Any, int, int]]:
    """将序列转换为片段列表: (状态, start, end) [start, end)"""
    segs = []
    n = len(seq)
    if n == 0: return segs
    i = 0
    while i < n:
        s = seq[i]
        if ignore_label is not None and s == ignore_label:
            i += 1
            continue
        j = i + 1
        while j < n and seq[j] == s:
            j += 1
        segs.append((s, i, j))
        i = j
    return segs

def sov_score(
    y_preds: Sequence[Any],
    y_labels: Sequence[Any],
    datalens: Sequence[int],
    states: Optional[Sequence[Any]] = None,
    ignore_label: Optional[Any] = None,
    return_per_state: bool = False,
) -> Union[float, Tuple[float, Dict[Any, float]]]:
    """
    严格版 SOV'99 实现。
    解决了分母累加错误导致的 Case 3 (83.33) 和 Case 5 (16.67) 问题。
    """
    y_preds = np.array(y_preds)
    y_labels = np.array(y_labels)

    # 确定要计算的状态集合
    if states is None:
        st = set(y_labels.tolist())
        if ignore_label is not None and ignore_label in st:
            st.remove(ignore_label)
        states = sorted(list(st), key=lambda x: str(x))

    num_by_state = {s: 0.0 for s in states}
    den_by_state = {s: 0.0 for s in states}

    offset = 0
    for L in datalens:
        lab_sub = y_labels[offset : offset + L]
        pre_sub = y_preds[offset : offset + L]
        offset += L

        # 获取当前序列的所有片段
        obs_segs = _to_segments(lab_sub, ignore_label=ignore_label)
        pre_segs = _to_segments(pre_sub, ignore_label=ignore_label)

        # 按状态分类
        for s in states:
            O_list = [seg for seg in obs_segs if seg[0] == s]
            P_list = [seg for seg in pre_segs if seg[0] == s]
            
            if not O_list:
                continue

            for (_, oa, ob) in O_list:
                len_o = ob - oa
                # 找出所有与当前 O 有重叠的同状态 P
                intersecting_ps = [p for p in P_list if p[2] > oa and p[1] < ob]

                if not intersecting_ps:
                    den_by_state[s] += len_o
                    continue

                # --- 计算 SOV'99 参数 ---
                p_start = min(p[1] for p in intersecting_ps)
                p_end = max(p[2] for p in intersecting_ps)
                
                # minov: O 与所有重叠 P 的交集之和
                minov = 0
                for (_, pa, pb) in intersecting_ps:
                    minov += (min(ob, pb) - max(oa, pa))
                
                # maxov: O 与 P 集合的外包跨度
                maxov = max(ob, p_end) - min(oa, p_start)
                
                # len_p_span: 重叠预测片段的并集长度
                len_p_span = p_end - p_start
                
                # delta 计算
                delta = min(maxov - minov, minov, len_o // 2, len_p_span // 2)

                num_by_state[s] += ((minov + delta) / maxov) * len_o
                den_by_state[s] += len_o

    # 归一化
    overall_num = sum(num_by_state.values())
    overall_den = sum(den_by_state.values())
    overall = (overall_num / overall_den * 100.0) if overall_den > 0 else 0.0

    if return_per_state:
        per_state = {s: (num_by_state[s]/den_by_state[s]*100 if den_by_state[s]>0 else 0.0) for s in states}
        return overall, per_state
    
    return overall