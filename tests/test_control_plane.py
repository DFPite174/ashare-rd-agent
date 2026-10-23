"""
RD-Agent 自主进化控制面端到端集成测试
"""

import pytest
import numpy as np
import pandas as pd
from ashare_rd_agent.control_plane import AutonomousEvolutionControlPlane


def test_autonomous_evolution_control_plane_multi_iteration():
    # 构造标准测试 K 线数据 (包含涨跌趋势)
    dates = pd.date_range("2024-01-01", periods=80, freq="B").strftime("%Y-%m-%d").tolist()
    bars = []
    np.random.seed(42)
    p = 20.0
    for d in dates:
        p *= (1.0 + np.random.normal(0.001, 0.02))
        bars.append({
            "trade_date": d,
            "symbol": "000001.SZ",
            "open": p * 0.99,
            "high": p * 1.02,
            "low": p * 0.98,
            "close": p,
            "volume": 100000,
            "amount": 2000000.0,
        })
    df = pd.DataFrame(bars)

    # 初始化自进化控制面
    plane = AutonomousEvolutionControlPlane(bars_df=df, initial_cash=10000000.0)

    # 运行 3 轮自主进化迭代循环
    report = plane.run_evolution_loop(max_iterations=3)

    assert report["total_iterations_run"] == 3
    assert report["final_active_version_id"] is not None
    assert len(report["version_history_summary"]) == 3

    # 验证版本变动分析 (Diff Attributions)
    assert len(report["diff_attributions"]) >= 2
    for diff in report["diff_attributions"]:
        assert "narrative" in diff
        assert len(diff["narrative"]) > 0

    # 验证两级递归知识库沉淀
    mem_summary = report["knowledge_memory_summary"]
    assert mem_summary["total_iterations"] == 3
    assert mem_summary["accepted_count"] >= 1
