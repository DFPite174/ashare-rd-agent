# ashare-rd-agent (A 股量化策略自主进化控制面)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple.svg)]()
[![Harness²](https://img.shields.io/badge/Evolution-Harness%C2%B2-orange.svg)]()

> 基于 LangGraph 图工程、Google Research Harness² 两级递归、HarnessPro 多目标 Pareto 前沿与 Google Cloud AI RRSI 正则化防过拟合的 A 股量化策略自主演进控制面。

---

## 🌟 理论根基与顶尖前沿参考

1. **Graph Engineering (LangChain LangGraph)**: 有状态循环图（StateGraph）、检查点回滚与条件状态路由。
2. **Harness² (Google Research Oct 2026)**: 无 Ground-Truth 真实标签环境下的两级递归（Task-level Propose-Probe-Compose 闭环 + Meta-level Knowledge Memory 经验沉淀）。
3. **HarnessPro (Oct 2026)**: 多目标 Pareto 前沿优化（收益 vs 交易摩擦 vs 下行风险）。
4. **RRSI (Google Cloud AI Sep 2026)**: 正则化自进化防过拟合机制（第一性原理假说硬门禁、奥卡姆剃刀复杂度惩罚、CSCV PBO 熔断）。
5. **AReaL (清华 IIIS & 蚂蚁 2026)**: 异步探索解耦范式。

---

## 🚀 核心架构与模块说明

```
ashare_rd_agent/
├── contracts.py           # 假说、版本、差分归因与 LangGraph EvolutionState 契约
├── hypothesis_engine.py   # 第一性原理经济假说引擎 (动量、微观库存偏离、摩擦套利)
├── evaluator/
│   ├── pareto_evaluator.py # HarnessPro 多目标 Pareto 前沿优劣判定
│   ├── version_diff.py     # Harness² 版本因果解构与对比归因分析
│   └── rrsi_regularizer.py # RRSI 防过拟合正则化守卫与复杂度惩罚
├── knowledge/
│   └── strategy_memory.py  # 两级递归跨轮次知识库 (正向机制与负向陷阱记忆)
└── control_plane.py       # LangGraph 自主进化控制面总控状态机
```

---

## 📦 快速开始

```bash
pip install -e .
```

```python
from ashare_rd_agent import AutonomousEvolutionControlPlane
import pandas as pd

# 注入行情数据并启动 4 轮自主演进循环
plane = AutonomousEvolutionControlPlane(bars_df=my_bars_df)
report = plane.run_evolution_loop(max_iterations=4)

print("最终活跃版本:", report["final_active_version_id"])
print("演化谱系总结:", report["version_history_summary"])
```
