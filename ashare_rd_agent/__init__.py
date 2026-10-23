"""
ashare-rd-agent (A 股量化策略自主进化控制面)
===========================================
基于 LangGraph 图工程、Google Research Harness² 两级递归、
HarnessPro 多目标 Pareto 前沿与 Google Cloud AI RRSI 正则化防过拟合。
"""

from .contracts import (
    EconomicHypothesis,
    HypothesisCategory,
    StrategyCodeSpec,
    StrategyVersion,
    VersionDiff,
    EvolutionState,
)
from .hypothesis_engine import FirstPrinciplesHypothesisEngine
from .evaluator.pareto_evaluator import HarnessProParetoEvaluator
from .evaluator.version_diff import VersionDiffAnalyzer
from .evaluator.rrsi_regularizer import RRSIRegularizer
from .knowledge.strategy_memory import StrategyKnowledgeMemory
from .control_plane import AutonomousEvolutionControlPlane

__version__ = "1.0.0"

__all__ = [
    "EconomicHypothesis",
    "HypothesisCategory",
    "StrategyCodeSpec",
    "StrategyVersion",
    "VersionDiff",
    "EvolutionState",
    "FirstPrinciplesHypothesisEngine",
    "HarnessProParetoEvaluator",
    "VersionDiffAnalyzer",
    "RRSIRegularizer",
    "StrategyKnowledgeMemory",
    "AutonomousEvolutionControlPlane",
]
