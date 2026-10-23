"""
RD-Agent 自主进化控制面契约与状态机规范 (Autonomous Evolution Control Plane Contracts)
参考:
- Google Research Harness² (Two-Level Recursion & Propose-Probe-Compose)
- Google Cloud AI RRSI (Regularized Recursive Self-Improvement)
- HarnessPro Pareto Optimization Framework
- LangGraph Stateful Agent Graph Specification
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, TypedDict
from datetime import datetime
import uuid


class HypothesisCategory(str, Enum):
    """第一性原理经济假说分类"""
    LIQUIDITY_PREMIUM = "LIQUIDITY_PREMIUM"
    MOMENTUM_UNDERREACTION = "MOMENTUM_UNDERREACTION"
    MEAN_REVERSION_INVENTORY = "MEAN_REVERSION_INVENTORY"
    QUALITY_GROWTH_DIVERGENCE = "QUALITY_GROWTH_DIVERGENCE"
    VOLATILITY_CLUSTERING = "VOLATILITY_CLUSTERING"
    STATUTORY_FRICTION_ARBITRAGE = "STATUTORY_FRICTION_ARBITRAGE"


@dataclass(frozen=True)
class EconomicHypothesis:
    """第一性原理经济假说契约"""
    hypothesis_id: str = field(default_factory=lambda: f"hypo_{uuid.uuid4().hex[:8]}")
    category: HypothesisCategory = HypothesisCategory.MOMENTUM_UNDERREACTION
    title: str = ""
    economic_rationale: str = ""
    falsifiable_prediction: str = ""
    target_holding_period_days: int = 5
    complexity_degree: int = 1
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyCodeSpec:
    """策略代码载荷与超参定义"""
    strategy_type: str = "trend"
    parameters: Dict[str, Any] = field(default_factory=dict)
    custom_factor_code: Optional[str] = None
    code_version: str = "v1.0.0"


@dataclass
class StrategyVersion:
    """策略演进历史版本归档对象"""
    version_id: str
    iteration: int
    parent_version_id: Optional[str] = None
    hypothesis: Optional[EconomicHypothesis] = None
    code_spec: Optional[StrategyCodeSpec] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    audit_verdict: Dict[str, Any] = field(default_factory=dict)
    pareto_rank: int = 0
    pareto_score: float = 0.0
    is_accepted: bool = False
    rejection_reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class VersionDiff:
    """版本变动对比分析契约 (Harness² Contrastive Attribution)"""
    base_version_id: str
    candidate_version_id: str
    iteration: int
    
    annualized_return_delta: float = 0.0
    sharpe_delta: float = 0.0
    max_drawdown_delta: float = 0.0
    calmar_delta: float = 0.0
    
    turnover_delta: float = 0.0
    total_friction_delta: float = 0.0
    
    dsr_delta: float = 0.0
    pbo_delta: float = 0.0
    
    is_pareto_improvement: bool = False
    is_overfitted: bool = False
    attribution_narrative: str = ""


class EvolutionState(TypedDict, total=False):
    """LangGraph 状态机载荷类型定义"""
    iteration: int
    max_iterations: int
    active_version: Optional[StrategyVersion]
    candidate_version: Optional[StrategyVersion]
    current_hypothesis: Optional[EconomicHypothesis]
    version_history: List[StrategyVersion]
    version_diffs: List[VersionDiff]
    knowledge_memory: Dict[str, Any]
    last_error: Optional[str]
    termination_reason: Optional[str]
    status: str
