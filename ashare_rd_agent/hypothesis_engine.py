"""
第一性原理经济假说引擎 (First-Principles Economic Hypothesis Engine)
参考:
- Microsoft Research RD-Agent Hypothesis-driven Quantitative Generation
- Behavioral Finance & Market Microstructure Foundations
- RRSI First-Principles Regularization Barrier
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import random

from .contracts import (
    EconomicHypothesis,
    HypothesisCategory,
    StrategyCodeSpec,
    VersionDiff,
)


# 内置已严格通过金融学第一性原理论证的假说库 (Builtin Hypotheses Catalog)
BUILTIN_HYPOTHESIS_TEMPLATES: List[Dict[str, Any]] = [
    {
        "category": HypothesisCategory.MOMENTUM_UNDERREACTION,
        "title": "中期动量在流动性充裕标的上的缓释溢价",
        "rationale": "基于 Hong & Stein (1999) 不完全信息扩散模型，机构投资者对基本面信息的逐步反应导致中期（20-60日）价格呈现持续趋势漂移。",
        "prediction": "在快线均线上穿慢线时构建多头权重，在中期趋势未破坏前持有，年化夏普比率应显著跑赢等权基准。",
        "holding_days": 20,
        "complexity": 1,
        "default_strategy": "trend",
        "base_params": {"fast_period": 10, "slow_period": 45},
    },
    {
        "category": HypothesisCategory.STATUTORY_FRICTION_ARBITRAGE,
        "title": "摩擦敏感型均线平滑与换手率抑制假说",
        "rationale": "A 股单边千五印花税与双边佣金对高频信号具有致命磨损；拉长滤波窗口并提高信号穿越死区（Deadband），降低假突破换手，净收益反而提升。",
        "prediction": "将慢线均线从 45 延长至 75，并引入双边确认周期，年化换手率下降 40% 以上，扣除税费后的净夏普显著改善。",
        "holding_days": 40,
        "complexity": 2,
        "default_strategy": "trend",
        "base_params": {"fast_period": 15, "slow_period": 75},
    },
    {
        "category": HypothesisCategory.MEAN_REVERSION_INVENTORY,
        "title": "极端波动率偏离下的流动性提供者库存回归假说",
        "rationale": "根据 Ho-Stoll 微观做市商库存模型，当价格急剧跌破布林线下轨 2.2 倍标准差时，非理性恐慌抛售导致暂态流动性枯竭，做市商低吸平抑库存后价格快速均值回归。",
        "prediction": "在价格突破下轨时阶梯式吸纳筹码，在均线处获利平仓，可获得显著的短期反转 Alpha。",
        "holding_days": 10,
        "complexity": 2,
        "default_strategy": "mean_reversion",
        "base_params": {"period": 20, "std_dev": 2.2},
    },
    {
        "category": HypothesisCategory.LIQUIDITY_PREMIUM,
        "title": "量价结合：成交额过滤与突破阻力最小路径假说",
        "rationale": "缩量突破多为诱多陷阱，带量突破表明主力资金实质性建仓。引入成交量加权特征可有效剔除无成交支撑的假信号。",
        "prediction": "仅在成交量高于 20 日均值 1.2 倍时确认均线金叉，策略胜率可提高 5-8 个百分点，最大回撤显著收敛。",
        "holding_days": 25,
        "complexity": 3,
        "default_strategy": "trend",
        "base_params": {"fast_period": 15, "slow_period": 60, "volume_multiplier": 1.2},
    },
]


class FirstPrinciplesHypothesisEngine:
    """
    第一性原理假说引擎:
    1. 负责针对初始策略生成符合经济学机理的种子假说
    2. 基于前序轮次的回测归因与失败教训（Reflective Feedback），推演下一代改进假说
    3. 严禁无经济学意义的纯随机参数过拟合挖掘
    """

    def __init__(self):
        self._history_hypotheses: Dict[str, EconomicHypothesis] = {}

    def generate_seed_hypothesis(
        self,
        category: Optional[HypothesisCategory] = None,
    ) -> Tuple[EconomicHypothesis, StrategyCodeSpec]:
        """生成初始基准假说与代码规格"""
        matched = [
            t for t in BUILTIN_HYPOTHESIS_TEMPLATES
            if category is None or t["category"] == category
        ]
        template = matched[0] if matched else BUILTIN_HYPOTHESIS_TEMPLATES[0]

        hypo = EconomicHypothesis(
            category=template["category"],
            title=template["title"],
            economic_rationale=template["rationale"],
            falsifiable_prediction=template["prediction"],
            target_holding_period_days=template["holding_days"],
            complexity_degree=template["complexity"],
        )

        code_spec = StrategyCodeSpec(
            strategy_type=template["default_strategy"],
            parameters=dict(template["base_params"]),
            code_version="v1.0.0",
        )

        self._history_hypotheses[hypo.hypothesis_id] = hypo
        return hypo, code_spec

    def mutate_hypothesis_from_feedback(
        self,
        current_hypo: EconomicHypothesis,
        current_spec: StrategyCodeSpec,
        diff: Optional[VersionDiff] = None,
        failure_reason: Optional[str] = None,
        iteration: int = 1,
    ) -> Tuple[EconomicHypothesis, StrategyCodeSpec]:
        """
        基于前序失败反馈执行定向第一性原理突变 (Propose Step)
        依据具体失败症状（如换手过高、回撤过大、过拟合预警）选择针对性金融优化机制
        """
        params = dict(current_spec.parameters)
        strategy_type = current_spec.strategy_type
        new_version_tag = f"v1.{iteration}.0"

        # 诊断类型 A: 摩擦税费过高，换手击穿收益 (Turnover Drag)
        if diff and diff.total_friction_delta > 100000 or (failure_reason and "friction" in failure_reason.lower()):
            new_title = f"{current_hypo.title} [低换手平滑迭代]"
            new_rationale = (
                f"上一轮测算显示年化换手过大，产生过高印花税与佣金磨损。"
                f"根据第一性原理，拉大周期过滤网格，消除边缘震荡引发的非必要再平衡。"
            )
            if "slow_period" in params:
                params["slow_period"] = int(params["slow_period"] * 1.3)
                params["fast_period"] = int(params.get("fast_period", 10) * 1.2)
            elif "period" in params:
                params["period"] = int(params["period"] * 1.25)
                params["std_dev"] = float(params.get("std_dev", 2.0) + 0.2)

            category = HypothesisCategory.STATUTORY_FRICTION_ARBITRAGE
            complexity = current_hypo.complexity_degree + 1

        # 诊断类型 B: 发生过拟合 (PBO > 0.30 或 DSR 衰减)
        elif diff and diff.pbo_delta > 0.10 or (failure_reason and "overfit" in failure_reason.lower()):
            new_title = f"{current_hypo.title} [抗过拟合正则化约束]"
            new_rationale = (
                f"上一版本在交叉验证中表现出参数敏感与过拟合征兆。"
                f"根据奥卡姆剃刀准则，收缩参数离散度并向中心稳健参数靠拢。"
            )
            if "slow_period" in params:
                params["slow_period"] = 60
                params["fast_period"] = 15
            elif "period" in params:
                params["period"] = 20
                params["std_dev"] = 2.0

            category = HypothesisCategory.MOMENTUM_UNDERREACTION
            complexity = max(1, current_hypo.complexity_degree - 1)

        # 诊断类型 C: 最大回撤过大 (Drawdown 突破风控容忍)
        elif diff and diff.max_drawdown_delta > 0.05 or (failure_reason and "drawdown" in failure_reason.lower()):
            new_title = f"{current_hypo.title} [右侧止损保护与安全边际加固]"
            new_rationale = "前序版本在单边下行行情中承受了过大最大回撤，根据风险控制第一性原理，缩短快线探测窗口以实现更灵敏的右侧离场。"
            if "fast_period" in params:
                params["fast_period"] = max(5, int(params["fast_period"] * 0.8))
            category = HypothesisCategory.VOLATILITY_CLUSTERING
            complexity = current_hypo.complexity_degree + 1

        # 默认温和探索演化
        else:
            new_title = f"{current_hypo.title} [周期微调演化]"
            new_rationale = "在前序稳定收益基础上，探索更贴合当前市场中枢特征的平滑参数组合。"
            if "slow_period" in params:
                params["slow_period"] = int(params["slow_period"] + 5)
            elif "std_dev" in params:
                params["std_dev"] = round(params["std_dev"] + 0.1, 2)
            category = current_hypo.category
            complexity = current_hypo.complexity_degree

        new_hypo = EconomicHypothesis(
            category=category,
            title=new_title,
            economic_rationale=new_rationale,
            falsifiable_prediction=f"在 {current_hypo.target_holding_period_days} 日持仓周期内，改善换手摩擦或夏普比率。",
            target_holding_period_days=current_hypo.target_holding_period_days,
            complexity_degree=complexity,
        )

        new_spec = StrategyCodeSpec(
            strategy_type=strategy_type,
            parameters=params,
            code_version=new_version_tag,
        )

        self._history_hypotheses[new_hypo.hypothesis_id] = new_hypo
        return new_hypo, new_spec
