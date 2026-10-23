"""
HarnessPro 多目标 Pareto 前沿评价器 (HarnessPro Multi-Objective Pareto Evaluator)
参考:
- HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization
- Multi-Objective Quantitative Strategy Selection
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from ..contracts import StrategyVersion


class HarnessProParetoEvaluator:
    """
    基于 HarnessPro 前沿优化理论的多目标评价器:
    同时权衡【净收益/夏普】与【换手税费代价】以及【AFML稳健性防线】
    杜绝牺牲几十万手续费换取 0.05 虚假夏普的劣质优化。
    """

    def __init__(
        self,
        weight_sharpe: float = 0.40,
        weight_friction_penalty: float = 0.25,
        weight_calmar: float = 0.20,
        weight_dsr_bonus: float = 0.15,
    ):
        self.w_sharpe = weight_sharpe
        self.w_friction = weight_friction_penalty
        self.w_calmar = weight_calmar
        self.w_dsr = weight_dsr_bonus

    def compute_pareto_score(self, version: StrategyVersion) -> float:
        """
        计算版本的综合效用得分 (Pareto Fitness Score)
        得分越高，表示在同等或更低摩擦下实现了更优质的风险调整后超额。
        """
        m = version.metrics
        aud = version.audit_verdict

        sharpe = float(m.get("sharpe_ratio", 0.0))
        calmar = float(m.get("calmar_ratio", 0.0))
        ann_turnover = float(m.get("annualized_turnover", 1.0))
        total_fees = float(m.get("total_fees_paid", 0.0))

        # 摩擦惩罚项 (归一化：换手率每增加 10x 扣减 0.2 分，税费按初始资金比例扣减)
        friction_penalty = (ann_turnover / 50.0) * 0.2 + (total_fees / 10000000.0) * 5.0

        # AFML 稳健性增益
        dsr_val = float(aud.get("dsr_value", 0.95))
        pbo_val = float(aud.get("pbo_value", 0.20))
        audit_robustness = dsr_val - (pbo_val * 1.5)

        score = (
            self.w_sharpe * sharpe +
            self.w_calmar * min(calmar, 3.0) +
            self.w_dsr * audit_robustness -
            self.w_friction * friction_penalty
        )
        return float(round(score, 4))

    def evaluate_pareto_dominance(
        self,
        candidate: StrategyVersion,
        baseline: StrategyVersion,
    ) -> Tuple[bool, str]:
        """
        判定候选版本是否弱支配（Pareto Dominates）或优于基准版本
        返回: (is_dominant, reason_narrative)
        """
        c_m = candidate.metrics
        b_m = baseline.metrics
        c_aud = candidate.audit_verdict
        b_aud = baseline.audit_verdict

        c_sharpe = c_m.get("sharpe_ratio", 0.0)
        b_sharpe = b_m.get("sharpe_ratio", 0.0)

        c_fees = c_m.get("total_fees_paid", 0.0)
        b_fees = b_m.get("total_fees_paid", 0.0)

        c_mdd = c_m.get("max_drawdown", 1.0)
        b_mdd = b_m.get("max_drawdown", 1.0)

        c_pbo = c_aud.get("pbo_value", 0.20)
        b_pbo = b_aud.get("pbo_value", 0.20)

        c_score = self.compute_pareto_score(candidate)
        b_score = self.compute_pareto_score(baseline)
        candidate.pareto_score = c_score
        baseline.pareto_score = b_score

        # 核心硬门禁：如果 PBO 显著上升超过安全红线 (0.35)，即便收益提升也一票否决
        if c_pbo > 0.35 and c_pbo > b_pbo:
            return False, f"拒绝：候选版本过度拟合风险加剧 (PBO {c_pbo:.1%} > 35.0%)"

        # 核心硬门禁：如果换手增加导致税费剧增 (>1.5倍) 但夏普几乎未动，判定为劣质高频摩擦
        if c_fees > b_fees * 1.5 and (c_sharpe - b_sharpe) < 0.1:
            return False, f"拒绝：手续费与滑点磨损剧增 (¥{c_fees:,.0f} vs ¥{b_fees:,.0f})，但夏普未见显著突破"

        # Pareto 综合效用提升且未击穿底线
        if c_score > b_score and c_sharpe >= (b_sharpe - 0.05):
            gain_desc = f"夏普变动 {c_sharpe - b_sharpe:+.3f}, 回撤变动 {c_mdd - b_mdd:+.2%}, 费用变动 ¥{c_fees - b_fees:+,.0f}"
            return True, f"采纳：Pareto 综合效用显著提升 ({c_score:.3f} > {b_score:.3f}) [{gain_desc}]"

        return False, f"拒绝：未能实现 Pareto 净效用改善 ({c_score:.3f} <= {b_score:.3f})"
