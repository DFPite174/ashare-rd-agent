"""
RRSI 正则化自进化防过拟合守卫 (RRSI Anti-Overfitting Regularizer)
参考:
- Google Cloud AI "Regularized Recursive Self-Improvement of Agent Harnesses" (RRSI)
- Bailey & Lopez de Prado (2014) The Deflated Sharpe Ratio
- Occam's Razor Model Selection Criterion
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from ..contracts import EconomicHypothesis, StrategyVersion, VersionDiff


class RRSIRegularizer:
    """
    RRSI 正则化自进化守卫:
    1. 强制要求第一性原理经济假说支持（拒绝纯粹拟合的黑箱符号）
    2. 依据奥卡姆剃刀施加模型复杂度正则化惩罚
    3. 严密监视样本外泛化性衰减，主动阻止自进化循环中的过拟合漂移
    """

    def __init__(
        self,
        complexity_penalty_lambda: float = 0.05,
        max_allowed_pbo: float = 0.35,
        min_allowed_dsr: float = 0.90,
    ):
        self.complexity_lambda = complexity_penalty_lambda
        self.max_pbo = max_allowed_pbo
        self.min_dsr = min_allowed_dsr

    def validate_first_principles(self, hypothesis: Optional[EconomicHypothesis]) -> Tuple[bool, str]:
        """验证假说是否具备合格的第一性原理经济学机理"""
        if hypothesis is None:
            return False, "RRSI 拦截: 缺少经济假说载体"

        if not hypothesis.economic_rationale or len(hypothesis.economic_rationale.strip()) < 15:
            return False, "RRSI 拦截: 经济学机理解释过于单薄，存在黑箱数据挖掘嫌疑"

        if not hypothesis.falsifiable_prediction:
            return False, "RRSI 拦截: 假说未给出明确的可证伪预测"

        return True, "第一性原理审查通过"

    def apply_regularization_score(
        self,
        version: StrategyVersion,
    ) -> float:
        """
        施加复杂度正则化惩罚后的最终净效用得分:
        RegularizedScore = RawScore - lambda * ComplexityDegree
        """
        raw_score = version.pareto_score
        complexity = version.hypothesis.complexity_degree if version.hypothesis else 1
        reg_penalty = self.complexity_lambda * complexity
        final_score = raw_score - reg_penalty
        return round(final_score, 4)

    def check_overfitting_alarm(
        self,
        diff: VersionDiff,
        candidate_version: StrategyVersion,
    ) -> Tuple[bool, str]:
        """
        探测新版本是否发生自进化过拟合
        """
        aud = candidate_version.audit_verdict
        pbo = float(aud.get("pbo_value", 0.20))
        dsr = float(aud.get("dsr_value", 0.95))

        if pbo > self.max_pbo:
            return True, f"RRSI 过拟合熔断: CSCV PBO 过拟合概率达 {pbo:.1%}，超出 {self.max_pbo:.1%} 安全上限"

        if dsr < self.min_dsr and diff.sharpe_delta > 0.1:
            return True, f"RRSI 假阳性熔断: 虽回测夏普提升，但紧缩夏普比率 DSR 降至 {dsr:.2f}，未能通过多重测试惩罚检验"

        return False, "未触发过拟合预警"
