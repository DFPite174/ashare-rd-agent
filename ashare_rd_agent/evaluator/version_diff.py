"""
版本变动解构与归因分析器 (Version Diff & Contrastive Attribution Engine)
参考:
- Harness² Propose-Probe-Compose Contrastive Attribution
- Brinson / Barra Quantitative Factor & Friction Decomposition
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from ..contracts import StrategyVersion, VersionDiff


class VersionDiffAnalyzer:
    """
    版本变动分析器:
    负责细粒度对比迭代版本与基准版本之间的差异，
    定量解构收益变动的真实来源（是纯阿尔法提升、承受了更高风险、还是被交易摩擦吞噬）。
    """

    def analyze(
        self,
        candidate: StrategyVersion,
        baseline: StrategyVersion,
        iteration: int = 1,
    ) -> VersionDiff:
        c_m = candidate.metrics
        b_m = baseline.metrics
        c_aud = candidate.audit_verdict
        b_aud = baseline.audit_verdict

        ann_ret_delta = float(c_m.get("annualized_return", 0.0) - b_m.get("annualized_return", 0.0))
        sharpe_delta = float(c_m.get("sharpe_ratio", 0.0) - b_m.get("sharpe_ratio", 0.0))
        mdd_delta = float(c_m.get("max_drawdown", 0.0) - b_m.get("max_drawdown", 0.0))
        calmar_delta = float(c_m.get("calmar_ratio", 0.0) - b_m.get("calmar_ratio", 0.0))

        turnover_delta = float(c_m.get("annualized_turnover", 0.0) - b_m.get("annualized_turnover", 0.0))
        friction_delta = float(c_m.get("total_fees_paid", 0.0) - b_m.get("total_fees_paid", 0.0))

        dsr_delta = float(c_aud.get("dsr_value", 0.95) - b_aud.get("dsr_value", 0.95))
        pbo_delta = float(c_aud.get("pbo_value", 0.20) - b_aud.get("pbo_value", 0.20))

        # 判定是否发生过拟合 (若夏普上升但 PBO 恶化或置换检验未通过)
        is_overfitted = bool(pbo_delta > 0.08 or (sharpe_delta > 0.2 and dsr_delta < -0.05))

        # 构建对比诊断解构评语 (Contrastive Attribution Narrative)
        narrative_parts = []
        if sharpe_delta > 0:
            narrative_parts.append(f"风险调整收益提升: 夏普比率改善 {sharpe_delta:+.3f} (年化收益 {ann_ret_delta:+.2%})")
        else:
            narrative_parts.append(f"风险调整收益下滑: 夏普比率下降 {sharpe_delta:+.3f}")

        if mdd_delta < 0:
            narrative_parts.append(f"最大回撤收敛 {abs(mdd_delta):.2%}，尾部下行风险改善")
        elif mdd_delta > 0.02:
            narrative_parts.append(f"警告: 最大回撤扩大 {mdd_delta:.2%}，下行波动放大")

        if friction_delta < 0:
            narrative_parts.append(f"交易摩擦节约 ¥{abs(friction_delta):,.0f} (换手率下降 {abs(turnover_delta):.1f}x)")
        elif friction_delta > 50000:
            narrative_parts.append(f"交易摩擦增加 ¥{friction_delta:,.0f} (换手率增加 {turnover_delta:+.1f}x)")

        if is_overfitted:
            narrative_parts.append(f"【过拟合警报】PBO 恶化 {pbo_delta:+.1%}，存在过拟合历史噪音嫌疑！")

        narrative = " | ".join(narrative_parts)

        return VersionDiff(
            base_version_id=baseline.version_id,
            candidate_version_id=candidate.version_id,
            iteration=iteration,
            annualized_return_delta=ann_ret_delta,
            sharpe_delta=sharpe_delta,
            max_drawdown_delta=mdd_delta,
            calmar_delta=calmar_delta,
            turnover_delta=turnover_delta,
            total_friction_delta=friction_delta,
            dsr_delta=dsr_delta,
            pbo_delta=pbo_delta,
            is_overfitted=is_overfitted,
            attribution_narrative=narrative,
        )
