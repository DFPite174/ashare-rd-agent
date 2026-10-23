"""
HarnessPro Pareto 评价器、归因解构与两级递归记忆单元测试
"""

import pytest
from ashare_rd_agent.contracts import StrategyVersion, StrategyCodeSpec, EconomicHypothesis, VersionDiff
from ashare_rd_agent.evaluator.pareto_evaluator import HarnessProParetoEvaluator
from ashare_rd_agent.evaluator.version_diff import VersionDiffAnalyzer
from ashare_rd_agent.knowledge.strategy_memory import StrategyKnowledgeMemory


def test_pareto_evaluator_dominance_and_scoring():
    evaluator = HarnessProParetoEvaluator()

    base_ver = StrategyVersion(
        version_id="base",
        iteration=1,
        metrics={
            "sharpe_ratio": 1.0,
            "calmar_ratio": 1.2,
            "annualized_turnover": 15.0,
            "total_fees_paid": 100000.0,
            "max_drawdown": 0.15,
        },
        audit_verdict={"dsr_value": 0.96, "pbo_value": 0.18},
    )

    # 1. 优秀候选：夏普提升，回撤缩小，且税费未失控
    good_cand = StrategyVersion(
        version_id="good",
        iteration=2,
        metrics={
            "sharpe_ratio": 1.45,
            "calmar_ratio": 1.8,
            "annualized_turnover": 12.0,
            "total_fees_paid": 90000.0,
            "max_drawdown": 0.11,
        },
        audit_verdict={"dsr_value": 0.98, "pbo_value": 0.12},
    )

    is_dom, reason = evaluator.evaluate_pareto_dominance(good_cand, base_ver)
    assert is_dom is True
    assert "Pareto 综合效用显著提升" in reason
    assert good_cand.pareto_score > base_ver.pareto_score

    # 2. 伪劣优化：夏普微增 0.02，但换手与手续费翻了三倍
    toxic_cand = StrategyVersion(
        version_id="toxic",
        iteration=2,
        metrics={
            "sharpe_ratio": 1.02,
            "calmar_ratio": 1.1,
            "annualized_turnover": 65.0,
            "total_fees_paid": 400000.0,
            "max_drawdown": 0.16,
        },
        audit_verdict={"dsr_value": 0.92, "pbo_value": 0.22},
    )

    is_dom2, reason2 = evaluator.evaluate_pareto_dominance(toxic_cand, base_ver)
    assert is_dom2 is False
    assert "磨损剧增" in reason2


def test_version_diff_analyzer_narrative():
    analyzer = VersionDiffAnalyzer()

    v1 = StrategyVersion(
        version_id="v1", iteration=1,
        metrics={"annualized_return": 0.20, "sharpe_ratio": 1.1, "max_drawdown": 0.15, "total_fees_paid": 80000.0},
        audit_verdict={"dsr_value": 0.95, "pbo_value": 0.15},
    )
    v2 = StrategyVersion(
        version_id="v2", iteration=2,
        metrics={"annualized_return": 0.28, "sharpe_ratio": 1.5, "max_drawdown": 0.10, "total_fees_paid": 70000.0},
        audit_verdict={"dsr_value": 0.97, "pbo_value": 0.12},
    )

    diff = analyzer.analyze(v2, v1, iteration=2)
    assert diff.sharpe_delta > 0.3
    assert diff.max_drawdown_delta < -0.04
    assert diff.total_friction_delta < 0
    assert "风险调整收益提升" in diff.attribution_narrative
    assert "最大回撤收敛" in diff.attribution_narrative


def test_strategy_knowledge_memory_persistence():
    memory = StrategyKnowledgeMemory()

    v1 = StrategyVersion(version_id="v1", iteration=1, pareto_score=1.2)
    v2 = StrategyVersion(version_id="v2", iteration=2, pareto_score=0.8)

    memory.record_iteration(v1, diff=None, is_accepted=True, decision_reason="Pareto Top")
    memory.record_iteration(v2, diff=None, is_accepted=False, decision_reason="Friction killed alpha")

    summary = memory.summarize()
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["best_version_id"] == "v1"
