"""
第一性原理假说引擎与 RRSI 守卫单元测试
"""

import pytest
from ashare_rd_agent.contracts import (
    EconomicHypothesis,
    HypothesisCategory,
    StrategyCodeSpec,
    VersionDiff,
)
from ashare_rd_agent.hypothesis_engine import FirstPrinciplesHypothesisEngine
from ashare_rd_agent.evaluator.rrsi_regularizer import RRSIRegularizer


def test_seed_hypothesis_first_principles():
    engine = FirstPrinciplesHypothesisEngine()
    hypo, spec = engine.generate_seed_hypothesis(HypothesisCategory.MOMENTUM_UNDERREACTION)

    assert hypo.category == HypothesisCategory.MOMENTUM_UNDERREACTION
    assert len(hypo.economic_rationale) > 20
    assert "Hong & Stein" in hypo.economic_rationale
    assert hypo.falsifiable_prediction != ""
    assert spec.strategy_type == "trend"
    assert "fast_period" in spec.parameters
    assert "slow_period" in spec.parameters


def test_mutate_hypothesis_on_turnover_friction_feedback():
    engine = FirstPrinciplesHypothesisEngine()
    hypo, spec = engine.generate_seed_hypothesis()

    # 构造换手税费剧增的反馈
    mock_diff = VersionDiff(
        base_version_id="v0",
        candidate_version_id="v1",
        iteration=1,
        total_friction_delta=150000.0, # 摩擦显著上升
    )

    new_hypo, new_spec = engine.mutate_hypothesis_from_feedback(
        current_hypo=hypo,
        current_spec=spec,
        diff=mock_diff,
        iteration=2,
    )

    # 验证定向引导向低换手平滑假说
    assert new_hypo.category == HypothesisCategory.STATUTORY_FRICTION_ARBITRAGE
    assert "换手" in new_hypo.title or "平滑" in new_hypo.title
    # 慢线周期延长以降低换手
    assert new_spec.parameters["slow_period"] > spec.parameters["slow_period"]


def test_rrsi_first_principles_validator():
    validator = RRSIRegularizer()

    # 合格假说
    valid_hypo = EconomicHypothesis(
        category=HypothesisCategory.LIQUIDITY_PREMIUM,
        title="Valid Hypo",
        economic_rationale="基于市场微观做市商流动性枯竭与反弹补仓需求构建反转信号",
        falsifiable_prediction="价格在布林下轨收敛前产生超额收益",
    )
    ok, msg = validator.validate_first_principles(valid_hypo)
    assert ok is True

    # 纯数据拟合、无经济解释的垃圾假说 -> 必须拦截
    invalid_hypo = EconomicHypothesis(
        category=HypothesisCategory.LIQUIDITY_PREMIUM,
        title="Invalid Hypo",
        economic_rationale="x = y * 2", # 单薄无经济机理
        falsifiable_prediction="sharpe will be high",
    )
    ok2, msg2 = validator.validate_first_principles(invalid_hypo)
    assert ok2 is False
    assert "RRSI 拦截" in msg2
