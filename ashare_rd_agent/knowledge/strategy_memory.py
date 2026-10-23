"""
策略进化两级递归知识沉淀库 (Two-Level Recursion Strategy Knowledge Memory)
参考:
- Google Research Harness² Meta-Level Recursion
- Empirical Quant Research Repository
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime
from ..contracts import StrategyVersion, VersionDiff, EconomicHypothesis


class StrategyKnowledgeMemory:
    """
    两级递归知识库:
    记录跨轮次策略自进化过程中的正向阿尔法经验与负向失败教训，
    防止后续迭代生成曾经已被证明无效或高摩擦的劣质结构。
    """

    def __init__(self):
        self._accepted_versions: List[StrategyVersion] = []
        self._rejected_versions: List[StrategyVersion] = []
        self._lessons_learned: List[Dict[str, Any]] = []
        self._pareto_frontier: List[StrategyVersion] = []

    def record_iteration(
        self,
        candidate: StrategyVersion,
        diff: Optional[VersionDiff],
        is_accepted: bool,
        decision_reason: str,
    ):
        """记录单个迭代轮次的决策与沉淀知识"""
        candidate.is_accepted = is_accepted
        candidate.rejection_reason = decision_reason if not is_accepted else None

        if is_accepted:
            self._accepted_versions.append(candidate)
            self._update_pareto_frontier(candidate)
            lesson = {
                "iteration": candidate.iteration,
                "type": "POSITIVE_ALPHA",
                "hypothesis_title": candidate.hypothesis.title if candidate.hypothesis else "",
                "params": candidate.code_spec.parameters if candidate.code_spec else {},
                "insight": f"成功采纳: {decision_reason}",
                "timestamp": datetime.now().isoformat(),
            }
        else:
            self._rejected_versions.append(candidate)
            lesson = {
                "iteration": candidate.iteration,
                "type": "NEGATIVE_TRAP",
                "hypothesis_title": candidate.hypothesis.title if candidate.hypothesis else "",
                "params": candidate.code_spec.parameters if candidate.code_spec else {},
                "insight": f"失败规避: {decision_reason}",
                "timestamp": datetime.now().isoformat(),
            }

        self._lessons_learned.append(lesson)

    def _update_pareto_frontier(self, new_version: StrategyVersion):
        """维护全局当前非劣的 Pareto 前沿集合"""
        self._pareto_frontier.append(new_version)
        # 按综合 Pareto 得分降序排序
        self._pareto_frontier.sort(key=lambda v: v.pareto_score, reverse=True)
        # 保留 Top 5 最佳前沿
        self._pareto_frontier = self._pareto_frontier[:5]

    def get_pareto_frontier(self) -> List[StrategyVersion]:
        return list(self._pareto_frontier)

    def get_lessons(self, lesson_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if lesson_type:
            return [l for l in self._lessons_learned if l["type"] == lesson_type]
        return list(self._lessons_learned)

    def summarize(self) -> Dict[str, Any]:
        """产出全生命周期的知识沉淀总览"""
        return {
            "total_iterations": len(self._accepted_versions) + len(self._rejected_versions),
            "accepted_count": len(self._accepted_versions),
            "rejected_count": len(self._rejected_versions),
            "pareto_frontier_count": len(self._pareto_frontier),
            "best_version_id": self._pareto_frontier[0].version_id if self._pareto_frontier else None,
            "best_pareto_score": self._pareto_frontier[0].pareto_score if self._pareto_frontier else None,
            "positive_lessons_count": len(self.get_lessons("POSITIVE_ALPHA")),
            "negative_traps_count": len(self.get_lessons("NEGATIVE_TRAP")),
        }
