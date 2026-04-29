"""Phase 7a: ViolationBreakdown structured aggregation tests."""
from __future__ import annotations

from analytics.tester import (
    Violation, ViolationBreakdown, ViolationReport,
)


def _vio(name='ClubGameSpread', message='msg', clubs=None, metric=None,
         affected_games=None, week=None):
    return Violation.create(
        name, message, affected_games=affected_games or [],
        affected_clubs=clubs or [], metric_value=metric, week=week,
    )


class TestViolationBreakdownEmpty:
    def test_empty_violations_empty_breakdown(self):
        bd = ViolationBreakdown.from_violations([])
        assert bd.by_club == {}
        assert bd.by_type == {}
        assert bd.by_severity == {}
        assert bd.soft_pressure == {}


class TestViolationBreakdownGrouping:
    def test_by_type_groups_constraint(self):
        v1 = _vio('ClubGameSpread', clubs=['Tigers'])
        v2 = _vio('ClubGameSpread', clubs=['Wests'])
        v3 = _vio('ClubGradeAdjacency', clubs=['Tigers'])
        bd = ViolationBreakdown.from_violations([v1, v2, v3])
        assert sorted(bd.by_type) == ['ClubGameSpread', 'ClubGradeAdjacency']
        assert len(bd.by_type['ClubGameSpread']) == 2
        assert len(bd.by_type['ClubGradeAdjacency']) == 1

    def test_by_club_aggregates_per_club(self):
        v1 = _vio(clubs=['Tigers', 'Wests'])
        v2 = _vio(clubs=['Tigers'])
        bd = ViolationBreakdown.from_violations([v1, v2])
        assert len(bd.by_club['Tigers']) == 2
        assert len(bd.by_club['Wests']) == 1

    def test_by_severity_uses_severity_label(self):
        # NoDoubleBookingTeams = CRITICAL
        v_crit = _vio('NoDoubleBookingTeams', clubs=['X'])
        # ClubGameSpread = MEDIUM
        v_med = _vio('ClubGameSpread', clubs=['X'])
        bd = ViolationBreakdown.from_violations([v_crit, v_med])
        assert 'CRITICAL' in bd.by_severity
        assert 'MEDIUM' in bd.by_severity


class TestViolationBreakdownSoftPressure:
    def test_metric_value_aggregates_into_pressure(self):
        v1 = _vio('ClubGameSpread', clubs=['Tigers'], metric=2)
        v2 = _vio('ClubGameSpread', clubs=['Wests'], metric=5)
        bd = ViolationBreakdown.from_violations([v1, v2])
        sp = bd.soft_pressure['ClubGameSpread']
        assert sp['over_limit'] == 2
        assert sp['total_penalty'] == 7
        assert sp['worst_value'] == 5
        assert sp['worst_club'] == 'Wests'

    def test_no_metric_value_skips_pressure(self):
        v1 = _vio('ClubGameSpread', clubs=['Tigers'])  # no metric
        bd = ViolationBreakdown.from_violations([v1])
        # No soft_pressure entry — only violations with metric_value contribute.
        assert 'ClubGameSpread' not in bd.soft_pressure


class TestViolationReportBreakdown:
    def test_report_breakdown_property(self):
        v = _vio('ClubGameSpread', clubs=['Tigers'], metric=3)
        report = ViolationReport(
            draw_description='t', total_games=1, violations=[v],
        )
        bd = report.breakdown
        assert isinstance(bd, ViolationBreakdown)
        assert bd.by_type == {'ClubGameSpread': [v]}
        assert bd.soft_pressure['ClubGameSpread']['worst_club'] == 'Tigers'
