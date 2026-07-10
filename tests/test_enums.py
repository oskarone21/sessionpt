"""Tests for sessionpt.enums module."""

from sessionpt.enums.direction import Direction
from sessionpt.enums.exit_reason import ExitReason
from sessionpt.enums.pivot_level import PivotLevel
from sessionpt.enums.pivot_type import PivotType


class TestDirection:
    def test_long_value(self):
        assert Direction.LONG.value == "long"

    def test_short_value(self):
        assert Direction.SHORT.value == "short"

    def test_long_sign(self):
        assert Direction.LONG.sign == 1

    def test_short_sign(self):
        assert Direction.SHORT.sign == -1

    def test_from_value(self):
        assert Direction("long") == Direction.LONG
        assert Direction("short") == Direction.SHORT


class TestPivotType:
    def test_all_types(self):
        assert len(PivotType) == 6

    def test_traditional(self):
        assert PivotType.TRADITIONAL.value == "traditional"

    def test_woodie(self):
        assert PivotType.WOODIE.value == "woodie"

    def test_camarilla(self):
        assert PivotType.CAMARILLA.value == "camarilla"

    def test_dm(self):
        assert PivotType.DM.value == "dm"

    def test_fibonacci(self):
        assert PivotType.FIBONACCI.value == "fibonacci"

    def test_classic(self):
        assert PivotType.CLASSIC.value == "classic"


class TestPivotLevel:
    def test_supports(self):
        s = PivotLevel.supports()
        assert PivotLevel.P in s
        assert PivotLevel.S1 in s
        assert PivotLevel.R1 not in s

    def test_resistances(self):
        r = PivotLevel.resistances()
        assert PivotLevel.P in r
        assert PivotLevel.R1 in r
        assert PivotLevel.S1 not in r

    def test_ordering(self):
        assert PivotLevel.S3 < PivotLevel.S2
        assert PivotLevel.S1 < PivotLevel.P
        assert PivotLevel.P < PivotLevel.R1


class TestExitReason:
    def test_all_reasons(self):
        assert len(ExitReason) == 7

    def test_sl(self):
        assert ExitReason.SL.value == "SL"

    def test_tp(self):
        assert ExitReason.TP.value == "TP"

    def test_eod(self):
        assert ExitReason.EOD.value == "EOD"

    def test_data_end(self):
        assert ExitReason.DATA_END.value == "DATA_END"
