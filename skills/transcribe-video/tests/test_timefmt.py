import timefmt


def test_fmt_ts_does_not_emit_60_seconds():
    # 59.6s must render as 59 seconds, never carry to ":60"
    assert timefmt.fmt_ts(59.6) == "00:00:59,600"


def test_fmt_ts_rounds_milliseconds_not_seconds():
    assert timefmt.fmt_ts(0.0) == "00:00:00,000"
    assert timefmt.fmt_ts(3661.250) == "01:01:01,250"


def test_fmt_ts_hour_boundary():
    assert timefmt.fmt_ts(3599.999) == "00:59:59,999"
    assert timefmt.fmt_ts(3600.0) == "01:00:00,000"


def test_fmt_ts_negative_clamps_to_zero():
    assert timefmt.fmt_ts(-5.0) == "00:00:00,000"


def test_fmt_clock_compact():
    assert timefmt.fmt_clock(0) == "000000"
    assert timefmt.fmt_clock(3661) == "010101"
    assert timefmt.fmt_clock(59.9) == "000059"  # truncates, no carry
