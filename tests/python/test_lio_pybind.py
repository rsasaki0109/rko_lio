import pytest

from rko_lio.lio import LIO, LIOConfig
from rko_lio.rko_lio_pybind import (
    _TimestampProcessingConfig,
    _VectorDouble,
    _process_timestamps,
)


def test_lioconfig_creation():
    config = LIOConfig()
    assert config is not None


def test_lioconfig_attributes():
    config = LIOConfig(max_range=50.0)
    config.voxel_size = 2.0
    config.deskew = False

    assert config.voxel_size == 2.0
    assert config.max_range == 50.0
    assert config.deskew is False


def test_lio_init_with_config():
    config = LIOConfig()
    lio_obj = LIO(config)
    assert lio_obj is not None


@pytest.mark.parametrize(
    ("raw_timestamps", "header_stamp_ns", "force_absolute"),
    [
        ([0.0, 0.05, 0.1], 100_000_000_000, False),
        ([100.0, 100.05, 100.1], 100_000_000_000, True),
    ],
)
def test_timestamp_offset_applies_after_absolute_time_conversion(
    raw_timestamps, header_stamp_ns, force_absolute
):
    config = _TimestampProcessingConfig()
    config.offset_seconds = -0.1
    config.force_absolute = force_absolute
    config.force_relative = not force_absolute

    start, end, timestamps = _process_timestamps(
        _VectorDouble(raw_timestamps), header_stamp_ns, config
    )

    assert start == 99_900_000_000
    assert end == 100_000_000_000
    assert list(timestamps) == [99_900_000_000, 99_950_000_000, 100_000_000_000]
