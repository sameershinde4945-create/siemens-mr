"""
Siemens MRI – 2D Spin Echo Sequence (Production Grade)
Author  : Sameer Shinde
Reviewer: Senior MRI Software Engineer, Siemens Healthineers
Version : 2.0.0
Standard: IEC 60601-2-33 (MR Safety), Siemens ICE/IDEA conventions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import pypulseq as pp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("SiemensMRI.SpinEcho2D")

GAMMA_HZ_PER_T: Final[float] = 42.577e6
RF_PULSE_DURATION: Final[float] = 3e-3
RF_APODIZATION: Final[float] = 0.5
RF_TBW: Final[float] = 4.0
GX_FLAT_TIME: Final[float] = 4e-3


class Anatomy(str, Enum):
    BRAIN = "brain"
    KNEE  = "knee"
    ANKLE = "ankle"

    @classmethod
    def from_str(cls, value: str) -> "Anatomy":
        try:
            return cls(value.lower())
        except ValueError:
            supported = [a.value for a in cls]
            raise ValueError(
                f"Unsupported anatomy '{value}'. Supported: {supported}"
            ) from None


@dataclass(frozen=True)
class ScanProtocol:
    anatomy:         Anatomy
    fov:             float   # [m]
    matrix:          int
    slice_thickness: float   # [m]
    tr:              float   # [s]
    te:              float   # [s]
    n_slices:        int = 1

    @staticmethod
    def from_anatomy(anatomy: str, n_slices: int = 1) -> "ScanProtocol":
        anat = Anatomy.from_str(anatomy)
        base: dict = {
            Anatomy.BRAIN: dict(fov=0.22, matrix=256, slice_thickness=5e-3, tr=4.0, te=0.09),
            Anatomy.KNEE:  dict(fov=0.16, matrix=320, slice_thickness=3e-3, tr=3.5, te=0.07),
            Anatomy.ANKLE: dict(fov=0.16, matrix=256, slice_thickness=3e-3, tr=3.5, te=0.07),
        }[anat]
        return ScanProtocol(anatomy=anat, n_slices=n_slices, **base)

    def validate(self) -> None:
        assert self.te < self.tr,             "TE must be less than TR"
        assert self.fov > 0 and self.matrix > 0, "FOV and matrix must be positive"
        assert self.slice_thickness > 0,      "Slice thickness must be positive"
        log.info(
            "Protocol OK | anatomy=%s FOV=%.0fmm matrix=%d slice=%.1fmm TR=%.2fs TE=%.3fs",
            self.anatomy.value, self.fov*1e3, self.matrix,
            self.slice_thickness*1e3, self.tr, self.te,
        )


def build_system() -> pp.Opts:
    return pp.Opts(
        max_grad=80,        grad_unit="mT/m",
        max_slew=200,       slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


def build_rf_pulses(protocol: ScanProtocol, system: pp.Opts) -> tuple:
    rf90, gz90, gz90_reph = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        duration=RF_PULSE_DURATION,
        slice_thickness=protocol.slice_thickness,
        apodization=RF_APODIZATION,
        time_bw_product=RF_TBW,
        system=system,
        return_gz=True,
    )
    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=np.pi,
        duration=RF_PULSE_DURATION,
        slice_thickness=protocol.slice_thickness,
        apodization=RF_APODIZATION,
        time_bw_product=RF_TBW,
        system=system,
        return_gz=True,
    )
    return rf90, gz90, gz90_reph, rf180, gz180


def build_readout(protocol: ScanProtocol, system: pp.Opts) -> tuple:
    delta_k = 1.0 / protocol.fov
    gx = pp.make_trapezoid(
        channel="x",
        flat_area=protocol.matrix * delta_k,
        flat_time=GX_FLAT_TIME,
        system=system,
    )
    gx_pre = pp.make_trapezoid(
        channel="x",
        area=-gx.area / 2,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=protocol.matrix,
        duration=gx.flat_time,
        delay=gx.rise_time,
        system=system,
    )
    return gx_pre, gx, adc


def build_phase_encode(pe_index: int, n_pe: int,
                       protocol: ScanProtocol, system: pp.Opts) -> tuple:
    delta_k = 1.0 / protocol.fov
    pe_step = (pe_index - n_pe // 2) * delta_k
    gy      = pp.make_trapezoid(channel="y", area=pe_step,  system=system)
    gy_reph = pp.make_trapezoid(channel="y", area=-pe_step, system=system)
    return gy, gy_reph


def build_crusher(area: float, system: pp.Opts, channel: str = "z") -> pp.SimpleNamespace:
    return pp.make_trapezoid(channel=channel, area=area, system=system)


def build_spin_echo_2d(
    protocol: ScanProtocol,
    system: pp.Opts,
    output_path: str = "spin_echo_2d.seq",
) -> pp.Sequence:
    protocol.validate()
    seq = pp.Sequence(system)

    rf90, gz90, gz90_reph, rf180, gz180 = build_rf_pulses(protocol, system)
    gx_pre, gx, adc                     = build_readout(protocol, system)

    crusher_area  = 4 * gz180.area
    gz_crush_pre  = build_crusher(area=crusher_area, system=system)
    gz_crush_post = build_crusher(area=crusher_area, system=system)

    n_pe = protocol.matrix

    te_delay_1 = protocol.te / 2 - pp.calc_duration(gz90) / 2 - pp.calc_duration(gz_crush_pre)
    te_delay_2 = protocol.te / 2 - pp.calc_duration(gz_crush_post) - pp.calc_duration(gx) / 2

    if te_delay_1 < 0 or te_delay_2 < 0:
        raise ValueError(f"TE={protocol.te*1e3:.1f}ms too short for hardware limits.")

    log.info("Building k-space loop: %d PE lines", n_pe)

    for pe_idx in range(n_pe):
        gy_pe, gy_pe_reph = build_phase_encode(pe_idx, n_pe, protocol, system)

        seq.add_block(rf90, gz90)
        seq.add_block(gz90_reph, gy_pe, gx_pre)
        seq.add_block(pp.make_delay(max(te_delay_1, 0)))
        seq.add_block(gz_crush_pre)
        seq.add_block(rf180, gz180)
        seq.add_block(gz_crush_post)
        seq.add_block(pp.make_delay(max(te_delay_2, 0)))
        seq.add_block(gx, adc)
        seq.add_block(gy_pe_reph)

        tr_consumed = (
            pp.calc_duration(rf90, gz90)
            + pp.calc_duration(gz90_reph, gy_pe, gx_pre)
            + te_delay_1
            + pp.calc_duration(gz_crush_pre)
            + pp.calc_duration(rf180, gz180)
            + pp.calc_duration(gz_crush_post)
            + te_delay_2
            + pp.calc_duration(gx, adc)
            + pp.calc_duration(gy_pe_reph)
        )
        tr_delay = protocol.tr - tr_consumed
        if tr_delay < 0:
            raise ValueError(f"TR too short. Min TR = {tr_consumed:.3f}s")
        seq.add_block(pp.make_delay(tr_delay))

    ok, error_report = seq.check_timing()
    if not ok:
        raise RuntimeError(f"Timing validation failed: {error_report}")

    seq.write(output_path)
    log.info("Sequence written → '%s'", output_path)
    return seq


if __name__ == "__main__":
    system   = build_system()
    protocol = ScanProtocol.from_anatomy("knee", n_slices=1)
    seq      = build_spin_echo_2d(protocol, system, output_path="spin_echo_2d.seq")
