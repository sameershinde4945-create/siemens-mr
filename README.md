"""
Siemens MRI – 2D Spin Echo Sequence (Production Grade)
Author  : Sameer Shinde
Reviewer: Senior MRI Software Engineer, Siemens Healthineers
Version : 2.0.0
Standard: IEC 60601-2-33 (MR Safety), Siemens ICE/IDEA conventions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

import numpy as np
import pypulseq as pp

# ---------------------------------------------------------------------------
# Logging — use Siemens-style structured logging in production pipelines
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("SiemensMRI.SpinEcho2D")

# ---------------------------------------------------------------------------
# Constants — never scatter magic numbers across the code
# ---------------------------------------------------------------------------
GAMMA_HZ_PER_T: Final[float] = 42.577e6       # ¹H gyromagnetic ratio [Hz/T]
RF_PULSE_DURATION: Final[float] = 3e-3         # [s]  sinc pulse duration
RF_APODIZATION: Final[float] = 0.5            # Hanning window factor
RF_TBW: Final[float] = 4.0                    # Time-bandwidth product
GX_FLAT_TIME: Final[float] = 4e-3             # [s]  readout flat time


# ---------------------------------------------------------------------------
# Anatomy Enum — eliminates silent string-typo bugs
# ---------------------------------------------------------------------------
class Anatomy(str, Enum):
    BRAIN = "brain"
    KNEE = "knee"
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


# ---------------------------------------------------------------------------
# Protocol — typed, validated, self-documenting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanProtocol:
    anatomy: Anatomy
    fov: float          # [m]   Field of view
    matrix: int         # []    Frequency-encoding matrix size (NX = NY assumed)
    slice_thickness: float  # [m]
    tr: float           # [s]   Repetition time
    te: float           # [s]   Echo time
    n_slices: int = 1

    @staticmethod
    def from_anatomy(anatomy: str, n_slices: int = 1) -> "ScanProtocol":
        anat = Anatomy.from_str(anatomy)
        base: dict = {
            Anatomy.BRAIN: dict(
                fov=0.22, matrix=256, slice_thickness=5e-3, tr=4.0, te=0.09
            ),
            Anatomy.KNEE: dict(
                fov=0.16, matrix=320, slice_thickness=3e-3, tr=3.5, te=0.07
            ),
            Anatomy.ANKLE: dict(
                fov=0.16, matrix=256, slice_thickness=3e-3, tr=3.5, te=0.07
            ),
        }[anat]
        return ScanProtocol(anatomy=anat, n_slices=n_slices, **base)

    def validate(self) -> None:
        """Basic SAR / timing sanity — real Siemens SW has deeper checks."""
        assert self.te < self.tr, "TE must be less than TR"
        assert self.fov > 0 and self.matrix > 0, "FOV and matrix must be positive"
        assert self.slice_thickness > 0, "Slice thickness must be positive"
        log.info(
            "Protocol validated | anatomy=%s FOV=%.0fmm matrix=%d "
            "slice=%.1fmm TR=%.2fs TE=%.3fs",
            self.anatomy.value,
            self.fov * 1e3,
            self.matrix,
            self.slice_thickness * 1e3,
            self.tr,
            self.te,
        )


# ---------------------------------------------------------------------------
# Scanner system limits
# ---------------------------------------------------------------------------
def build_system() -> pp.Opts:
    """
    Abstracts scanner gradient / RF hardware limits.
    Tune per Siemens platform:
      - MAGNETOM Vida  : max_grad=45, max_slew=200
      - MAGNETOM Terra : max_grad=80, max_slew=200
    """
    return pp.Opts(
        max_grad=80,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,   # [s] coil ringdown
        rf_dead_time=100e-6,      # [s] transmit/receive switch
        adc_dead_time=10e-6,      # [s] ADC settling
    )


# ---------------------------------------------------------------------------
# Gradient builders — isolated for unit-testability
# ---------------------------------------------------------------------------
def build_rf_pulses(
    protocol: ScanProtocol, system: pp.Opts
) -> tuple[pp.SimpleNamespace, pp.SimpleNamespace, pp.SimpleNamespace,
           pp.SimpleNamespace, pp.SimpleNamespace]:
    """
    Returns (rf90, gz90, gz90_reph, rf180, gz180).
    gz90_reph  : rephases slice-selection dephasing after 90° pulse.
    Crusher GZ : applied before/after 180° to spoil stimulated echoes.
    """
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


def build_readout(
    protocol: ScanProtocol, system: pp.Opts
) -> tuple[pp.SimpleNamespace, pp.SimpleNamespace, pp.SimpleNamespace]:
    """
    Returns (gx_pre, gx, adc).
    gx_pre : pre-phaser that moves to the start of k-space before readout.
    """
    delta_k: float = 1.0 / protocol.fov  # k-space step [1/m]

    gx = pp.make_trapezoid(
        channel="x",
        flat_area=protocol.matrix * delta_k,
        flat_time=GX_FLAT_TIME,
        system=system,
    )

    # Pre-phaser: area = -0.5 * gx flat area, runs in negative x
    gx_pre = pp.make_trapezoid(
        channel="x",
        area=-gx.area / 2,        # CRITICAL: centres echo in readout window
        system=system,
    )

    adc = pp.make_adc(
        num_samples=protocol.matrix,
        duration=gx.flat_time,
        delay=gx.rise_time,
        system=system,
    )

    return gx_pre, gx, adc


def build_phase_encode(
    pe_index: int,
    n_pe: int,
    protocol: ScanProtocol,
    system: pp.Opts,
) -> tuple[pp.SimpleNamespace, pp.SimpleNamespace]:
    """
    Returns (gy_pe, gy_pe_reph) for one phase-encode line.
    gy_pe_reph refocuses PE gradient after acquisition (prepares steady state).
    """
    delta_k: float = 1.0 / protocol.fov
    # Step from -n_pe/2 to +n_pe/2 - 1
    pe_step = (pe_index - n_pe // 2) * delta_k

    gy_pe = pp.make_trapezoid(
        channel="y",
        area=pe_step,
        system=system,
    )
    gy_pe_reph = pp.make_trapezoid(
        channel="y",
        area=-pe_step,     # Rewind PE dephasing
        system=system,
    )
    return gy_pe, gy_pe_reph


def build_crusher(
    area: float,
    system: pp.Opts,
    channel: str = "z",
) -> pp.SimpleNamespace:
