"""
Simulated light curve source (Lcurve wrapper stub).

Used when real survey data is unavailable — typically for training
on physics-based synthetic WDB light curves before survey data is
cross-matched and labeled.

Status: stub — Lcurve integration pending.
"""

from __future__ import annotations

from ml4em.types import LightCurve


class SimulatedSource:
    """Generate synthetic WDB light curves using Lcurve.

    Lcurve is Tom Marsh's White Dwarf Binary light curve modelling code.
    This source wraps it to produce realistic synthetic photometry for
    a given set of WDB physical parameters (masses, radii, inclination,
    limb-darkening coefficients, gravity darkening, etc.).

    Parameters
    ----------
    survey:
        Survey label to attach to output LightCurves.  Use "simulated".
    seed:
        Random seed for noise injection and observation-time sampling.

    Notes
    -----
    The planned fetch interface:
    - source_id is a path to an Lcurve model parameter file (.mod), or an
      integer index (as str) into a pre-generated parameter grid on disk.
    - The implementation will:
      1. Load the model parameters from disk.
      2. Call the Lcurve binary / Python wrapper to compute the model LC.
      3. Sample at realistic ZTF-like cadence timestamps (from a cadence library).
      4. Inject Gaussian photon noise scaled to the model's magnitude.
      5. Return a LightCurve with survey="simulated".
    """

    def __init__(self, survey: str = "simulated", seed: int = 42) -> None:
        self._survey = survey
        self._seed = seed

    # ------------------------------------------------------------------
    # Public interface  (satisfies LightCurveSource Protocol)
    # ------------------------------------------------------------------

    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        """Generate synthetic light curves for multiple Lcurve models.

        Parameters
        ----------
        source_ids:
            List of Lcurve model file paths or grid indices (as str).

        Returns
        -------
        list[LightCurve]
            Synthetic LightCurves for all requested models.
        """
        raise NotImplementedError(
            "SimulatedSource.fetch_batch is not yet implemented.\n"
            "Requires integration with Lcurve "
            "(Tom Marsh's WDB light curve modelling code)."
        )
