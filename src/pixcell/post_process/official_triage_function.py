from typing import Any, Dict, Mapping

TRIAGE_REQUIRED_KEYS = {
    "V_EDH",
    "V_SDH",
    "V_IPH",
    "V_SAH",
    "V_IVH",
    "fracture_prob",
    "MLS_mm",
}


def validate_intermediates(
    intermediates: Mapping[str, Any],
) -> Dict[str, float]:
    """
    Validate and normalize intermediate values for a single CT series.

    Ensures that:
    - All required keys are present.
    - No unexpected keys are provided.
    - Every value can be converted to float.

    Args:
        intermediates: Dictionary of intermediate values.

    Returns:
        Dictionary with identical keys and float values.

    Raises:
        ValueError:
            If required keys are missing or unexpected keys are present.

        TypeError:
            If any value cannot be converted to float.
    """

    missing = TRIAGE_REQUIRED_KEYS - intermediates.keys()
    extra = intermediates.keys() - TRIAGE_REQUIRED_KEYS

    if missing:
        raise ValueError(
            f"Missing keys: {sorted(missing)}. "
            f"Expected: {sorted(TRIAGE_REQUIRED_KEYS)}."
        )

    if extra:
        raise ValueError(
            f"Unexpected keys: {sorted(extra)}. "
            f"Expected: {sorted(TRIAGE_REQUIRED_KEYS)}."
        )

    cleaned: Dict[str, float] = {}

    for key in TRIAGE_REQUIRED_KEYS:
        try:
            cleaned[key] = float(intermediates[key])
        except Exception as exc:
            raise TypeError(
                f"Value for '{key}' must be convertible to float, "
                f"got {type(intermediates[key]).__name__}."
            ) from exc

    return cleaned


def triage_from_intermediates(
    intermediates: Mapping[str, Any],
) -> int:
    """
    Compute triage label from intermediate imaging measurements.

    Returns:
        0 -> Non-urgent
        1 -> Urgent
        2 -> Critical
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------

    vals = validate_intermediates(intermediates)

    # ------------------------------------------------------------------
    # Extract measurements
    # ------------------------------------------------------------------

    V_EDH = max(0.0, vals["V_EDH"])
    V_SDH = max(0.0, vals["V_SDH"])
    V_IPH = max(0.0, vals["V_IPH"])
    V_SAH = max(0.0, vals["V_SAH"])
    V_IVH = max(0.0, vals["V_IVH"])

    MLS_mm = max(0.0, vals["MLS_mm"])

    fracture_prob = vals["fracture_prob"]

    total_vol = (
        V_EDH
        + V_SDH
        + V_IPH
        + V_SAH
        + V_IVH
    )

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    EPS_VOLUME = 0.1
    EPS_MLS = 1.0

    MLS_CRITICAL = 5.0
    MLS_URGENT_LOW = 3.0

    EDH_CRIT = 30.0
    SDH_CRIT = 70.0
    IPH_CRIT = 70.0

    TOTAL_VOL_CRIT = 60.0

    COMBO_MLS = 3.0
    COMBO_VOL = 40.0

    FRAC_VOL_CRIT = 15.0

    FRACTURE_THRESHOLD = 0.5

    # ------------------------------------------------------------------
    # Derived flags
    # ------------------------------------------------------------------

    has_ich = total_vol >= EPS_VOLUME
    mls_present = MLS_mm >= EPS_MLS
    fracture_present = fracture_prob >= FRACTURE_THRESHOLD

    # ==================================================================
    # Critical (2)
    # ==================================================================

    # High MLS with hemorrhage or fracture
    if MLS_mm >= MLS_CRITICAL and (has_ich or fracture_present):
        return 2

    # Large EDH
    if V_EDH >= EDH_CRIT:
        return 2

    # Large SDH
    if V_SDH >= SDH_CRIT:
        return 2

    # Large IPH
    if V_IPH >= IPH_CRIT:
        return 2

    # Large total hemorrhage
    if total_vol >= TOTAL_VOL_CRIT:
        return 2

    # Moderate hemorrhage + MLS
    if has_ich and MLS_mm >= COMBO_MLS and total_vol >= COMBO_VOL:
        return 2

    # Fracture with substantial hemorrhage
    if fracture_present and total_vol >= FRAC_VOL_CRIT:
        return 2

    # ==================================================================
    # Urgent (1)
    # ==================================================================

    # High MLS without hemorrhage or fracture
    if MLS_mm >= MLS_CRITICAL and not (has_ich or fracture_present):
        return 1

    # Any meaningful hemorrhage
    if has_ich:
        return 1

    # Moderate MLS
    if MLS_URGENT_LOW <= MLS_mm < MLS_CRITICAL:
        return 1

    # Fracture alone / fracture with small hemorrhage
    if fracture_present:
        return 1

    # Small hemorrhage with measurable MLS
    if total_vol >= EPS_VOLUME and mls_present:
        return 1

    # ==================================================================
    # Non-urgent (0)
    # ==================================================================

    return 0