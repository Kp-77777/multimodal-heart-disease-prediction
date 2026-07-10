from .probability_checks import validate_probability


def late_fusion(P_tab, P_echo, alpha=0.6, as_category=False, thresholds=(0.40, 0.60)):
    """
    Late fusion of independent modalities producing either a probability or
    a categorical risk level.

    Parameters
    - P_tab: P(Heart Failure | Tabular)
    - P_echo: P(Heart Failure | ECHO)
    - alpha: weight for ECHO (default 0.6)
    - as_category: if True, return one of ('low','medium','high') instead of probability
    - thresholds: tuple (low_threshold, high_threshold) with 0 <= low < high <= 1

    Returns
    - float probability if `as_category` is False (default)
    - str category ('low', 'medium', 'high') if `as_category` is True
    """
    p_tab = validate_probability(P_tab, "Tabular probability")
    p_echo = validate_probability(P_echo, "Echo probability")

    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")

    beta = 1 - alpha
    P_multi = alpha * p_echo + beta * p_tab
    p_multi = float(P_multi)

    if as_category:
        try:
            low_thr, high_thr = thresholds
        except Exception:
            raise TypeError("thresholds must be a tuple (low_threshold, high_threshold)")

        if not (0.0 <= low_thr < high_thr <= 1.0):
            raise ValueError("thresholds must satisfy 0.0 <= low < high <= 1.0")

        if p_multi < low_thr:
            return "low"
        elif p_multi < high_thr:
            return "medium"
        else:
            return "high"

    return p_multi
