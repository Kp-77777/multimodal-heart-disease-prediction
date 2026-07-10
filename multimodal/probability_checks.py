def validate_probability(p, name="probability"):
    """Validate that `p` is a numeric probability in [0, 1].

    This accepts Python numeric types as well as numpy/torch scalar types
    by attempting to coerce to a Python float.
    """
    try:
        p_val = float(p)
    except Exception:
        raise TypeError(f"{name} must be numeric")

    if not (0.0 <= p_val <= 1.0):
        raise ValueError(f"{name} must be in [0,1], got {p}")

    return p_val
