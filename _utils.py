


def float_to_str(val, scale=10, width=2):
    """
    Convert a float to a zero-padded string.
    val:   float, e.g., 0.1
    scale: multiply before converting to int, e.g., 10 for 1 decimal place, 100 for 2, etc.
    width: total string length (zero-padded)
    """
    return f"{int(round(val * scale)):0{width}d}"