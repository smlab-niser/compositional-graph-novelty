"""
Utility metrics and statistical functions
"""

import numpy as np
from typing import Dict, List


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of values

    Args:
        values: List of numeric values

    Returns:
        Dictionary with mean, std, median, min, max, quartiles
    """
    values = np.array(values)

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'count': len(values)
    }
