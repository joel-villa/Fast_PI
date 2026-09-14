"""Just a helper online average method"""

import scipy

def online_avg(
        old_avg:scipy.sparse,
        new_val:scipy.sparse,
        new_total:int,
) -> scipy.sparse:
    """Generate the new average given the old + the new + the new total count

    Args:
        old_avg (scipy.sparse): The current average
        new_val (scipy.sparse): The value which is updating the average
        new_total (int): The grand total ammount

    Returns:
        scipy.sparse: The new average
    """

    return old_avg + ((new_val - old_avg) / new_total)