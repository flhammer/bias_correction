import numpy as np
from numpy._typing import NDArray


def dem_txt_to_numpy(file_path: str) -> NDArray:
    """
    Convert a DEM (Digital Elevation Model) text file to a NumPy array.

    This function reads a text file containing elevation data formatted with
    metadata (e.g., number of rows and columns) in the header, followed by rows
    of numerical elevation data. The header is skipped, and the data is parsed
    into a NumPy array.

    Parameters
    ----------
    file_path : str
        Path to the DEM text file.

    Returns
    -------
    NDArray
        A 2D NumPy array containing the elevation data.

    Raises
    ------
    ValueError
        If the number of columns in any row of the data does not match the
        expected number of columns (as specified in the file's header).

    Notes
    -----
    - The file is expected to have a header with at least the following two lines:
        - "ncols <number_of_columns>"
        - "nrows <number_of_rows>"
      followed by at least 4 additional lines to be skipped.
    - Each row of data must have the same number of columns, as specified in the
      "ncols" line in the header.
    - Lines with zero-length rows are ignored.

    Examples
    --------
    Assuming a DEM text file `example.dem` with the following content:

    ```
    ncols 4
    nrows 3
    xllcorner 0.0
    yllcorner 0.0
    cellsize 1.0
    NODATA_value -9999
    1.0 2.0 3.0 4.0
    5.0 6.0 7.0 8.0
    9.0 10.0 11.0 12.0
    ```

    You can load the file as follows:

    >>> import numpy as np
    >>> dem_data = dem_txt_to_array("example.dem")
    >>> print(dem_data)
    [[ 1.  2.  3.  4.]
     [ 5.  6.  7.  8.]
     [ 9. 10. 11. 12.]]
    """
    with open(file_path, "r") as f:

        ncols = int(f.readline().strip().split()[-1])
        nrows = int(f.readline().strip().split()[-1])
        skip_rows = 4
        for _ in range(skip_rows):
            next(f)

        rows = []
        for i, line in enumerate(f):
            row = list(map(float, line.strip().split()))
            if len(row) == 0:
                continue
            rows.append(row)
            if len(row) != ncols and len(row) > 0:
                raise ValueError(
                    f"The provided file has {len(row)} columns instead of {ncols} on line {i + 7}."
                )
    return np.array(rows)
