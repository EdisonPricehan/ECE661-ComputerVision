#!/usr/bin/env python

import sys

if sys.version_info[0] == 3:
    from NonlinearLeastSquares.NonlinearLeastSquares import __version__
    from NonlinearLeastSquares.NonlinearLeastSquares import __author__
    from NonlinearLeastSquares.NonlinearLeastSquares import __date__
    from NonlinearLeastSquares.NonlinearLeastSquares import __url__
    from NonlinearLeastSquares.NonlinearLeastSquares import __copyright__

    from NonlinearLeastSquares.NonlinearLeastSquares import NonlinearLeastSquares

else:
    from NonlinearLeastSquares import __version__
    from NonlinearLeastSquares import __author__
    from NonlinearLeastSquares import __date__
    from NonlinearLeastSquares import __url__
    from NonlinearLeastSquares import __copyright__

    from NonlinearLeastSquares import NonlinearLeastSquares

