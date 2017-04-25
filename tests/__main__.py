"""
"""

import os
import sys
import nose


if __name__ == "__main__":
    module_name = sys.modules[__name__].__file__
    # print("sys.argv = ", sys.argv)

    # Run the full directory
    module_name = os.path.split(module_name)[0]
    nose_args = [sys.argv[0], module_name]
    if len(sys.argv) > 1:
        nose_args.extend(sys.argv[1:])
    # print("nose_args = ", nose_args)
    result = nose.run(argv=nose_args)
