#!/usr/bin/env python3

from test import *
from util import *


if __name__ == '__main__':
    try:
        unittest.main()
    except Exception as e:
        print("Failed to pass all tests")
        raise e

    print("Passed all tests")
