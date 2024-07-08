#!/bin/env python3

import json
import z3
from test import *
from util import *


if __name__ == '__main__':
    try:
        unittest.main()
    except Exception as e:
        print("Not pass tests")
        raise e

    print("You have passes all tests")
