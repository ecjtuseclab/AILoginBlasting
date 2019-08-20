# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 20:41:03 2018
@author: 康文洋
"""

import os
import re

class fileOperator:

    def getCurrentPath(self):
        path = os.getcwd()
        path = re.sub('\\\\', "/", path)
        #print(path)
        return path

