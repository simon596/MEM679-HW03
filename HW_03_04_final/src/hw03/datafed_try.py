# Import packages

import os
import getpass
import subprocess
from platform import platform
import sys
sys.path.append(r"C:\Users\sw3493\.conda\envs\MEM679\Lib\site-packages")
from datafed.CommandLib import API
from datafed import version as df_ver

try:
    datapath = os.mkdir("./datapath")
except:
    datapath = "./datapath"