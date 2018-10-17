#!/usr/bin/python
import os,sys
SURESNET_DIR = os.path.dirname(os.path.abspath(__file__))
SURESNET_DIR = os.path.dirname(SURESNET_DIR)
sys.path.insert(0,SURESNET_DIR)
from sparse_uresnet import flags

def main():
  cfg = flags.FLAGS()
  cfg.parse_args()
  
if __name__ == '__main__':
  main()

