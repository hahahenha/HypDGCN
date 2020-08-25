#!/usr/bin/env bash
export HypDGCN_HOME=$(pwd)
export LOG_DIR="$HypDGCN_HOME/logs"
export PYTHONPATH="$HypDGCN_HOME:$PYTHONPATH"
export DATAPATH="$HypDGCN_HOME/data"
source activate HypDGCN  # replace with source HypDGSCN/bin/activate if you used a virtualenv
