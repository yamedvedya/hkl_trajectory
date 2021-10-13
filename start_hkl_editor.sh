#!/usr/bin/env bash
BASEDIR=`dirname $0`
cd $BASEDIR ||exit

export VIEWERPATH=$PWD/
export PYTHONPATH=$PYTHONPATH:$VIEWERPATH
python3 ./main.py