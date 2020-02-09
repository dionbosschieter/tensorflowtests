#!/bin/sh

set -xe

virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/tensorflow/docs
