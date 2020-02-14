#!/src/sh

set -xe

virtualenv venv
source venv/src/activate
pip install -r requirements.txt
pip install git+https://github.com/tensorflow/docs
