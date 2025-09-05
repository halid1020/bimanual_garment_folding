# cd ../agent-arena-v0
# . ./setup.sh

# if [ -d "../softgym" ]; then
#   cd ../softgym
#   . ./setup.sh
# else
#   echo "Directory ../softgym does not exist. Skipping."
# fi

# cd ../LaGarNet

# export PYTHONPATH=${PWD}:$PYTHONPATH

#!/bin/bash

cd ../agent-arena-v0
. ./setup.sh

if [ -d "../softgym" ]; then
  cd ../softgym
  . ./setup.sh
else
  echo "Directory ../softgym does not exist. Skipping."
fi

cd ../bimanual_garment_folding

export PYTHONPATH=${PWD}:$PYTHONPATH

if [ "$1" == "real-world" ]; then
  cd ../ws_ur3e
  source ./agar_build/install/setup.sh
  cd src/robot_control_cloth
  . ./setup.sh
  cd ../../../LaGarNet
fi
