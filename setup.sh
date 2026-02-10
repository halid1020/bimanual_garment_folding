conda activate mp-fold
if [ -d "../softgym" ]; then
  cd ../softgym
  . ./setup.sh
else
  echo "Directory ../softgym does not exist. Skipping."
fi

cd ../bimanual_garment_folding

export PYTHONPATH=${PWD}:$PYTHONPATH
export MP_FOLD_PATH=${PWD}
export REAL_ROBOT_PATH="${PWD}/real_robot"