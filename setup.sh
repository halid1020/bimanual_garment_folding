conda activate mp-fold
if [ -d "../softgym" ]; then
  cd ../softgym
  . ./setup.sh
else
  echo "Directory ../softgym does not exist. Skipping."
fi

cd ../bimanual_garment_folding

export PYTHONPATH=${PWD}:$PYTHONPATH

cd ../agent-arena-v0/agent_arena

export AGENT_ARENA_PATH=${PWD}
export RAVENS_ASSETS_DIR=${AGENT_ARENA_PATH}/arena/raven/environments/assets

cd ../../bimanual_garment_folding

