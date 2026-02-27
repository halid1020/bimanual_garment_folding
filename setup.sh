source ~/anaconda3/bin/activate
conda activate mp-fold-old
if [ -d "../softgym" ]; then
  cd ../softgym
  . ./setup.sh
else
  echo "Directory ../softgym does not exist. Skipping."
fi

cd ../bimanual_garment_folding

export PYTHONPATH=${PWD}:$PYTHONPATH
export AGENT_ARENA_PATH='../agent_arena_v0/agent_arena'

