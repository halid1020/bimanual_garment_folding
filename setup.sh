if [ -d "../softgym" ]; then
  cd ../softgym
  . ./setup.sh
else
  echo "Directory ../softgym does not exist. Skipping."
fi

cd ../bimanual_garment_folding

export PYTHONPATH=${PWD}:$PYTHONPATH
