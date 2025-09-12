
```
python env/generate_initial_states.py     \
    --mesh_category longsleeve.json    \
    --path assets/multi-longsleeve-eval.hdf5    \
    --num_processes 1  \
    --trial_difficulty hard  \
    --num_trials 30    \
    --randomize_direction  \
    --random_translation 0.3  \
    --local_mode  \
    --scale 0.8
```

1. Fix the problem for `generate_initial_states.py`
2. mask-biased exploration might be needed for reducing the exploration space.