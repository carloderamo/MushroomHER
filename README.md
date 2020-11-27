# MushroomHER
A simple, readable, and complete, implementation of Hindsight Experience Replay and its variants.

How to run
----------
The `run.py` script uses the library `mpi4py` to do multithreading, as described in [Hindsight Experience Replay](#https://arxiv.org/abs/1707.01495). To run a simple
experiment, execute:

```
mpirun --use-hwthread-cpus -n 1 python3 run.py --name FetchReach-v1.
```

To run a simple an advanced experiment, using 19 threads for 38 train episodes, execute:

```
mpirun --use-hwthread-cpus -n 19 python run.py --name FetchPush-v1 --train-episodes 38
```


