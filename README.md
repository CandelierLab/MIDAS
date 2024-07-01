# Toolbox_MIDAS
A python toolbox for *Multiple Information-Driven Agent Simulation* (MIDAS).

It includes the RIPO and RINNO models.

## Installation via Conda:

```
conda install numpy scipy numba cudatoolkit matplotlib pyqt imageio[ffmpeg] colorama
```

NB: for older cuda toolkit versions:
```
conda install cudatoolkit=11.3.1
```

```
pip install screeninfo alive-progress
```

+ requires the Animation Toolbox.

To remove the low occupancy warnings:

```
conda env config vars set NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
```