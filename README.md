# Toolbox_MIDAS
A python toolbox for *Multiple Information-Driven Agent Simulation* (MIDAS).

It includes the RIPO and RINNO models.

## Installation via Conda:

```
conda install numpy numba matplotlib pyqt imageio[ffmpeg] 
```

+ requires the Animation Toolbox.

To remove the low occupancy warnings:

```
conda env config vars set NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
```