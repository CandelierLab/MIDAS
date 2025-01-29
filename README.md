# Toolbox_MIDAS
A python toolbox for *Multiple Information-Driven Agent Simulation* (MIDAS).

## Installation

```
sudo apt install nvidia-cuda-toolkit
pip3 install --upgrade pip
pip install numpy scipy numba matplotlib pyqt6 imageio[ffmpeg] colorama screeninfo alive-progress
```

+ Requires to set the Animation toolbox as an external source.

To remove the low occupancy warnings:

```
nano ~/Science/Projects/.virtual_environments/midas/bin/activate
```

And add at the end:

```
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
```