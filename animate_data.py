#%%
import dedalus.public as d3
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import argparse
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, type=str)
args = parser.parse_args()

DATA_NAME = args.file
DATA_PATH = f"Data/{DATA_NAME}"
n_files = np.sum([os.path.isfile(f"{DATA_PATH}/{f}") for f in os.listdir(DATA_PATH)])
FILE_PATHS = [f"{DATA_PATH}/{DATA_NAME}_s{i}.h5" for i in range(1, n_files+1)]

Ra = float(DATA_NAME[DATA_NAME.index('Ra_')+3:DATA_NAME.index('_Ta')].replace('pt', '.'))
Ta = float(DATA_NAME[DATA_NAME.index('Ta_')+3:].replace('pt', '.'))

tasks = [d3.load_tasks_to_xarray(file, tasks=['buoyancy', 'w']) for file in FILE_PATHS]

b = xr.concat([task['buoyancy'] for task in tasks], dim='t')
w = xr.concat([task['w'] for task in tasks], dim='t')

w_rms = np.sqrt((w**2).mean(dim=["x", "z"]))

z = b.z
x = b.x


# Function to update the plot for each frame
def update(frame):
    plt.clf()
    ax1 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    ax1 = plt.imshow(b.isel(t=frame).T, extent=(np.amin(x), np.amax(x), np.amin(z), np.amax(z)), origin='lower', aspect='auto', cmap="RdBu_r", vmin=np.amin(b), vmax=np.amax(b))
    plt.colorbar(label='b')
    plt.title(f'No Stress, Buoyancy Value, Ra = {Ra}, Ta = {Ta} \n Time: {np.format_float_scientific(b.t[frame].values, precision=3)}')
    plt.xlabel('x')
    plt.ylabel('z')

    ax2 = plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.plot(b.t.values, w_rms)
    plt.yscale('log')
    plt.axvline(b.t[frame].values)
    plt.xlabel("t")
    plt.ylabel("rms(w)")

    plt.tight_layout()

fps = 30

# Create the figure and animation
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=len(b.t), interval=1000/fps, repeat=False)

# Save the animation as an mp4 file
ani.save(f'Data/{DATA_NAME}.mp4', writer='ffmpeg', dpi=500, fps=fps)

# Display the animation (optional)
plt.show()
#%%