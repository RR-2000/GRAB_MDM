import numpy as np
import os, glob, subprocess
from tqdm import tqdm


def make_gif(base_path = './', render_type = 'combined'):

    # Base Paths
    preds_base = './predicts'
    renders_base = './renders'
    gif_base = './gifs'
    preds_paths = os.listdir(preds_base)
    print(preds_paths)
    pred_seqs = []


    for idx in tqdm(range(len(preds_paths))):
        i = preds_paths[idx]
        data = i.split('_')
        if len(data) < 3:
            continue
        
        pred_seqs.append(os.path.join(renders_base, render_type, data[0], f'{data[1]}_{data[2]}'))
        # print(pred_seqs[-1])


        cmd = f'ffmpeg -i {pred_seqs[-1]}/%0004d.png {os.path.join(gif_base, i[:-4])}.gif'
        # print(cmd)

        subprocess.call(cmd.split(' '))


if __name__ == "__main__":
    make_gif()