import numpy as np
import torch

import sys
from PIL import Image, ImageDraw

def rectangle(output_path):
    image = Image.new("RGB", (400, 400), "blue")
    draw = ImageDraw.Draw(image)
    draw.rectangle((200, 100, 300, 200), fill="red")
    draw.rectangle((50, 50, 150, 150), fill="green", outline="yellow",
                   width=3)
    image.save(output_path)


if __name__ == '__main__':
    # patch0 = torch.load('patch_folder/0.pt')
    # patch1 = torch.load('patch_folder/1.pt')
    # patch2 = torch.load('patch_folder/2.pt')
    # patch3 = torch.load('patch_folder/3.pt')
    # patch4 = torch.load('patch_folder/4.pt')
    original_patch = torch.load('patch_folder/original_patches.pt')

    best_patch = torch.load('patch_folder/123.pt')

    print(original_patch.size())
    print(best_patch[0, 0, :, 1, 1])

    with Image.open("patch_folder/000104_left.png") as im:

        draw = ImageDraw.Draw(im)
        for i in range(96):
            v1 = best_patch[0, i, 0:2, 0, 0]*4
            v2 = best_patch[0, i, 0:2, 0, 2]*4
            v3 = best_patch[0, i, 0:2, 2, 0]*4
            v4 = best_patch[0, i, 0:2, 2, 2]*4
            # print(v1, v2, v3, v4)
            draw.rectangle((v1[0], v1[1], v4[0], v4[1]), fill="red")

            # v1 = patch0[0, i, 0:2, 0, 0]
            # v2 = patch0[0, i, 0:2, 0, 2]
            # v3 = patch0[0, i, 0:2, 2, 0]
            # v4 = patch0[0, i, 0:2, 2, 2]
            # draw.rectangle((v1[0], v1[1], v4[0], v4[1]), fill="blue")
            # # print(v1, v2, v3, v4)
        for i in range(96):
            v1 = original_patch[0, i, 0:2, 0, 0]*4
            v2 = original_patch[0, i, 0:2, 0, 2]*4
            v3 = original_patch[0, i, 0:2, 2, 0]*4
            v4 = original_patch[0, i, 0:2, 2, 2]*4
            # print(v1, v2, v3, v4)
            draw.rectangle((v1[0], v1[1], v4[0], v4[1]), fill="blue")
        # write to stdout
        im.save('patch_folder/image_with_patches.png')