from pathlib import Path
from PIL import Image

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).parent

def read_result():
    results = []
    with open(SCRIPT_PATH / "result.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Traceback" in line:
                print(f"Image-{len(results)} does not exist")
                results.append((-1, -1))
            elif "<answer>" in line:
                two_nums = line.split("<answer>", 1)[1].split("</answer>", 1)[0].strip().strip("()").split(",")
                try:
                    x = int(two_nums[0].strip())
                    y = int(two_nums[1].strip())
                    results.append((x, y))
                except ValueError:
                    print(f"Invalid answer format in line: {line}")
                    results.append((-1, -1))
                
    return results
                

def verify(results):
    tested = 0
    correct = 0
    for i in range(len(results)):
        if results[i] == (-1, -1):
            continue
        mask_img = SCRIPT_PATH / "mask" / f"image-{i}.png"
        mask_img = cv2.imread(str(mask_img), cv2.IMREAD_GRAYSCALE)
        shape = mask_img.shape
        if results[i][0] >= shape[1] or results[i][1] >= shape[0]:
            tested += 1
            continue
        if mask_img[results[i][1], results[i][0]] >= 0:
            correct += 1
        tested += 1
    print(f"Correct: {correct}/{tested} ({correct/tested*100:.2f}%)")
    
if __name__ == "__main__":
    verify(read_result())
    
    