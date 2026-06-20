import os
import cv2
import numpy as np

all_values = set()

for file in os.listdir("input/masked_images"):
    mask = cv2.imread(
        os.path.join("input/masked_images", file),
        cv2.IMREAD_GRAYSCALE
    )

    all_values.update(np.unique(mask))

print("All unique values:", sorted(all_values))
print("Number of classes:", len(all_values))