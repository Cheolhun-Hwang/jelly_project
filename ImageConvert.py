import numpy as np
import pandas as pd
from PIL import Image

dataset = pd.read_csv('D:/data/csv/dataset/nm_nom_image/bu_jelly_nom_nm_byte.csv')
data_columns = list(dataset.columns.values)
X = dataset[dataset.columns[0]]
y = dataset[dataset.columns[1]]

X_s = np.array(X)

for main in range(len(X_s)):
    line = X_s[main]
    img = Image.new('1', (25, 70))
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = 0

    subLine = line.split("&")
    print(len(subLine))
    for i in range(len(subLine)) :
        text = subLine[i]
        if(len(text) < 1):
            continue
        else :
            for j in range(len(text)) :
                char = text[j]
                # print(text, char)
                print(i, j)
                pixels[i, j] = int(char)

    # img.show()
    img.save('D:/data/csv/dataset/nm_nom_image/item_'+str(main)+'.bmp')