import cv2
import os
import numpy as np
import glob
'''
 0  sky          = ( 128, 128, 128 )    23
 1  building     = (   0,   0, 128 )    11-12, 15-16
 2  pole         = ( 128, 192, 192 )    17-18
 3  road_marking = (   0,  69, 255 )
 4  road         = ( 128,  64, 128 )    7, 9-10
 5  pavement     = ( 222,  40,  60 )    8
 6  tree         = (   0, 128, 128 )    21-22
 7  sign_symbol  = ( 128, 128, 192 )    19-20
 8  fence        = ( 128,  64,  64 )    13-14
 9  car          = ( 128,   0,  64 )    25-32
10  pedestrian   = (   0,  64,  64 )    24
11  bicyclist    = ( 192, 128,   0 )    33
12  unlabeled    = (   0,   0,   0 )    # 0-6
'''


palette = (
    ( 70,130,180),  #sky
    ( 70, 70, 70),  #building,
    (153,153,153),  #pole,
    (  0, 69,255),  #road_marking,
    (128, 64,128),  #road,
    (244, 35,232),  #pavement,
    (107,142, 35),  #tree,
    (220,220,  0),  #sign_symbol,
    (190,153,153),  #fence,
    (  0,  0,142),  #car,
    (220, 20, 60),  #pedestrian,
    (119, 11, 32),  #bicyclist,
    (  0,  0,  0),  #unlabeled
)

table = np.zeros((34,), dtype=np.uint16)
table[:] = 12
table[ 7] = 4
table[ 8] = 5
table[ 9] = 4
table[10] = 4
table[11] = 1
table[12] = 1
table[13] = 8
table[14] = 8
table[15] = 1
table[16] = 1
table[17] = 2
table[18] = 2
table[19] = 7
table[20] = 7
table[21] = 6
table[22] = 6
table[23] = 0
table[24] = 10
table[25] = 11
table[26] = 9
table[27] = 9
table[28] = 9
table[29] = 9
table[30] = 9
table[31] = 9
table[32] = 11
table[33] = 11



def replace(class_map):
    assert class_map.ndim == 2
    canvas = np.ndarray(class_map.shape, dtype='uint8')
    for label in range(len(table)):
        canvas[class_map == label] = table[label]
    return canvas

def visualize(class_map, palette, save_file):
    assert class_map.ndim == 2
    map_height, map_width = class_map.shape
    canvas = np.ndarray((map_height, map_width, 3), dtype='uint8')
    for label in range(len(palette)):
        canvas[class_map == label] = palette[label][::-1]
    cv2.imwrite(save_file, canvas)


ids = 'ids/*.png'
colors = 'color/*.png'
id_files = sorted(glob.glob(ids))
color_files = sorted(glob.glob(colors))
assert 0 < len(id_files)
assert 0 < len(color_files)

max_ = []
for id_file, color_file in zip(id_files, color_files):
    label = cv2.imread(id_file, cv2.IMREAD_GRAYSCALE)
    replaced = replace(label)
    cv2.imwrite('segnet_label/{}'.format(os.path.basename(id_file)), replaced)
    visualize(replaced, palette, 'segnet_color/{}'.format(os.path.basename(color_file)))
    max_.append(replaced.max())
print max_