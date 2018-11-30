@echo off

python  test_segmentation.py ^
    --testset      D:\works\cityscapes\segnet\segnet-float\result/out_maps/*.npy  ^
    --labelset     D:\ownCloud\segnet-label\segnet_label/*.png  ^
    --palette      palette/palette.py  ^
    --save_dir     D:\works\cityscapes\segnet\segnet-float\result/visualize  ^
    --visualize


pause