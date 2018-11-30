'''
sky          = ( 128, 128, 128 )
building     = (   0,   0, 128 )
pole         = ( 128, 192, 192 )
road_marking = (   0,  69, 255 )
road         = ( 128,  64, 128 )
pavement     = ( 222,  40,  60 )
tree         = (   0, 128, 128 )
sign_symbol  = ( 220, 220,   0 )
fence        = ( 128,  64,  64 )
car          = ( 128,   0,  64 )
pedestrian   = (   0,  64,  64 )
bicyclist    = ( 192, 128,   0 )
unlabeled    = (   0,   0,   0 )
palette = (
    sky,
    building,
    pole,
    road_marking,
    road,
    pavement,
    tree,
    sign_symbol,
    fence,
    car,
    pedestrian,
    bicyclist,
    unlabeled
)
'''
# segnet for cityscapes (RGB)
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
'''

# segnet for cityscapes (BGR)
palette = (
    (180,130, 70),  #sky
    ( 70, 70, 70),  #building,
    (153,153,153),  #pole,
    (255, 69,  0),  #road_marking,
    (128, 64,128),  #road,
    (232, 35,244),  #pavement,
    ( 35,142,107),  #tree,
    (  0,220,220),  #sign_symbol,
    (153,153,190),  #fence,
    (142,  0,  0),  #car,
    ( 60, 20,220),  #pedestrian,
    ( 32, 11,119),  #bicyclist,
    (  0,  0,  0),  #unlabeled
)