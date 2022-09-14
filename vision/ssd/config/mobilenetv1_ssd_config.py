import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 640
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# # For 640x640
specs = [
    SSDSpec(40, 16, SSDBoxSizes(64, 128), [2, 3]),
    SSDSpec(20, 32, SSDBoxSizes(128, 236.8), [2, 3]),
    SSDSpec(10, 64, SSDBoxSizes(236.8, 345.6), [2, 3]),
    SSDSpec(5, 128, SSDBoxSizes(345.6, 454.4), [2, 3]),
    SSDSpec(3, 214, SSDBoxSizes(454.4, 563.2), [2, 3]),
    SSDSpec(2, 320, SSDBoxSizes(563.2, 672.0), [2, 3])
]

# For 512x512
# specs = [
#     SSDSpec(32, 16, SSDBoxSizes(20, 35), [2, 3]),
#     SSDSpec(16, 32, SSDBoxSizes(35, 50), [2, 3]),
#     SSDSpec(8, 64, SSDBoxSizes(50, 65), [2, 3]),
#     SSDSpec(4, 100, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(285, 512), [2, 3])
# ]

# For 300x300
"""
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]
"""

priors = generate_ssd_priors(specs, image_size)


def set_image_size(size=640, min_ratio=20, max_ratio=90):
    global image_size
    global specs
    global priors
    
    from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
    
    import torch
    import math
    import logging
        
    image_size = size
    ssd = create_mobilenetv1_ssd(num_classes=3) # TODO does num_classes matter here?
    x = torch.randn(1, 3, image_size, image_size)
    
    feature_maps = ssd(x, get_feature_map_size=True)
    
    steps = [
        math.ceil(image_size * 1.0 / feature_map) for feature_map in feature_maps
    ]
    step = int(math.floor((max_ratio - min_ratio) / (len(feature_maps) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(image_size * ratio / 100.0)
        max_sizes.append(image_size * (ratio + step) / 100.0)
    min_sizes = [image_size * (min_ratio / 2) / 100.0] + min_sizes
    max_sizes = [image_size * min_ratio / 100.0] + max_sizes
    
    # this update logic makes different boxes than the original for 300x300 (but better for power-of-two)
    # for backwards-compatibility, keep the default 300x300 config if that's what's being called for
    if image_size != 300:
        specs = []
        
        for i in range(len(feature_maps)):
            # ssd-mobilenet-* aspect ratio is [2,3]
            specs.append(SSDSpec(feature_maps[i], steps[i], SSDBoxSizes(min_sizes[i], max_sizes[i]), [2, 3]))

    logging.info(f'model resolution {image_size}x{image_size}')
    for spec in specs:
        logging.info(str(spec))
    
    priors = generate_ssd_priors(specs, image_size)

#
# def set_image_width_height(width, height, min_ratio=20, max_ratio=90):
#     global image_size
#     global specs
#     global priors
#
#     from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
#
#     import torch
#     import math
#     import logging
#
#     ssd = create_mobilenetv1_ssd(num_classes=3)  # TODO does num_classes matter here?
#     x = torch.randn(1, 3, height, width)
#
#     feature_maps = ssd(x, get_feature_map_size=True)
#
#     steps = [
#         math.ceil(width * 1.0 / feature_map) for feature_map in feature_maps
#     ]
#     step = int(math.floor((max_ratio - min_ratio) / (len(feature_maps) - 2)))
#     min_sizes = []
#     max_sizes = []
#     for ratio in range(min_ratio, max_ratio + 1, step):
#         min_sizes.append(image_size * ratio / 100.0)
#         max_sizes.append(image_size * (ratio + step) / 100.0)
#     min_sizes = [image_size * (min_ratio / 2) / 100.0] + min_sizes
#     max_sizes = [image_size * min_ratio / 100.0] + max_sizes
#
#     specs = []
#     for i in range(len(feature_maps)):
#         # ssd-mobilenet-* aspect ratio is [2,3]
#         specs.append(SSDSpec(feature_maps[i], steps[i], SSDBoxSizes(min_sizes[i], max_sizes[i]), [2, 3]))
#
#     logging.info(f'model resolution {image_size}x{image_size}')
#     for spec in specs:
#         print(str(spec))
#         # logging.info(str(spec))
#
#     priors = generate_ssd_priors(specs, image_size)

#print(' ')
#print('SSD-Mobilenet-v1 priors:')
#print(priors.shape)
#print(priors)
#print(' ')

#import torch
#torch.save(priors, 'mb1-ssd-priors.pt')

#np.savetxt('mb1-ssd-priors.txt', priors.numpy())
