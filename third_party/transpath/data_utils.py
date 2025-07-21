import numpy as np
import cv2


def resize_image(data, image_size, mode='image'):
    assert mode in ['image', 'coordinate']

    if not isinstance(data, list):
        data = [data]

    data_resized = []

    if mode == 'image':
        for data_per in data:
            org_shape = data_per.shape
            new_shape = (*org_shape[:-2], image_size, image_size)
            h, w = org_shape[-2:]

            data_per = data_per.reshape(-1, h, w)
            num_samples = data_per.shape[0]
            div_size = 100
            num_divs = (num_samples + div_size - 1) // div_size
            data_clip_list = []
            data_per = data_per.transpose((1, 2, 0)).astype(np.float32)

            for i in range(num_divs):
                data_clip = data_per[:, :, i * div_size : min((i + 1) * div_size, num_samples)]
                data_clip = cv2.resize(data_clip, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR_EXACT)
                data_clip = data_clip.reshape(image_size, image_size, -1)
                data_clip_list.append(data_clip)
            
            data_clip = np.concatenate(data_clip_list, axis=-1).transpose((2, 0, 1))
            data_resized.append(data_clip.reshape(new_shape))
    elif mode == 'coordinate':
        for data_per in data:
            org_img_size = 128
            factor = image_size / org_img_size
            data_per = np.floor(data_per * factor).astype(np.int64)
            data_resized.append(data_per)

    return data_resized