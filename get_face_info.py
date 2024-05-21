import cv2
import numpy as np
from PIL import Image
import pickle
from diffusers.utils import load_image
from insightface.app import FaceAnalysis


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image



root = "model_zoo/instantID/"
app = FaceAnalysis(name='antelopev2', root=root, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

app.prepare(ctx_id=0, det_size=(640, 640))


face_image = load_image("./examples/wenyongshan.png")
# face_image = load_image("./examples/zhuyilong.jpg")


face_image = resize_img(face_image)
face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
face_info = sorted(face_info, key=lambda x :(x['bbox'][2 ] -x['bbox'][0] ) *x['bbox'][3 ] -x['bbox'][1])[-1] # only use the maximum face

print(type(face_info))

# 创建一个示例字典，其值是 dtype 为 float32 的 NumPy 数组
data_dict = {
    'embedding': face_info['embedding'],
    'kps': face_info['kps']
}

# 使用 pickle 序列化字典并保存到文件
with open('examples/face_info.pkl', 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)