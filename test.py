# import torch

# k = torch.tensor([1,2,3,4])

# print(k.tolist())


# 
# import shutil
# import os

# path_yolo = './yoloR_labels/'
# path_gt = './retinaface_labels/'
# path_result = './retina_labels/'

# yolo_labels = os.listdir(path_yolo)
# gt_labels = os.listdir(path_gt)

# for i in yolo_labels:
#     for j in gt_labels:
#         if i==j:
#             shutil.move(path_gt+j,path_result+j)



# import os
# import shutil
# path = './retinaface_labels/'

# retina = os.listdir(path)

# path_res = './reti/'
# for i in retina:

#     f = open(path+i,'r')
#     k = f.readlines()
#     m = open(path_res+i,'w')
#     for a in k:
#         a = a.split()
#         a.insert(1,'0')
#         print(f'{a[0]} {a[1]} {a[2]} {a[3]} {a[4]} {a[5]}\n')
#         m.write(f'{a[0]} {a[1]} {a[2]} {a[3]} {a[4]} {a[5]}\n')
#     m.close()



