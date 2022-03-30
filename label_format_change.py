import os


path = './yoloR_outputs_with_label/'

k = os.listdir(path)

out_path = './yoloR_labels/'

print(k)
for i in k:
    f = open(path+i,'r')
    lines = f.readlines()
    for line in lines:
        k = open(out_path+i,'a')
        k.write(i[:-4]+" "+line)

    f.close()
    k.close()