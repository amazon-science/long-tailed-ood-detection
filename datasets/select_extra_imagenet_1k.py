from genericpath import isdir
import os, argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='data root path')
args = parser.parse_args()

imagenet_train_dir = os.path.join(args.data_root_path, 'imagenet', 'train')
imagenet1k_wnid_list = os.listdir(imagenet_train_dir)
assert len(imagenet1k_wnid_list)==1000, "%d is not 1000" % len(imagenet1k_wnid_list)

imagenet_o_dir = os.path.join(args.data_root_path, 'imagenet-o')
imagenet_o_wnid_list = []
for item in os.listdir(imagenet_o_dir):
    if 'README' not in item:
        imagenet_o_wnid_list.append(item.split('.')[0])

imagenet10k_dir = os.path.join(args.data_root_path, 'imagenet10k')
imagenet10k_wnid_list = []
for item in os.listdir(imagenet10k_dir):
    if '.tar' in item:
        imagenet10k_wnid_list.append(item.split('.')[0])

imagenet_extra_1k_wnid_list = [] # 200 classes
for wnid in imagenet10k_wnid_list:
    if wnid not in imagenet1k_wnid_list and wnid not in imagenet_o_wnid_list:
        imagenet_extra_1k_wnid_list.append(wnid)
        if len(imagenet_extra_1k_wnid_list) == 500:
            break

fp = open('./datasets/imagenet_extra_1k_wnid_list.txt', 'w+')
# fp = open('./imagenet_extra_1k_wnid_list.txt', 'w+')
for wnid in imagenet_extra_1k_wnid_list:
    fp.write(wnid)
    fp.write('\n')
    fp.flush()
