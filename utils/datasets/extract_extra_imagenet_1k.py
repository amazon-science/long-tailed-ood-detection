import os, argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='data root path')
args = parser.parse_args()

with open("datasets/imagenet_extra_1k_wnid_list.txt", "r") as fp:
	imagenet_extra_1k_wnid_list = fp.read().splitlines()

imagenet10k_dir = os.path.join(args.data_root_path, 'imagenet10k')
for item in os.listdir(imagenet10k_dir):
    wnid = item.split('.')[0]
    if wnid in imagenet_extra_1k_wnid_list:
        if os.path.exists(os.path.join(imagenet10k_dir, wnid)):
            continue
        else:
            os.makedirs(os.path.join(imagenet10k_dir, wnid))
            os.system('tar xvf %s -C %s' % (os.path.join(imagenet10k_dir, item), os.path.join(imagenet10k_dir, wnid)))

# make symbolic link:
imagenet_extra_1k_dir = os.path.join(args.data_root_path, 'imagenet_extra_1k')
os.mkdir(imagenet_extra_1k_dir)
for wnid in imagenet_extra_1k_wnid_list:
    os.symlink(os.path.join(imagenet10k_dir, wnid), os.path.join(imagenet_extra_1k_dir, wnid))
