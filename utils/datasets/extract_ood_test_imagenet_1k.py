import os, argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='data root path')
args = parser.parse_args()

with open("datasets/imagenet_ood_test_1k_wnid_list.txt", "r") as fp:
	imagenet_ood_test_1k_wnid_list = fp.read().splitlines()

imagenet10k_dir = os.path.join(args.data_root_path, 'imagenet10k')
for item in os.listdir(imagenet10k_dir):
    wnid = item.split('.')[0]
    if wnid in imagenet_ood_test_1k_wnid_list:
        if os.path.exists(os.path.join(imagenet10k_dir, wnid)):
            continue
        else:
            os.makedirs(os.path.join(imagenet10k_dir, wnid))
            os.system('tar xvf %s -C %s' % (os.path.join(imagenet10k_dir, item), os.path.join(imagenet10k_dir, wnid)))

# make symbolic link:
imagenet_ood_test_1k_dir = os.path.join(args.data_root_path, 'imagenet_ood_test_1k')
for wnid in imagenet_ood_test_1k_wnid_list:
    if not os.path.exists(os.path.join(imagenet_ood_test_1k_dir, wnid)):
        os.makedirs(os.path.join(imagenet_ood_test_1k_dir, wnid))
    c = 0
    file_name_list = os.listdir(os.path.join(imagenet10k_dir, wnid))
    file_name_list.sort()
    for item in file_name_list:
        if '.JPEG' in item:
            os.symlink(os.path.join(imagenet10k_dir, wnid, item), os.path.join(imagenet_ood_test_1k_dir, wnid, item))
            c += 1
        if c >= 50:
            break
