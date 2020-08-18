import argparse
import multiprocessing
from functools import partial
from io import BytesIO

import lmdb
from PIL import Image
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm


def resize_and_convert(img, size, quality=100):
    img = trans_fn.resize(img, size, Image.LANCZOS)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()
    return val


def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = [resize_and_convert(img, size, 100) for size in sizes]
    return i, out


def prepare(env, dataset, n_worker, sizes=(128, 256, 512, 1024)):
    resize_fn = partial(resize_worker, sizes=sizes)
    files = [(i, file) for i, (file, label) in enumerate(sorted(dataset.imgs, key=lambda x: x[0]))]

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')

                with env.begin(write=True) as txn:
                    txn.put(key, img)

        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(i + 1).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--size', type=str, default='128,256,512,1024')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.size.split(',')]

    print(f'Make dataset of image sizes:', ', '.join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes)
