import argparse
import multiprocessing
from functools import partial
from io import BytesIO

import lmdb
from PIL import Image
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
import time

PRINTERVALL = 1024

def resize_and_convert(img, size):
    buffer = BytesIO()
    trans_fn.center_crop(trans_fn.resize(img, size, Image.LANCZOS), size).save(buffer, format='jpeg', quality=100)
    return buffer.getvalue()


def resize_worker(img_file, sizes):
    img = Image.open(img_file[1]).convert('RGB')
    return img_file[0], [resize_and_convert(img, size) for size in sizes]


def open_db(name):
    def _make():
        return lmdb.open(name, map_size=1e12, readahead=False, writemap=True)
    return _make

def prepare(out, dataset, n_worker, sizes=(128, 256, 512, 1024), buffer=64):
    resize_fn = partial(resize_worker, sizes=sizes)
    image_count = str(len(dataset))
    files = [(i, file) for i, (file, label) in enumerate(sorted(dataset.imgs, key=lambda x: x[0]), 1)]
    start = time.time()
    make_db = open_db(out)
    db = make_db()
    txn = db.begin(write=True)

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in pool.imap_unordered(resize_fn, files):
            for size, img in zip(sizes, imgs):
                txn.put(f'{size}-{i:05d}'.encode('utf-8'), img)
            if i%PRINTERVALL == 0:
                print(f"\r[{i:{len(image_count)}d}/{image_count}] Rate: {i/(time.time()-start):.2f} Img/s", end='')
            if i%buffer == 0:
                txn.commit()
                db.close()
                db = make_db()
                txn = db.begin(write=True)

        txn.put('length'.encode('utf-8'), str(i + 1).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="out.lmdb")
    parser.add_argument('--size', type=str, default=','.join(map(str, (64 * 2 ** i * (j + 2) for i in range(4) for j in range(2)))))
    parser.add_argument('--n_worker', type=int, default=12)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.size.split(',')]

    print(f'Make dataset of image sizes:', ', '.join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    prepare(args.out, imgset, args.n_worker, sizes=sizes)
