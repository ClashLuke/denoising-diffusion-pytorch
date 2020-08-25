#!/bin/bash
import argparse
import multiprocessing
from functools import partial
from io import BytesIO

import rocksdb
from PIL import Image
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
import time

# WARNING: The code is expected to use 16 GiB just as DB cache. 
# If you don't have this much RAM available, make sure to decrease the buffer variables
# below (write_buffer_size, block_cache, block_cache_compressed).

PRINTERVALL = 1024

def resize_and_convert(img, size):
    buffer = BytesIO()
    trans_fn.center_crop(trans_fn.resize(img, size, Image.LANCZOS), size).save(buffer, format='jpeg', quality=100)
    return buffer.getvalue()


def resize_worker(img_file, sizes):
    img = Image.open(img_file[1]).convert('RGB')
    return img_file[0], [resize_and_convert(img, size) for size in sizes]


def prepare(db, dataset, n_worker, sizes=(128, 256, 512, 1024), buffer=64):
    resize_fn = partial(resize_worker, sizes=sizes)
    image_count = str(len(dataset))
    files = [(i, file) for i, (file, label) in enumerate(sorted(dataset.imgs, key=lambda x: x[0]), 1)]
    start = time.time()
    
    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in pool.imap_unordered(resize_fn, files):
            for size, img in zip(sizes, imgs):
                db.put(f'{size}-{i:05d}'.encode('utf-8'), img)
            if i%PRINTERVALL == 0:
                print(f"\r[{i:{len(image_count)}d}/{image_count}] Rate: {i/(time.time()-start):.2f} Img/s", end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sizes = (64 * 2 ** i * (j + 2) for i in range(4) for j in range(2))
    sizes = [s for s in sizes if s <= 1024]
    parser.add_argument('--out', type=str, default="out.lmdb")
    parser.add_argument('--size', type=str, default=','.join(map(str, sizes)))
    parser.add_argument('--n_worker', type=int, default=12)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.size.split(',')]

    print(f'Make dataset of image sizes:', ', '.join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    opts = rocksdb.Options(create_if_missing=True, 
                           write_buffer_size=2**32,
                           max_write_buffer_number=2**14,
                           max_open_files=-1,
                           target_file_size_base=2**26,
                           target_file_size_multiplier=4,
                           max_bytes_for_level_base=2**28,
                           manifest_preallocation_size=2**36,
                           table_factory=rocksdb.BlockBasedTableFactory(filter_policy=rocksdb.BloomFilterPolicy(10),
                                                                        block_cache=rocksdb.LRUCache(2 ** 33),
                                                                        block_cache_compressed=rocksdb.LRUCache(2 ** 32)
                                                                        )
                           )
    db = rocksdb.DB("out.rdb", opts)
    prepare(db, imgset, args.n_worker, sizes=sizes)
