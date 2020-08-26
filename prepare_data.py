import argparse
import multiprocessing
import time
from functools import partial
from io import BytesIO

import rocksdb
from PIL import Image
from torchvision import datasets
from torchvision.transforms import functional as trans_fn

PRINTERVALL = 1024

# WARNING: The code is expected to use 32 GiB (it never goes above 3.4GB) just as DB cache. 
# If you don't have this much RAM available, make sure to decrease the buffer variables
# below (write_buffer_size, block_cache, block_cache_compressed). 
# Changing it to 8GB reduces the speed by ~3%.

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
            if i % PRINTERVALL == 0:
                print(f"\r[{i:{len(image_count)}d}/{image_count}] Rate: {i / (time.time() - start):.2f} Img/s", end='')


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
    opts = rocksdb.Options()
    opts.create_if_missing = True
    opts.write_buffer_size = 2 ** 34
    opts.max_write_buffer_number = 2 ** 14
    opts.max_open_files = -1
    opts.compression = rocksdb.CompressionType.no_compression
    opts.target_file_size_base = 2 ** 26
    opts.target_file_size_multiplier = 4
    opts.max_bytes_for_level_base = 2 ** 28
    opts.manifest_preallocation_size = 2 ** 36
    opts.table_factory = rocksdb.BlockBasedTableFactory(filter_policy=rocksdb.BloomFilterPolicy(32),
                                                        block_cache=rocksdb.LRUCache(2 ** 34, 32),
                                                        )
    db = rocksdb.DB("out.rdb", opts)
    prepare(db, imgset, args.n_worker, sizes=sizes)
