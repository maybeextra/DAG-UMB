from __future__ import print_function, absolute_import
import os.path as osp

import re


def get_imagedata_info(data):
    pids, cams = [], []
    for _, pid, camid, name in data:
        pids += [pid]
        cams += [camid]
    pids = set(pids)
    cams = set(cams)
    num_pids = len(pids)
    num_cams = len(cams)
    num_imgs = len(data)
    return num_pids, num_imgs, num_cams


def print_dataset_statistics(train, query, gallery):
    num_train_pids, num_train_imgs, num_train_cams = get_imagedata_info(train)
    num_query_pids, num_query_imgs, num_query_cams = get_imagedata_info(query)
    num_gallery_pids, num_gallery_imgs, num_gallery_cams = get_imagedata_info(gallery)

    print("Dataset statistics:")
    print("  ----------------------------------------")
    print("  subset   | # ids | # images | # cameras")
    print("  ----------------------------------------")
    print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
    print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
    print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
    print("  ----------------------------------------")


class MVTruck:
    dataset_dir_name = {
        'F': ['front'],
        'S': ['side'],
        'B': ['back'],
        'FS': ['front', 'side'],
        'FB': ['front', 'back'],
        'SB': ['side', 'back'],
        'FSB': ['front', 'side', 'back'],
    }
    base_folder = 'truck'
    def __init__(self, root='/home/xrs/备份', kind=None, verbose=True, **kwargs):
        super(MVTruck, self).__init__()
        self.root = osp.join(root, self.base_folder)
        self.name = self.dataset_dir_name[kind]

        dataset_dir = [osp.join(self.root, n) for n in self.name]
        self.train_dir = [osp.join(d_d, 'train') for d_d in dataset_dir]
        self.query_dir = [osp.join(d_d, 'query') for d_d in dataset_dir]
        self.gallery_dir = [osp.join(d_d, 'gallery') for d_d in dataset_dir]
        self._check_before_run()

        self.train = self._process_dir(kind='train', relabel=True)
        self.query = self._process_dir(kind='query', relabel=False)
        self.gallery = self._process_dir(kind='gallery', relabel=False)

        if verbose:
            print("=> MVTruck loaded")
            print_dataset_statistics(self.train, self.query, self.gallery)
            self.num_train_pids, self.num_train_imgs, self.num_train_cams = get_imagedata_info(self.train)
            self.num_query_pids, self.num_query_imgs, self.num_query_cams = get_imagedata_info(self.query)
            self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = get_imagedata_info(self.gallery)

    def _process_dir(self, kind, relabel=False):
        names_file_path = f'{self.root}/{kind}_names.txt'
        with open(names_file_path, 'r', encoding='utf-8') as file:
            names = file.readlines()
        names_list = [line.strip() for line in names]
        dirs = getattr(self, f'{kind}_dir')

        pattern = re.compile(r'([-\d]+)_c(\d+)')
        if relabel:
            pid_container = set()
            for name in names_list:
                pid, _ = map(int, pattern.search(name).groups())
                if pid == -1:
                    continue  # junk images are just ignored
                pid_container.add(pid)
                pid2label = {pid: label for label, pid in enumerate(pid_container)}

        result = []
        for name in names_list:
            pid, cid = map(int, pattern.search(name).groups())
            assert 1 <= cid <= 3
            cid -= 1  # index starts from 0

            if relabel:
                pid = pid2label[pid]

            img_path = [osp.join(d, f'{name}_{n[0].upper()}.jpg') for d, n in zip(dirs, self.name)]
            result.append((img_path, pid, cid, name))

        return result

    def _check_before_run(self):
        """Check if all files are available before going deeper for lists of directories"""
        # 检查训练目录列表
        for dir_path in self.train_dir:
            if not osp.exists(dir_path):
                raise RuntimeError("'{}' is not available".format(dir_path))

        # 检查查询目录列表
        for dir_path in self.query_dir:
            if not osp.exists(dir_path):
                raise RuntimeError("'{}' is not available".format(dir_path))

        # 检查图库目录列表
        for dir_path in self.gallery_dir:
            if not osp.exists(dir_path):
                raise RuntimeError("'{}' is not available".format(dir_path))


if __name__ == '__main__':
    dataset = MVTruck(kind='F')
