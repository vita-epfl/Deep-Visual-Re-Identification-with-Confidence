# encoding: utf-8
"""
@author:  weijian
@contact: dengwj16@gmail.com
"""

import glob
import re
import xml.dom.minidom as XD
import os.path as osp
import os

from .bases import BaseImageDataset


class Wild(BaseImageDataset):
    """

    VeRi

    """
    dataset_dir = 'vehicle/veri-wild'
    test_small = "test_3000"
    test_medium = "test_5000"
    test_large = "test_10000"
    def __init__(self, root='/data/linah-data/', verbose=True, split_wild="large", **kwargs):
        super(Wild, self).__init__()
        if split_wild == "small":
            chosen_test = self.test_small
        elif split_wild == "medium":
            chosen_test = self.test_medium
        elif split_wild == "large":
            chosen_test = self.test_large
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_path = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.query_path = osp.join(self.dataset_dir, 'train_test_split/'+chosen_test+'_query.txt')
        self.gallery_path = osp.join(self.dataset_dir, 'train_test_split/'+chosen_test+'.txt')
        self.veh_info = osp.join(self.dataset_dir, 'train_test_split/vehicle_info.txt')

        self._check_before_run()

        train = self._process_dir_test(self.train_path, self.dataset_dir, self.veh_info, relabel=True)
        query = self._process_dir_test(self.query_path, self.dataset_dir, self.veh_info, relabel=False)
        gallery = self._process_dir_test(self.gallery_path, self.dataset_dir, self.veh_info, relabel=False)

        if verbose:
            print("=> Wild loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        #if not osp.exists(self.train_path):
         #   raise RuntimeError("'{}' is not available".format(self.train_path))
        #if not osp.exists(self.query_path):
         #   raise RuntimeError("'{}' is not available".format(self.query_path))
        #if not osp.exists(self.gallery_path):
         #   raise RuntimeError("'{}' is not available".format(self.gallery_path))


    def _process_dir_test(self, list_path, img_folder, info_folder, relabel=False):
        img_paths = [] #glob.glob(osp.join(img_folder,'**/*.jpg'))
        for root, dirs, files in os.walk(img_folder):
            for f in files :
                if f.endswith('.jpg'):
                    img_paths.append(osp.join(root,f))
        #for dir_ in os.listdir(img_folder):
         #   for f in glob.glob('*.jpg'):
          #      img_paths.append(osp.join(img_folder,f))

        pid_container = set()
        with open (list_path) as pth :
            path = pth.readlines()
            for p in path :
                pid,imgid = p.split('/')
                if pid == -1: continue  # junk images are just ignored
                pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []

        info_container = []
        id_container = []
        with open(info_folder) as inf :
            info = inf.readlines()
            #print(len(info))
            for f in info: #organized as : id/image;Camera ID;Time;Model;Type;Color
                camid = f.split(';')[1]
                pid_imgid = f.split(';')[0]
                pid,imgid = pid_imgid.split('/')
                id_=(pid,imgid)
                info_container.append(camid)
                id_container.append(id_)
            all_container = list(zip(id_container,info_container))
            #print('all_container contains '+str(len(all_container)))
            pid2camid = {id_ : camid for id_,camid in all_container} #dictionnary that gets the camid of the picture from the pid and imgid that are in the vehicule info text file


        for img_path in img_paths:
            path_split = img_path.split('/')
            image_split = path_split[-1].split('.') #for removing the extension
            pid_p,imgid_p = path_split[-2],image_split[0]
            id_p = (pid_p, imgid_p)
            if pid_p == -1: continue  # junk images are just ignored
            if pid_p not in pid2label: continue
            if id_p not in pid2camid: continue
            camid = int(camid)
            camid -= 1  # index starts from 0
            if relabel:
                pid_new = pid2label[pid_p]
                dataset.append((img_path, pid_new, pid2camid[id_p]))
            else :
                dataset.append((img_path, pid_p, pid2camid[id_p]))
        #print('In '+str(list_path)+' there are '+str(len(dataset))+' images.')

        return dataset
