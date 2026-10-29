import os
import numpy as np
from tqdm import tqdm
import h5py
from utils import read_list, read_nifti, config
from PIL import Image
import cv2
from scipy import misc
import scipy

config = config.Config("isic2018")
base_dir = config.base_dir



def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def process_npy():
    for tag in ['train', 'test']:
        img_list = []
        with open(base_dir+f'/{tag}.txt', 'r') as f:
            image_list = f.readlines()
        for img_id in tqdm(image_list):
            img_id = img_id.replace('\n', '')
            img_list.append(img_id)
            if tag == 'train':
                image_path = os.path.join(base_dir, "ISIC2018_Task1-2_Training_Input", 'ISIC_'+img_id + ".jpg")
                label_path = os.path.join(base_dir, "ISIC2018_Task1_Training_GroundTruth", 'ISIC_'+img_id + "_segmentation.png")
            if tag == 'test':
                image_path = os.path.join(base_dir, "ISIC2018_Task1-2_Validation_Input", 'ISIC_'+img_id + ".jpg")
                label_path = os.path.join(base_dir, "ISIC2018_Task1_Validation_GroundTruth", 'ISIC_'+img_id + "_segmentation.png")
            image = Image.open(image_path)#.convert('RGB')    
            label = Image.open(label_path)#.convert('L')
            image = np.array(image)
            label = np.array(label)
            print(label_path)
            '''print((label==0).sum())
            print((label>0).sum())
            print((label==255).sum())'''
            print(image.shape)
            print(label.shape)
            image = image.astype(np.float32)
            if not os.path.exists(os.path.join(config.save_dir, 'npy')):
                os.makedirs(os.path.join(config.save_dir, 'npy'))
            
            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id}_image'),
                image
            )
            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id}_label'),
                label
            )
            
            if not os.path.exists(os.path.join(config.save_dir, 'split_txts')):
                os.makedirs(os.path.join(config.save_dir, 'split_txts'))
            write_txt(
                img_list,
                os.path.join(config.save_dir, f'split_txts/{tag}.txt')
            )






def process_split_semi(split='train', labeled_ratio=0.05):
    ids_list = read_list(split, task="isic")
    split_idx = int(len(ids_list) * labeled_ratio)
    labeled_ids = ids_list[:split_idx]
    unlabeled_ids = ids_list[split_idx:]
    
    write_txt(
        labeled_ids,
        os.path.join(config.save_dir, f'split_txts/labeled_{labeled_ratio}.txt')
    )
    write_txt(
        labeled_ids,
        os.path.join(config.save_dir, f'split_txts/eval_{labeled_ratio}.txt')
    )
    write_txt(
        unlabeled_ids,
        os.path.join(config.save_dir, f'split_txts/unlabeled_{labeled_ratio}.txt')
    )


if __name__ == '__main__':
    process_npy()
    #process_split_semi(labeled_ratio=0.05)
    #process_split_semi(labeled_ratio=0.1)
