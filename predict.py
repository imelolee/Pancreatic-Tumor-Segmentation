from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
# from utils import logger,common
from dataset.dataloader import PancreaticDataset
import SimpleITK as sitk
import os
import numpy as np
import setproctitle
from models.TransPTS.TransPTS import PTS
from models.unet3d.unet_model import UNet
from collections import OrderedDict
from utils import read_split_data
import matplotlib.pyplot as plt


BASE = os.getcwd()
setproctitle.setproctitle('test')


def predict_one_img(model, filename, result_save_path, device, args):
    
    model.eval()
    img = PancreaticDataset.load(filename)
    img = img.to(device)
    
    with torch.no_grad():
        output = model(img, img.shape[2:])   
        
        # output = model(img)     
        pred = torch.argmax(output,dim=1).squeeze(0).cpu()
    
        # for i in range(len(pred)):
        #     plt.figure()
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(pred[i], cmap="bone")
        #     plt.savefig(result_save_path + "/{}.png".format(i))
        #     plt.close()
        
        pred = np.asarray(pred.numpy(),dtype='uint8')
        
        pred_img = sitk.GetImageFromArray(pred)
        save_name = filename.split('/')[-1][:-4]
        sitk.WriteImage(pred_img, os.path.join(result_save_path,  save_name + '_pred.nii.gz'))
        print(save_name + " done.")

        return pred

if __name__ == '__main__':
    args = config.args
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    # model info
    # model = UNet().to(device)
    model = PTS().to(device)
    
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # test_log = logger.Test_Logger(save_path,"test_log")
    # data info
    result_save_path = os.path.join(args.results_dir, 'pred_mask')
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    _, _, test_img_list, test_label_list = read_split_data(train_dir=os.path.join(BASE, args.train_dir), 
                                                            val_dir=os.path.join(BASE, args.valid_dir))
    
    for img in test_img_list:
    
        pred_img = predict_one_img(model, img, result_save_path, device, args)
    # for img_dataset,file_idx in datasets:
    #     test_dice,pred_img = predict_one_img(model, img_dataset, args)
    #     test_log.update(file_idx, test_dice)
    #     sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx+'.gz'))