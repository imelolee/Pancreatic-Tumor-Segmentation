import config
import sys
import os
import random
import traceback
import numpy as np
import logging
from dataset.dataloader import PancreaticDataset
import setproctitle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.criterions import *
from models.unet2d.unet_model import UNet2D
from utils import read_split_data, create_lr_scheduler

BASE = os.getcwd()
setproctitle.setproctitle('train')



def train(model, optimizer, data_loader, device, epoch, lr_scheduler, writer):

    model.train()
    
    tversky_loss = []    
    loss_func = TverskyLoss2d()
    
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, targets = data

        targets = F.one_hot(targets, 2).permute(0, 4, 1, 2, 3).float().to(device)
        for slice in range(images.shape[1]):
            
            output = model(images[:, slice, ...].to(device))
            # output = model(images.to(device))
    
            loss = loss_func(output, targets[:, :, slice, ...]) 
        
            tversky_loss.append(loss.cpu().data.item())


        data_loader.desc = "[train epoch {}]".format(epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
   
    writer.add_scalar('tversky loss', np.average(tversky_loss), epoch)
    
    logging.info('[train epoch %d] Tversky Loss: %.4f ' % (epoch, np.average(tversky_loss)))
   
    torch.cuda.empty_cache()
    



def valid(model, data_loader, device, epoch, writer):
   
    model.eval()

    tversky_loss = [] 
    loss_func = TverskyLoss2d()
    
    with torch.no_grad():
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):    
            images, targets = data
            targets = F.one_hot(targets, 2).permute(0, 4, 1, 2, 3).float().to(device)
            for slice in range(images.shape[1]):
                
                output = model(images[:, slice, ...].to(device))
                # output = model(images.to(device))
                
                loss = loss_func(output, targets[:, :, slice, ...]) 
            
                tversky_loss.append(loss.cpu().data.item())


            data_loader.desc = "[valid epoch {}]".format(epoch)

    writer.add_scalar('tversky loss', np.average(tversky_loss), epoch) 
    
    logging.info('[valid epoch %d] Tversky Loss: %.4f ' % (epoch, np.average(tversky_loss)))
    
    torch.cuda.empty_cache()
    


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_img_list, train_label_list, val_img_list, val_label_list = read_split_data(train_dir=os.path.join(BASE, args.train_dir), 
                                                                                     val_dir=os.path.join(BASE, args.valid_dir))

    train_set = PancreaticDataset(ct_list=train_img_list,
                                  seg_list=train_label_list, 
                                  mode='train')
    val_set = PancreaticDataset(ct_list=val_img_list,
                                seg_list=val_label_list, 
                                mode='val')

    logging.info(f"Using {args.num_workers} dataloader workers every process")

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              drop_last=True, num_workers=args.num_workers, 
                              pin_memory=True, collate_fn=PancreaticDataset.collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,
                            drop_last=True, num_workers=args.num_workers, 
                            pin_memory=True, collate_fn=PancreaticDataset.collate_fn)

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {len(train_set)}
        Validation size: {len(val_set)}
        Checkpoints:     {args.checkpoint}
        Device:          {device.type}
       
    ''')

    model = UNet2D().to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                 weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    

    
    start_epoch = 0
    
    initial_checkpoint = args.checkpoint
    if initial_checkpoint:
        logging.info(f'Model loaded from {initial_checkpoint}')
        checkpoint = torch.load(initial_checkpoint, map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        state = model.state_dict()
        state.update(checkpoint['state_dict'])

        try:
            model.load_state_dict(state)
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            logging.info(f'Load something failed!')
            traceback.print_exc()

    start_epoch = start_epoch + 1

    epochs = args.epochs
    
    model_out_dir = os.path.join(args.results_dir,'weights')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
        
    tb_out_dir = os.path.join(args.results_dir, 'runs')
    tb_writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    
    for epoch in range(start_epoch, epochs + 1):
        # train
        train(
            model=model, 
            optimizer=optimizer,
            data_loader=train_loader, 
            device=device,
            epoch=epoch, 
            lr_scheduler=lr_scheduler,
            writer=train_writer
        )

        # validate
        valid(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            writer=val_writer
        )

          
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'state_dict': state_dict, 'optimizer' : optimizer.state_dict()},
                    os.path.join(model_out_dir, '%03d.pth' % epoch))
            logging.info(f'Checkpoint {epoch} saved!')
    
    tb_writer.close()
    train_writer.close()
    val_writer.close()
    
    logging.info('Finished Training')


if __name__ == '__main__':
    
    args = config.args
    main(args)
