from time import time
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def init_seed():
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def preproc_data():
    from data import split_data
    split_data('/content/itr/itr/data/hin-eng/hin.txt', '/content/itr/itr/data/hin-eng')


from data import IndicDataset, PadSequence
import model as M


def gen_model_loaders(config):
    
    model, tokenizers = M.build_model(config)
    pad_sequence = PadSequence(tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)

    train_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, True), 
                            batch_size=config.batch_size, 
                            shuffle=False, 
                            collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, False), 
                           batch_size=config.eval_size, 
                           shuffle=False, 
                           collate_fn=pad_sequence)
    return model, tokenizers, train_loader, eval_loader

from config import replace, preEnc, preEncDec

   
class MyLightninModule(pl.LightningModule):

    
    def __init__(self):
        super().__init__()         
        self.model = gen_model_loaders(preEncDec)[0]      
        

    def forward(self, source,target):
        return self.model(source,target)
  
    def training_step(self, batch, batch_idx):
        # REQUIRED
        source, target = batch
        loss, logits = self.forward(source, target)
        return {'loss': loss}

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train",avg_loss,self.current_epoch)
        epoch_dictionary={'loss': avg_loss}
        return epoch_dictionary

    def validation_step(self,batch,batch_idx):

        source, target = batch
        loss, logits = self.forward(source, target)
        target = target.flatten()
        preds = torch.argmax(logits,2).flatten()
        return {'val_loss': loss, 'correct':(preds == target).float()}

    
    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.cat([x['correct'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'avg_val_loss': avg_loss,'avg_val_acc': acc, 'log': logs, 'progress_bar': logs}

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return gen_model_loaders(preEncDec)[2]

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return gen_model_loaders(preEncDec)[3]

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=preEncDec.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(gen_model_loaders(preEncDec)[2]), eta_min=preEncDec.lr)
        return [optimizer], [scheduler]

    def save(self):
      self.model.save(gen_model_loaders(preEncDec)[1], preEncDec.model_output_dirs)
  
    

from config import replace, preEnc, preEncDec


def main():
    
    rconf = preEncDec
    model, tokenizers, train_loader, eval_loader = gen_model_loaders(rconf)
    model = MyLightninModule()
    logger = TensorBoardLogger("tb_logs", name="my_modelFin")
    trainer = Trainer(
        max_epochs =rconf.epochs,
        gpus=[0],
        logger=logger
        # use_amp=False,
    )
    trainer.fit(model)
    model.save()

if __name__ == '__main__':
    #preproc_data()
    main()




