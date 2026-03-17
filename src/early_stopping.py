import torch

class EarlyStopping:
    def __init__(self,patience=7,verbose=True,delta=0,checkpoint_path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.checkpoint_path = checkpoint_path

    def __call__(self,val_loss,model,optimizer,epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,optimizer,epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,optimizer,epoch)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        if self.verbose:
            print(f'  验证 loss 下降 ({self.val_loss_min:.4f} → {val_loss:.4f})，保存 checkpoint')
        if self.checkpoint_path is not None:
            torch.save({
                'epoch'           : epoch,
                'model_state'     : model.state_dict(),
                'optimizer_state' : optimizer.state_dict(),
                'val_loss'        : val_loss,
            }, self.checkpoint_path)
        self.val_loss_min = val_loss