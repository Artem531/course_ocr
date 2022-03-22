from trainer.trainer import Trainer
from dataset.cards import BCDataset
from model.centernet import CenterNet
from config.hw_config import Config
from loss.loss import Loss, JointLoss
from torch.utils.data import DataLoader
from data import MidvPackage
from pathlib import Path

def train(cfg):
    data_path = Path(cfg.root)
    data_packs = MidvPackage.read_midv500_dataset(data_path)

    test_split = []
    train_split = []

    for dp in data_packs:
        for i in range(len(dp)):
            if dp[i].is_test_split():
                if dp[i].is_quad_inside():
                    test_split.append(dp[i])
            else:
                if dp[i].is_quad_inside():
                    train_split.append(dp[i])

    train_ds = BCDataset(train_split)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True)

    eval_ds = BCDataset(test_split)

    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True)

    model = CenterNet(cfg)
    if cfg.gpu:
        model = model.cuda()
    loss_func = JointLoss(cfg)

    epoch = 100
    cfg.max_iter = len(train_dl) * epoch
    cfg.steps = (int(cfg.max_iter * 0.6), int(cfg.max_iter * 0.8))
    trainer = Trainer(cfg, model, loss_func, train_dl, eval_dl)
    trainer.train()

if __name__ == '__main__':
    cfg = Config
    train(cfg)
