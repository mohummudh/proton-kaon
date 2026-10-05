"""Check exact training parity and preservation of the paper's split contract."""
import copy
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from run_study import run_epoch,digest
from src.models.build import build_vae
from src.train.train import train
from src.losses.vae import vae_loss

class ContractTests(unittest.TestCase):
    def test_run_plan(self):
        tasks=json.loads((HERE/'plan.json').read_text())
        self.assertEqual(len(tasks),187)
        self.assertEqual(len({t['id'] for t in tasks}),len(tasks))
        base=yaml.safe_load(next((HERE.parents[1]/'configs').glob('run_0093*')).read_text())
        for task in tasks:
            cfg=yaml.safe_load((HERE/task['config']).read_text())
            self.assertEqual(cfg['model']['channels'],base['model']['channels'])
            self.assertEqual(cfg['optimizer'],base['optimizer'])
            self.assertEqual(cfg['train']['batch_size'],32)
            self.assertEqual(cfg['train']['epochs'],200)
            self.assertEqual(cfg['train']['beta'],task['beta'])
            splitpath=HERE/'splits'/f"split_all_{task['tag']}.npz"
            self.assertEqual(digest(splitpath),task['split_sha256'])
        main=np.load(HERE/'splits/split_all_bal9419.npz')
        self.assertEqual(len(main['train_idx']),9419)
        self.assertEqual(len(main['val_idx']),18238)
        self.assertFalse(np.intersect1d(main['train_idx'],main['val_idx']).size)
        counts=[np.count_nonzero((main['train_idx']>=lo)&(main['train_idx']<hi))
                for lo,hi in [(0,10466),(10466,18693),(18693,27657)]]
        self.assertEqual(counts,[3139,3140,3140])

    def test_epoch_parity_with_original_trainer(self):
        # An uneven last batch checks the original equal-batch weighting too.
        torch.set_num_threads(2)
        cfg=yaml.safe_load(next((HERE.parents[1]/'configs').glob('run_0093*')).read_text())
        cfg['model']['channels']=[2,4,8,16]
        torch.manual_seed(81)
        images=torch.rand(5,2,48,48)
        model=build_vae(cfg,'cpu'); initial=copy.deepcopy(model.state_dict())
        def loaders():
            return (DataLoader(images,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(3)),
                    DataLoader(images,batch_size=2,shuffle=False))
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
        tr,va=loaders(); torch.manual_seed(91)
        result=train('cpu',tr,va,model,opt,vae_loss,epochs=2,beta=32.)
        reference=copy.deepcopy(model.state_dict())
        model=build_vae(cfg,'cpu');model.load_state_dict(initial)
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
        tr,va=loaders();torch.manual_seed(91)
        history=[]; best=float('inf');state=None
        for _ in range(2):
            a=run_epoch(model,opt,tr,32.,'cpu',True)
            b=run_epoch(model,opt,va,32.,'cpu',False)
            history.append((a,b))
            if b[0]<best-1e-4: best=b[0];state=copy.deepcopy(model.state_dict())
        for i in range(2):
            np.testing.assert_array_equal(history[i][0],[result[1][i],result[2][i],result[3][i]])
            np.testing.assert_array_equal(history[i][1],[result[4][i],result[5][i],result[6][i]])
        for k in reference: torch.testing.assert_close(state[k],reference[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
