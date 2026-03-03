import numpy as np
import os
import time
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from loss import PRetLoss
from data_loader import data_provider
from model.poptimizer import POptimizer

class Exp:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.model = self._build_model().to(self.device)
        
    def _build_model(self):
        model = POptimizer(self.configs)
        return model

    def _get_data(self):
        dataset, dataloader = data_provider(self.configs)
        return dataset, dataloader
    
    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.configs.learning_rate, weight_decay=self.configs.l2)
        return model_optim

    def _select_criterion(self):
        criterion = PRetLoss(self.configs)  
        return criterion
    
    def _eval(self, prets):
        nav = np.cumprod(1 + prets, axis=0)
        sharpe = np.mean(prets) / (np.std(prets) + 1e-8) * np.sqrt(252 / len(prets))
        maxdrawdown = np.min((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav))
        return sharpe, maxdrawdown

    def backtest(self, setting='default'):
        import shutil  
        all_epoch_times = []

        rw_dataset, rw_loader = self._get_data()
        max_steps = getattr(self.configs, "max_steps", None)
        # max_steps=0 or None means run all steps
        if max_steps is None or max_steps <= 0:
            max_steps = None
        total_steps = len(rw_dataset) if max_steps is None else min(len(rw_dataset), max_steps)
        
        weights = []
        prets = []
        eval_metrics = []
        indices = []
        all_pred_returns = []  

        for i, data_tuple in enumerate(rw_dataset):
            if max_steps is not None and i >= max_steps:
                break
            x0_train, x_train, y_train, x0_test, x_test, y_test, test_start, test_end = data_tuple
                
            x0_train = x0_train.to(self.device)
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            x0_test = x0_test.to(self.device)
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)
            
            train_dataset = TensorDataset(x0_train, x_train, y_train)
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=self.configs.batch_size, 
                shuffle=False
            )
            test_dataset = TensorDataset(x0_test, x_test, y_test)

            # model = self.train(train_dataloader, val_dataset, path)
            model, epoch_times = self.train(train_dataloader)
            all_epoch_times.extend(epoch_times)



            result = self.validate(test_dataset, model)
            if len(result) == 3:
                weights_step, prets_step, eval_metric = result
                pred_returns_step = None
            else:
                weights_step, prets_step, eval_metric, pred_returns_step = result
                if pred_returns_step is not None:
                    all_pred_returns.append(pred_returns_step)

            weights.append(weights_step)
            prets.append(prets_step)
            indices.extend([test_start, test_end])
            eval_metrics.append(eval_metric)
            print(f'Step {i+1}/{total_steps}: Eval Metric: {eval_metric}', flush=True)
        
        weights = np.concatenate(weights, axis=0)
        prets = np.concatenate(prets, axis=0)
        indices = np.array(indices)
        eval_metrics = np.array(eval_metrics)

        test_folder_path = f'./test_results/{setting}/'
        tmp_path = test_folder_path.rstrip('/\\') + '.tmp'
        if len(all_epoch_times) > 0:
            global_mean = np.mean(all_epoch_times)
            print(f"\n==== Global average epoch time over all steps: {global_mean:.4f} seconds ====\n", flush=True)

        try:
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            os.makedirs(tmp_path, exist_ok=True)

            if len(all_pred_returns) > 0:
                segs = []
                for arr in all_pred_returns:
                    a = np.asarray(arr)
                    if a.ndim == 2:           # [H,N] -> [1,H,N]
                        a = a[np.newaxis, ...]
                    elif a.ndim != 3:         
                        raise ValueError(f"Unexpected pred_returns shape: {a.shape}")
                    segs.append(a)

                H_ref, N_ref = segs[0].shape[1], segs[0].shape[2]
                keep = [a for a in segs if (a.shape[1] == H_ref and a.shape[2] == N_ref)]

                if len(keep) == 0:
                    raise ValueError("No return segments match reference (H,N); abort saving.")

                returns = np.concatenate(keep, axis=0)  # [T_total, H_ref, N_ref]
                np.save(os.path.join(tmp_path, 'returns.npy'), returns)

            np.save(os.path.join(tmp_path, 'weights.npy'), weights)
            np.save(os.path.join(tmp_path, 'prets.npy'), prets)
            np.save(os.path.join(tmp_path, 'indices.npy'), indices)
            np.save(os.path.join(tmp_path, 'eval_metrics.npy'), eval_metrics)

            if os.path.exists(test_folder_path):
                shutil.rmtree(test_folder_path)
            os.rename(tmp_path, test_folder_path)

        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    shutil.rmtree(tmp_path)
            except Exception:
                pass
            print(f"[error] saving aborted; nothing saved. reason: {e}")

        return weights, prets


    def validate(self, dataset, model):
        weights = []
        prets = []
        model.eval()
        with torch.no_grad():
            w = None
            for x0, x, y in dataset:
                H, N = y.shape
                x0 = x0.to(self.device)
                x = x.to(self.device)
                y0 = y[0].to(self.device)
                y = y.to(self.device)

                if w is None:
                    w = torch.full((N,), 1.0 / N, device=self.device)
                w = self.model(x0.unsqueeze(0), x.unsqueeze(0), opt=True, z0=w.clone().detach())
                w = w[:, 0, :] if w.dim() == 3 else w

                pret = (w.squeeze(0) * y0).sum(dim=-1)

                weights.append(w.detach().cpu().numpy())  # save [N]
                prets.append(pret.detach().cpu().item())

        model.train()
        weights = np.stack(weights, axis=0)  # [T,N]
        prets = np.array(prets)              # [T]
        eval_metric = self._eval(prets)
        return weights, prets, eval_metric

    def train(self, dataloader):
        train_steps = len(dataloader)
        
        self.model = self._build_model().to(self.device)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        epoch_times = []  

        for epoch in range(self.configs.num_epochs):
            train_loss = []
            self.model.train()
            self.model.covupdater.reset()
            epoch_start = time.time()
            
            # Timing accumulators for this epoch
            epoch_forward_time = 0.0
            epoch_backward_time = 0.0
            
            w = None
            for i, (batch_x0, batch_x, batch_y) in enumerate(dataloader):
                B, H, N = batch_y.shape
                batch_x0 = batch_x0.to(self.device)
                batch_x = batch_x.to(self.device)
                batch_y0 = batch_y[:, 0, :].to(self.device)
                batch_y = batch_y.to(self.device)
                
                model_optim.zero_grad()
                if w is None:
                    w = torch.full((1, N), 1.0 / N, device=self.device)
                w_path = torch.zeros((B, N), device=self.device)
                
                # Forward pass timing
                forward_start = time.time()
                for b in range(B):
                    x0 = batch_x0[b].unsqueeze(0)
                    x = batch_x[b].unsqueeze(0)
                    w = self.model(x0, x, opt=True, z0=w.clone().detach())
                    w = w[:, 0, :] if w.dim() == 3 else w
                    w_path[b] = w.squeeze(0)
                    w = w.squeeze(0).detach()
                forward_end = time.time()
                epoch_forward_time += (forward_end - forward_start)

                loss = criterion(w_path, batch_y0, self.model.covupdater.cov)
                
                # Backward pass timing
                backward_start = time.time()
                loss.backward()
                backward_end = time.time()
                epoch_backward_time += (backward_end - backward_start)
                
                model_optim.step()
                
                train_loss.append(loss.item())
                
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{self.configs.num_epochs}], Step [{i + 1}/{train_steps}], Loss: {loss.item():.6f}', flush=True)

            epoch_time = epoch_forward_time + epoch_backward_time
            epoch_times.append(epoch_time)

            train_loss_avg = np.average(train_loss)
            print(f'Epoch [{epoch + 1}/{self.configs.num_epochs}] | '
                  f'forward: {epoch_forward_time:.4f}s, backward: {epoch_backward_time:.4f}s, total: {epoch_time:.4f}s | '
                  f'loss: {train_loss_avg:.6f}', flush=True)

        mean_time = np.mean(epoch_times)
        print(f"\nAverage epoch time over {len(epoch_times)} epochs: {mean_time:.4f} seconds\n", flush=True)

        return self.model , epoch_times
    
    

