import argparse
import os
import torch
import random
import numpy as np
import pandas as pd

# Default to using Exp only; if two-stage is needed, please supplement Exp2S implementation
from exp import Exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model training and evaluation')
    
    # basic config
    parser.add_argument('--is_training', type=int, default=1, required=True, help='Status')
    parser.add_argument('--model', type=str, default='Linear', choices=['CNNLSTM', 'Linear'], help='Model type')
    parser.add_argument('--optimizer', type=str, default='dXPP', required=True, 
                        choices=['dQP', 'dXPP', 'OptNet', 'Cvxpy', 'SCQPTH'], help='QP optimizer')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--device', type=str, default= 'cpu', help='Device to use')
    parser.add_argument('--itr', type=int, default=1, help='Experiment times')
    
    # data loader
    parser.add_argument('--data', type=str, default='ETF', help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--train_len', type=int, default=250, help='Training length')
    parser.add_argument('--test_len', type=int, default=20, help='Testing length')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--window', type=int, default=30, help='Input window size')
    parser.add_argument('--horizon', type=int, default=20, help='Input horizon')
    parser.add_argument('--cov_window', type=int, default=20, help='Input cov_window')
    parser.add_argument('--windows', type=int, nargs='+', default=[1], help='Windows for cumulative returns')
    parser.add_argument('--stride', type=int, default=None, help='Stride for the sliding window, default = horizon')
    parser.add_argument('--decay', type=float, default=0.95, help='Decay factor for covariance matrix updates')
    parser.add_argument('--feature', type=str, default='raw', help='Feature engineering method')
    
    # model
    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension for the model')
    parser.add_argument('--num_channels', type=int, default=10, help='Number of assets')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension for the model')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model')
    parser.add_argument('--individual', type=int, default=0, help='Individual head; True 1 False 0')
    parser.add_argument('--revin', type=int, default=1, help='Use RevIN normalization; True 1 False 0')
    
    # model - FF
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension for Feed Forward')
    
    # model - CNNLSTM
    parser.add_argument('--conv_channels', type=int, default=32, help='Number of channels in CNN')
    parser.add_argument('--conv_kernel_size', type=int, default=5, help='Kernel size for CNN')
    parser.add_argument('--conv_stride', type=int, default=1, help='Stride for CNN')
    parser.add_argument('--conv_dropout', type=float, default=0.1, help='Dropout rate for CNN')
    parser.add_argument('--lstm_hidden_dim', type=int, default=32, help='Hidden dimension for LSTM')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of layers in LSTM')
    parser.add_argument('--lstm_dropout', type=float, default=0.1, help='Dropout rate for LSTM layers (if multiple)')

    # model - PatchTST
    parser.add_argument('--enc_in', type=int, default=None, help='Input dimension (PatchTST)')
    parser.add_argument('--seq_len', type=int, default=None, help='Input sequence length (PatchTST)')
    parser.add_argument('--pred_len', type=int, default=None, help='Prediction length (PatchTST)')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of Transformer encoder layers (PatchTST)')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads (PatchTST)')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension (PatchTST)')
    parser.add_argument('--d_ff', type=int, default=128, help='Feedforward dimension (PatchTST)')
    parser.add_argument('--fc_dropout', type=float, default=0.1, help='Dropout before head (PatchTST)')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='Dropout in head (PatchTST)')
    parser.add_argument('--patch_len', type=int, default=20, help='Patch length (<= seq_len) for PatchTST')
    parser.add_argument('--padding_patch', type=str, default='end', help='Padding strategy for patches: end or None (PatchTST)')
    parser.add_argument('--affine', type=int, default=1, help='RevIN affine flag (PatchTST)')
    parser.add_argument('--subtract_last', type=int, default=0, help='RevIN subtract_last flag (PatchTST)')
    parser.add_argument('--decomposition', type=int, default=0, help='Use series decomposition (PatchTST)')
    parser.add_argument('--kernel_size', type=int, default=25, help='Kernel size for series decomposition (PatchTST)')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout (PatchTST)')
    parser.add_argument('--res_attention', type=int, default=1, help='Use residual attention (PatchTST)')
    parser.add_argument('--pre_norm', type=int, default=0, help='Use pre-norm in encoder layers (PatchTST)')
    parser.add_argument('--pe', type=str, default='zeros', help='Positional encoding type (PatchTST)')
    parser.add_argument('--learn_pe', type=int, default=1, help='Learn positional encoding (PatchTST)')
    parser.add_argument('--pretrain_head', type=int, default=0, help='Use pretrain head (PatchTST)')
    parser.add_argument('--head_type', type=str, default='flatten', help='Head type (PatchTST)')

    # algorithm
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--r', type=float, default=0.5)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--switch_tol', type=float, default=1e-3)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--lam', type=float, default=1e-4, help='Regularization parameter for smoothing')
    parser.add_argument('--kappa', type=float, default=0.1, help='Threshold for smoothing')
    
    
    # training
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64) # Must be a divisor of window_size
    parser.add_argument('--learning_rate', type=float, default=5e-2)
    parser.add_argument('--learning_rate_2s', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--risk_param', type=float, default=0.5, help='Risk parameter for the loss function')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 regularization coefficient')
    parser.add_argument('--max_steps', type=int, default=1, help='Max rolling-window steps to run (None for all)')
    
    # test
    parser.add_argument('--reset_cov', type=int, default=0, help='Reset covariance matrix in testing; True 1 False 0')

    args = parser.parse_args()

    # Read data column count, automatically set number of assets to avoid RevIN/CNN channel mismatch
    try:
        df_all = pd.read_csv(args.data_path, header=None, usecols=[0])
        n_assets = df_all[0].nunique()
        if n_assets and n_assets > 0:
            args.num_channels = int(n_assets)
            print(f'[info] inferred num_channels = {args.num_channels} from data RIC count')
    except Exception as e:
        print(f'[warn] failed to infer num_channels from data: {e}')
    
    # random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print('Args:')
    print(args)

    def _fmt_num(x) :
        return f"{x:.6g}"

    def _combo_tag(feature, lam, delta, lr, l2,optimizer) :  
        return (f"ETF_MD_{feature}"f"optimizer_{optimizer}" f"_lam{_fmt_num(lam)}" f"_del{_fmt_num(delta)}" f"_lr{_fmt_num(lr)}" f"_l2{_fmt_num(l2)}")
    for ii in range(args.itr):
        if args.stride is None:
            args.stride = args.horizon

        setting = _combo_tag(args.feature, args.lam, args.delta, args.learning_rate_2s, args.l2, args.optimizer)
        exp = Exp(args)

        print('>>>>>>>start integrated>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.backtest(setting)

        print('<<<<<<<finished<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        torch.cuda.empty_cache()