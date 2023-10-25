import torch
from onelayerencoder import *

def model_initialization(args):
    """
    Neural netowrk initialization.
    :param args: parser arguments
    :return: neural network as torch.nn.Module
    """
    torch.manual_seed(args.seed_init)
    if args.attention_type == 'standard':
        attention_part = MultiheadAttention(args.input_dim, args.input_dim, args.num_heads)
    elif args.attention_type == 'modified':
        attention_part = ModifiedMultiheadAttention(args.input_dim, args.input_dim, args.num_heads)
    else:
        raise ValueError('attention_type should be standard or modified')
    
    if args.reducer_type == 'MLP':
        reducer_part = MLP(args.input_dim, args.dim_feedforward, args.dropout)
    elif args.reducer_type == 'L2NORM':
        reducer_part = L2Norm()
    elif args.reducer_type == 'NormLastColumn':
        reducer_part = NormLastColumn()
    else:
        raise ValueError('reducer_type should be MLP or L2NORM or L1NORM')
    
    if args.layer_norm == True and args.residual == True:
        model = OnelayerEncoder(attention_part, reducer_part, args)
    elif args.layer_norm == False and args.residual == False:
        model = OnelayerEncoderNoResidualNoLayerNorm(attention_part, reducer_part, args)
    elif args.layer_norm == True and args.residual == False:
        model = OnelayerEncoderNoResidual(attention_part, reducer_part, args)
    elif args.layer_norm == False and args.residual == True:
        model = OnelayerEncoderNoLayerNorm(attention_part, reducer_part, args)
    
    return model
    