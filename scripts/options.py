import argparse

def parse():
    print('Parsing arguments')
    parser = argparse.ArgumentParser(description='Temporal Model')
    
    parser.add_argument('lr', default=0.001, type=float)
    parser.add_argument('data_dim', default=7168, type=int)
    parser.add_argument('data_dim_skl', default=150, type=int)
    parser.add_argument('num_classes', default=60, type=int)
    parser.add_argument('batch_size', default=32, type=int)
    parser.add_argument('n_neuron', default=512, type=int)
    parser.add_argument('n_dropout', default=0.6, type=float)
    parser.add_argument('timesteps_seg1', default=2, type=int)
    parser.add_argument('timesteps_seg2', default=3, type=int)
    parser.add_argument('timesteps_seg3', default=4, type=int)
    parser.add_argument('timesteps_skl', default=5, type=int)
    parser.add_argument('n_neuron_skl', default=150, type=int)
    parser.add_argument('name', default='test')
    parser.add_argument('epochs', default=100, type=int)
    parser.add_argument('att', default='att_features_withoutSA/')
    parser.add_argument('train_file', default='../data/train_CS.txt')
    parser.add_argument('val_file', default='../data/validation_CS.txt')
    parser.add_argument('test_file', default='../data/test_CS.txt')

    args = parser.parse_args()
    return args

