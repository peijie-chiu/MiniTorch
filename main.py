import os
import argparse
import numpy as np
from os.path import normpath as fn # Fix Linux/Windows path issue
import sys
sys.path.append("nn") # add the nn module into system path

import nn.layer as layer
import nn.graph as graph
import nn.solver as solver
import nn.container as container
import nn.loss as loss

from nn.model import ConvNet, MLP

def get_batches(data_len, BSZ, shuffle=False):
    batches = range(0,data_len-BSZ+1,BSZ)
    if shuffle:
        np.random.shuffle(batches)
    return batches

def train(args, model, train_dl, val_dl, loss_func, optimizer):
    accuracy = loss.Accuracy()

    # Build Computation Graph
    inp = layer.Tensor()
    lab = layer.Tensor()
    y = model(inp)
    criterion = loss_func(y,lab)
    acc = accuracy(y,lab)

    niter, avg_loss, avg_acc=0, 0., 0.
    BSZ = args.batch_size
    train_batches = get_batches(len(train_dl[1]), BSZ, shuffle=False)
    
    for ep in range(args.epochs+1):
        graph.eval()
        batches = get_batches(len(val_dl[1]), BSZ, shuffle=False)
        vacc, vloss, viter= 0., 0., 0
        for b in batches:
            inp.set(val_dl[0][b:b+BSZ,...])
            lab.set(val_dl[1][b:b+BSZ])
            
            graph.Forward()
            viter += 1
            vacc += acc.top
            vloss += criterion.top
        vloss, vacc = vloss / viter, vacc / viter * 100
        print("%06d: #### %d Epochs: Val Loss = %.3e, Accuracy = %.2f%%" % (niter,ep,vloss,vacc))

        if ep == args.epochs:
            break

        graph.train()
        # Shuffle Training Set
        idx = np.random.permutation(len(train_dl[1]))
        
        for b in train_batches:
            # Load a batch
            inp.set(train_dl[0][idx[b:b+BSZ],...])
            lab.set(train_dl[1][idx[b:b+BSZ]])

            graph.Forward()
            avg_loss += criterion.top 
            avg_acc += acc.top
            niter += 1
            if niter % args.log_interval == 0:
                avg_loss = avg_loss / args.log_interval
                avg_acc = avg_acc / args.log_interval * 100
                print("%06d: Training Loss = %.3e, Accuracy = %.2f%%" % (niter,avg_loss,avg_acc))
                avg_loss, avg_acc = 0., 0.

            graph.Backward(criterion)
            if args.solver == 'Adam':
                optimizer.step(niter)
            else:
                optimizer.step()
        

def test(args, model, test_lb):
    pass

def options():
    """
    Manual predifined Options 
    Note: add as needed 
    """
    parser = argparse.ArgumentParser(description="the script to run the experiments for this project")
    parser.add_argument('--dataroot', type=str, default='', help='dataroot path')
    parser.add_argument('--model', type=str, default='ConvNet', choices=['MLP','ConvNet','RNN'], help='choice of models')
    parser.add_argument('--n_class', type=int, default=10, help='number of classes')
    parser.add_argument('--nc', type=int, default=1, help='number of Input Channel')
    parser.add_argument('--nf', type=int, nargs='+', default=16, help='number of Channels for network')
    parser.add_argument('--image-size', type=int, default=28, help='number of Input Channel')
    parser.add_argument('--outf', type=str, default='', help='the output folder to store any result')
    parser.add_argument('--net', type=str, default='', help='choice of models')
    parser.add_argument('--manual-seed', type=int, default=0, help='manual seed for the randomness')

    subparsers = parser.add_subparsers(title="subcommands",
                                            dest="subcommand", help='sub-command help')

    train_parser = subparsers.add_parser('train', help='train phase help')
    train_parser.add_argument('--dry-run', action='store_true', help='run one iteration to test the correctness of the code')
    train_parser.add_argument('--epochs', type=int, help='number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, help='batch size to train')
    train_parser.add_argument('--lr', type=float, help='learning rate used for training')
    train_parser.add_argument('--solver', type=str, choices=['SGD', 'Momentum', 'Nesterov', 'Adam', 'RMSprop', 'Adagrad'], help='solver used to optimize the network')
    train_parser.add_argument('--log-interval', type=int, default=100, help='number of intervals to output log infomation')
    

    test_parser = subparsers.add_parser('test', help='test phase help')
    test_parser.add_argument('--batch-size', type=int, help='batch size to test')

    args = parser.parse_args()
    return args

def main():
    args = options()
    print(args)

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    np.random.seed(args.manual_seed)

    if args.model == 'MLP':
        model = MLP(args.nc*args.image_size**2, args.nf, args.n_class)
    elif args.model == 'ConvNet':
        model = ConvNet(args.nc, args.nf, args.n_class, args.image_size)
    elif args.model == 'RNN':
        pass
    
    print(model)

    # loss function: softmax + crossentropy
    smaxloss = loss.SmaxCELoss()

    if args.subcommand == 'train':
        # Load data
        data = np.load(fn(args.dataroot))
        train_im = np.float32(data['im_train'])/255.-0.5
        train_lb = data['lbl_train']

        val_im = np.float32(data['im_val'])/255.-0.5
        val_lb = data['lbl_val']

        if args.model == 'ConvNet':
            train_im = np.reshape(train_im,[-1,args.image_size,args.image_size,args.nc])
            val_im = np.reshape(val_im,[-1,args.image_size,args.image_size,args.nc])

        print(f"{'='*30}Training Start{'='*30}")
        if args.dry_run:
            args.epochs = 1

        if args.solver == 'SGD':
            optimizer = solver.SGD(graph.params, args.lr)
        elif args.solver == 'Momentum':
            optimizer = solver.Momentum(graph.params, args.lr, mom=0.9)
        elif args.solver == 'Nesterov':
            optimizer = solver.Nesterov(graph.params, args.lr, mom=0.9)
        elif args.solver == 'Adagrad':
            optimizer = solver.Adagrad(graph.params, args.lr)
        elif args.solver == 'Adam':
            optimizer = solver.Adam(graph.params, args.lr)
        elif args.solver == 'RMSprop':
            optimizer = solver.RMSprop(graph.params, args.lr)
    
        train(args, model, (train_im, train_lb), (val_im, val_lb), smaxloss, optimizer)
        print(f"{'='*30}Training End{'='*30}")
        
    elif args.subcommand == 'test':
        print(f"{'='*30}Testing Start{'='*30}")


if __name__ == '__main__':
    main()