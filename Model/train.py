import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
import time
from tensorboardX import SummaryWriter
from Model.model import *
import argparse


def classify(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, dataname, iter, fold_count, args):

    unsup_model = Net(args, 64, args.layers).to(args.device)
    loss = 0
    for unsup_epoch in range(args.unsup_epoch):
        optimizer = th.optim.Adam(unsup_model.parameters(), lr=lr, weight_decay=weight_decay)
        unsup_model.train()
        traindata_list, _ = loadBiData(dataname, treeDic, x_train+x_test, x_test, 0.2, 0.2)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        batch_idx = 0
        loss_all = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            optimizer.zero_grad()
            Batch_data = Batch_data.to(args.device)
            loss = unsup_model(Batch_data)
            loss_all += loss.item() * (max(Batch_data.batch) + 1)

            loss.backward()
            optimizer.step()
            batch_idx = batch_idx + 1
        loss = loss_all / len(train_loader)
        print('unsup_epoch:', (unsup_epoch) ,'   loss:', loss)
    name = "best_pre_"+ dataname +"_4unsup" + ".pkl"
    th.save(unsup_model.state_dict(), name)
    print('Finished the unsuperivised training.', '  Loss:', loss)
    print("Start classify!!!")
    # unsup_model.eval()

    model = Classfier(args, 64*args.layers, 64, args.num_class).to(args.device)
    if args.fine_tune_lr:
        opt = th.optim.Adam(
            [{'params':unsup_model.parameters(), 'lr':args.fine_tune_lr},
            {'params':model.parameters(), 'lr':lr}], 
            weight_decay=weight_decay)
    else:
        opt = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    x_train_full = x_train
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        if args.per<1.:
            lim = max(1, int(len(x_train_full)*args.per))
            x_train = x_train_full[0:lim]
            print("classifier train len: ", len(x_train))
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        model.train()
        unsup_model.train()
        for Batch_data in tqdm_train_loader:
            Batch_data.to(args.device)
            _, Batch_embed, _, xc_embed = unsup_model.encoder(Batch_data.x, Batch_data.edge_index, Batch_data.batch)
            out_labels = model(Batch_embed, Batch_data)
            finalloss=F.nll_loss(out_labels, Batch_data.y)
            loss=finalloss
            opt.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            opt.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f} | Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(), 
                                                                                                 train_acc))
            batch_idx = batch_idx + 1
            
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        unsup_model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(args.device)
            Batch_embed = unsup_model.encoder.get_embeddings(Batch_data)
            val_out = model(Batch_embed, xc_embed, Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f} | fold {:d}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs), fold_count))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'RDEA_'+str(fold_count)+'_', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if epoch>=199:
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4




def init_seed(seed=2023):
    th.manual_seed(seed)  # sets the seed for generating random numbers.
    th.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    th.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print("Init_seed....", seed)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0005, metavar='lr', help='lr')
    parser.add_argument('--fine_tune_lr', type=float, default=0., metavar='fine_tune_lr', help='fine_tune_lr')
    parser.add_argument('--weight_decay', type=float, default='1e-4', metavar='weight_decay', help='weight_decay')
    parser.add_argument('--patience', type=int, default=10, metavar='patience', help='patience')
    parser.add_argument('--batchsize', type=int, default=128, metavar='batchsize', help='batchsize')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='n_epochs', help='n_epochs')
    parser.add_argument('--TDdroprate', type=float, default=0.4, metavar='TDdroprate', help='TDdroprate')
    parser.add_argument('--BUdroprate', type=float, default=0.4, metavar='BUdroprate', help='BUdroprate') #没用
    parser.add_argument('--datasetname', type=str, default="Twitter16", metavar='datasetname', help='datasetname') #"Twitter15"、"Twitter16"
    parser.add_argument('--num_class', type=int, default=4, metavar='num_class', help='num_class')
    parser.add_argument('--iterations', type=int, default=1, metavar='iterations', help='iterations') 
    parser.add_argument('--unsup_epoch', type=int, default=25, metavar='unsup_epoch', help='unsup_epoch')
    parser.add_argument('--layers', type=int, default=3, metavar='layers', help='layers') 
    parser.add_argument('--seed', type=int, default=2023, metavar='seed', help='seed')
    parser.add_argument('--per', type=float, default=1., metavar='per', help='per')
    
    
#     parser.add_argument('--with_random', type=int, default=0, metavar='with_random', help='if add random')
#     parser.add_argument('--add_or_cat', type=str, default='add', metavar='add_or_cat', help='add_or_cat')
    
    args, unknown = parser.parse_known_args()
    
    args.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    #args.device = device
    print(args)
    init_seed(seed=args.seed)
    
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    for iter in range(args.iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test,  fold1_x_train,  \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test,fold4_x_train = load5foldData(args.datasetname)

        treeDic=loadTree(args.datasetname)
        t1 = time.time()
        train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = classify(treeDic,
                                                                                                   fold0_x_test,
                                                                                                   fold0_x_train,
                                                                                                   args.TDdroprate, args.BUdroprate,
                                                                                                   args.lr, args.weight_decay,
                                                                                                   args.patience,
                                                                                                   args.n_epochs,
                                                                                                   args.batchsize,
                                                                                                   args.datasetname,
                                                                                                   iter, 0, args)
        train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = classify(treeDic,
                                                                                                   fold1_x_test,
                                                                                                   fold1_x_train,
                                                                                                   args.TDdroprate, args.BUdroprate,
                                                                                                   args.lr, args.weight_decay,
                                                                                                   args.patience,
                                                                                                   args.n_epochs,
                                                                                                   args.batchsize,
                                                                                                   args.datasetname,
                                                                                                   iter, 1, args)
        train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = classify(treeDic,
                                                                                                   fold2_x_test,
                                                                                                   fold2_x_train,
                                                                                                   args.TDdroprate, args.BUdroprate,
                                                                                                   args.lr, args.weight_decay,
                                                                                                   args.patience,
                                                                                                   args.n_epochs,
                                                                                                   args.batchsize,
                                                                                                   args.datasetname,
                                                                                                   iter, 2,  args)
        train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = classify(treeDic,
                                                                                                   fold3_x_test,
                                                                                                   fold3_x_train,
                                                                                                   args.TDdroprate, args.BUdroprate,
                                                                                                   args.lr, args.weight_decay,
                                                                                                   args.patience,
                                                                                                   args.n_epochs,
                                                                                                   args.batchsize,
                                                                                                   args.datasetname,
                                                                                                   iter, 3, args)
        train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = classify(treeDic,
                                                                                                   fold4_x_test,
                                                                                                   fold4_x_train,
                                                                                                   args.TDdroprate, args.BUdroprate,
                                                                                                   args.lr, args.weight_decay,
                                                                                                   args.patience,
                                                                                                   args.n_epochs,
                                                                                                   args.batchsize,
                                                                                                   args.datasetname,
                                                                                                   iter, 4, args)
        test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
        NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
        print("check  iter: {:04d} | aaaaaccs: {:.4f}".format(iter, test_accs[iter]))
        t2 = time.time()
        print("total time:")
        print(t2 - t1)
    print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs)/args.iterations, sum(NR_F1)/args.iterations, sum(FR_F1)/args.iterations, sum(TR_F1)/args.iterations, sum(UR_F1)/args.iterations))
