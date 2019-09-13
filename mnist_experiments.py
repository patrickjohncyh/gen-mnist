import time
from tqdm import tqdm as Tqdm
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from gen_datasets import FTDataset
from poisson_datasets import PoissonSquareRoomInpDataset, \
        PoissonSquareRoomOutDataset
from poisson_square_experiments_utils import *
from neural_processes import NeuralProcesses
from GEN import GEN
from gen_softnn import GENSoftNN
from gen_planargrid import GENPlanarGrid
from utils import Net


from torchvision import datasets,transforms
import matplotlib.pyplot as plt

torch.manual_seed(0)
cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
model_type = 'GENPlanarGrid' #['GENSoftNN', 'GENPlanarGrid', 'NP'][0]
bs = 128
k = 128
node_train = 16
sqrt_num_nodes_list = [3]#[4,3,4,5,6,7]
copies_per_graph = 1
opt_nodes = True
slow_opt_nodes = False #Train node_pos only in part of each "house" data;slower
do_tensorboard = False
# Changed the random initialization because GeneralizedHalton
# doesn't install well on a Docker. We use another simple random initialization.

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('mnist_trainset', download=True, train=True, transform=transform)
test_dataset = datasets.MNIST('mnist_testset', download=True, train=False,transform=transform)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=8,
        shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset,  batch_size=bs, num_workers=8,
        shuffle=True, drop_last=False)

## Data Visualisation
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# print(images.shape)
# print(labels.shape)
# print(labels)
# plt.imsave('mnist_test.png',images[0].numpy().squeeze(), cmap='gray_r');

encoders = nn.ModuleList([Net(dims=[3,2*k,2*k,k])])
decoders = nn.ModuleList([Net(dims=[k,2*k,2*k,10])])
loss_fn = nn.CrossEntropyLoss()


assert min(sqrt_num_nodes_list) >= 1
model = GENPlanarGrid(encoders=encoders, decoders=decoders)
#model = torch.load('model-3x3-opt-nodes-e100m.pt')
mesh_list, mesh_params = create_mesh_list(
        num_datasets= 1,
        sqrt_num_nodes_list=sqrt_num_nodes_list,
        initialization='random' if opt_nodes else 'uniform',
        copies_per_graph=copies_per_graph, device=device)
max_mesh_list_elts = max([len(aux) for aux in mesh_list])
if cuda: model.cuda()
opt = torch.optim.Adam(params=model.parameters(), lr=3e-4)
if len(mesh_params):
    mesh_opt = torch.optim.Adam(params=mesh_params, lr=3e-4)
else: mesh_opt = None

if do_tensorboard: writer = SummaryWriter()
else: writer = None



coords = [(x/27.0,y/27.0) for x in range(0,28,1) for y in range(0,28,1)]
coords = torch.Tensor(coords)
if cuda:
    coords = coords.cuda()

loss_curves = { num**2:[] for num in sqrt_num_nodes_list}
accy_curves = { num**2:[] for num in sqrt_num_nodes_list}

print(loss_curves[9])


for epoch in Tqdm(range(1), position=0):
    train_loss = 0. ;  test_loss = 0.
    train_graphs = 0 ; test_graphs = 0
    train_loss_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}
    test_loss_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}
    pos_change_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}

    train_accy_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}
    test_accy_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}
    

    # Train Set
    for g_idx in (range(max_mesh_list_elts)):
        idx = 0
        for cnt, (Inp,Out) in enumerate(Tqdm(train_loader)):
            if cuda:
                Inp = Inp.cuda()
                Out = Out.cuda()
            if len(mesh_list[idx]) <= g_idx: continue
            G = mesh_list[idx][g_idx]

            Inp = (coords,Inp)
            Q = None
            targets = Out
            train_graphs += 1
        
            if slow_opt_nodes:
                FInp = [[inp[0][:node_train], inp[1][:node_train]]
                        for inp in Inp]
                FQ = [q[:node_train] for q in Q]
                EInp = [[inp[0][node_train:], inp[1][node_train:]]
                        for inp in Inp]
                EQ = [q[node_train:] for q in Q]
                Fpreds = model(FInp, FQ, G=G)
                Epreds = model(EInp, EQ, G=G)
                finetune_losses = [loss_fn(pred,
                    target[:node_train]).unsqueeze(0)
                    for (pred, target) in zip(Fpreds, targets)]
                finetune_loss = torch.sum(torch.cat(finetune_losses))
                exec_losses = [loss_fn(pred,
                    target[node_train:]).unsqueeze(0)
                    for (pred, target) in zip(Epreds, targets)]
                exec_loss = torch.sum(torch.cat(exec_losses))
                mesh_opt.zero_grad()
                finetune_loss.backward()
                mesh_opt.step()
                # project back to square
                graph_update_meshes_after_opt(mesh_list[idx][g_idx],
                        epoch=epoch, writer=writer)
                loss = exec_loss
            else:
                preds = model(Inp, Q, G=G)
                loss  = loss_fn(preds,targets)
                accy  = ((torch.max(preds,1)[1]-targets)==0).sum().item()/preds.shape[0]
                train_accy_summ[G.num_nodes][0] += accy
                train_accy_summ[G.num_nodes][1] += 1
        
            loss.backward()
            train_loss += loss.item()
            train_loss_summ[G.num_nodes][0] += loss.item()
            pos_change_summ[G.num_nodes][0] += (
                    torch.max(torch.abs(G.pos - G.ini_pos)).item())
            train_loss_summ[G.num_nodes][1] += 1
            pos_change_summ[G.num_nodes][1] += 1
            
            # print('Elapsed time : ',time.time()-start)
            opt.step()
            opt.zero_grad()
            num = sqrt_num_nodes_list[g_idx]
            # if (cnt % 32 == 32-1) or (cnt == len(train_loader)-1):
            #     print('')
            #     print('train/loss-'+str(num**2),
            #         train_loss_summ[num**2][0]/train_loss_summ[num**2][1],
            #         (cnt+1))
            #     print('train/accy-'+str(num**2),
            #         train_accy_summ[num**2][0]/train_accy_summ[num**2][1],
            #         (cnt+1))

    if do_tensorboard:
        for num in sqrt_num_nodes_list:
            writer.add_scalar('train/loss-'+str(num**2),
                    train_loss_summ[num**2][0]/train_loss_summ[num**2][1],
                    epoch)
            print('Epoch '+str(epoch)+'--- train/loss-'+str(num**2),
                    train_loss_summ[num**2][0]/train_loss_summ[num**2][1],
                    epoch)
    else:
        pass
        # for num in sqrt_num_nodes_list:
        #     continue
            # print('Epoch '+str(epoch)+'--- train/loss-'+str(num**2),
            #         train_loss_summ[num**2][0]/train_loss_summ[num**2][1],
            #         epoch)


    # Test Set
    for g_idx in Tqdm(range(max_mesh_list_elts), position=1):
        idx = 0
        for cnt, (Inp,Out) in enumerate(test_loader):
            
            if cuda:
                Inp = Inp.cuda()
                Out = Out.cuda()
            if len(mesh_list[idx]) <= g_idx: continue
            G = mesh_list[idx][g_idx]

            Inp = (coords,Inp)
            Q = None
            targets = Out
            train_graphs += 1
            preds = model(Inp, Q, G=G)
            if opt_nodes:
                if(preds.shape[0]>node_train):
                    finetune_loss  = loss_fn(preds[:node_train],targets[:node_train])
                    exec_loss  = loss_fn(preds[node_train:],targets[node_train:])
                    exec_accy  = ((torch.max(preds[node_train:],1)[1]-targets[node_train:])==0).sum().item()/preds[node_train:].shape[0]
                    finetune_loss.backward()
                    loss = exec_loss
                    accy = exec_accy
            else:
                loss  = loss_fn(preds,targets)
                accy  = ((torch.max(preds,1)[1]-targets)==0).sum().item()/preds.shape[0]

            if(preds.shape[0]>node_train or opt_nodes==False):
                test_accy_summ[G.num_nodes][0] += accy
                test_accy_summ[G.num_nodes][1] += 1    
                test_loss += loss.item()
                test_graphs += 1
                test_loss_summ[G.num_nodes][0] += loss.item()
                test_loss_summ[G.num_nodes][1] += 1
    opt.zero_grad() #Don't train Theta on finetune test set when optmizing nodes
    if mesh_opt is not None:
        mesh_opt.step()
        mesh_opt.zero_grad()
        update_meshes_after_opt(mesh_list, epoch=epoch, writer=writer)

    if do_tensorboard:
        for num in sqrt_num_nodes_list:
            writer.add_scalar('test/loss-'+str(num**2),
                    test_loss_summ[num**2][0]/test_loss_summ[num**2][1],epoch)
    else:
        print('train/loss-'+str(num**2),
        train_loss_summ[num**2][0]/train_loss_summ[num**2][1],epoch)
        print('test/loss-'+str(num**2),
            test_loss_summ[num**2][0]/test_loss_summ[num**2][1],epoch)

        print('train/accy-'+str(num**2),
            train_accy_summ[num**2][0]/train_accy_summ[num**2][1],epoch)
        print('test/accy-'+str(num**2),
            test_accy_summ[num**2][0]/test_accy_summ[num**2][1],epoch)

    loss_curves[num**2].append((train_loss_summ[num**2][0]/train_loss_summ[num**2][1],
                            test_loss_summ[num**2][0]/test_loss_summ[num**2][1]))
    accy_curves[num**2].append((train_accy_summ[num**2][0]/train_accy_summ[num**2][1],
                            test_accy_summ[num**2][0]/test_accy_summ[num**2][1]))

torch.save(model,'logs/3x3-no-opt-e100m-model.pt')
torch.save(mesh_list,'logs/3x3-no-opt-e100m-mesh-list.pt')
torch.save(mesh_params,'logs/3x3-no-opt-e100m-mesh-params.pt')   
torch.save(loss_curves,'logs/3x3-no-opt-e100m-loss-curves.pt')
torch.save(accy_curves,'logs/3x3-no-opt-ee100mm-accy-curves.pt')