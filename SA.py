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
sqrt_num_nodes_list = [2]#[4,3,4,5,6,7]
copies_per_graph = 1
opt_nodes = False
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

train_loader = DataLoader(train_dataset, batch_size=3000, num_workers=8,
        shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset,  batch_size=bs, num_workers=8,
        shuffle=True, drop_last=False)

loss_fn = nn.CrossEntropyLoss()


assert min(sqrt_num_nodes_list) >= 1
# model = GENPlanarGrid(encoders=encoders, decoders=decoders)
model = torch.load('/content/gen-mnist/logs/2x2-no-opt-e200m-model.pt')
mesh_list, mesh_params = create_mesh_list(
        num_datasets= 1,
        sqrt_num_nodes_list=sqrt_num_nodes_list,
        initialization='random' if opt_nodes else 'uniform',
        copies_per_graph=copies_per_graph, device=device)
max_mesh_list_elts = max([len(aux) for aux in mesh_list])
if cuda: model.cuda()
opt = torch.optim.Adam(params=model.parameters(), lr=3e-5)
if len(mesh_params):
    mesh_opt = torch.optim.Adam(params=mesh_params, lr=3e-4)
else: mesh_opt = None

if do_tensorboard: writer = SummaryWriter()
else: writer = None

  
coords = [(y/27.0,x/27.0) for y in range(0,28,1) for x in range(0,28,1)]
coords = torch.Tensor(coords)
if cuda:
    coords = coords.cuda()

sol = torch.randperm(784,device=device)[:250]

sol = torch.LongTensor([244, 480, 326, 719,  84,  17, 220, 284, 651, 710, 195, 747, 626, 695,
        229, 109, 542, 282, 386, 397, 362, 498, 342, 297, 182,  65, 585, 700,
        614, 173, 359, 294, 601, 390, 565, 776, 428, 587, 217, 237,  27, 775,
        721, 499, 414, 771, 416,  64, 658, 444, 644, 133, 179, 343, 150, 134,
        111, 663, 357, 153, 345, 163, 451, 703, 607, 641,  82, 661, 434, 764,
        758, 128, 170, 275, 157, 162, 200, 578, 680, 612, 233,  95,  50, 375,
        716, 554, 519, 409, 427, 219, 246, 514, 755, 267,   3, 242, 552, 583,
        759, 138, 407, 389, 259, 461, 744, 395, 570, 731, 571, 314, 377, 424,
         36, 172, 412, 555, 489,  53, 104, 141, 176, 228, 158, 177,  86, 436,
        545, 743, 779, 184, 119, 533, 212, 378, 283, 654, 458,   8,  72, 224,
         28,  32, 430, 353, 306, 683, 263, 168, 311,  81, 503, 738, 537, 666,
        238, 102, 382, 562, 704,   6, 516, 521, 512, 493, 225, 528, 591, 523,
        483, 605, 495,  57, 384, 235, 323, 712, 262,  99,  62, 595, 232,  26,
        341, 500, 438, 124, 603, 522,   2, 187, 752, 160, 678, 657,  60, 301,
        320, 469, 650, 504, 720, 553, 305,   4, 147,  42, 705, 484, 580, 324,
        344,  80, 374, 665,  41, 319, 674, 510, 563, 351, 445, 207, 646, 203,
        317, 783, 329, 108, 106, 245, 431, 367, 300, 741, 596, 567,  73, 617,
        699, 530, 633, 604,  69,  21, 337, 749,  91, 205, 268, 258])


def neighbour(sol):
  not_inside = [ i for i in range(0,784,1) if i not in sol]
  rand_idx1 = torch.randint(high=len(not_inside),size=(1,))[0].item()
  rand_idx2 = torch.randint(high=250,size=(1,))[0].item()
  new_px = not_inside[rand_idx1]
  new_sol = sol.clone()
  new_sol[rand_idx2] = new_px
  return new_sol

def val(sol):
  test_accy_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}
    
  for cnt, (Inp,Out) in enumerate(Tqdm(test_loader)):
    if cuda:
        Inp = Inp.cuda()
        Out = Out.cuda()
    G = mesh_list[0][0]
    Inp = Inp.view(-1,1,784)
    Inp = Inp[:,:,sol]
    Inp = (coords[sol],Inp)
    Q = None
    targets = Out
    preds = model(Inp, Q, G=G)

    accy  = ((torch.max(preds,1)[1]-targets)==0).sum().item()
    test_accy_summ[G.num_nodes][0] += accy
    test_accy_summ[G.num_nodes][1] += preds.shape[0]
  return test_accy_summ[G.num_nodes][0]/test_accy_summ[G.num_nodes][1]

def acceptance_probability(old_cost,new_cost,T):
  return np.exp(-(new_cost.item()-old_cost.item())/T)
 
def SA(sol): 
  T =  0.01
  T_min = 0.0001
  alpha = 0.9
  opt_val_sol = None
  opt_val_accy = 0
  while T > T_min:
    for i in range(0,5,1):
      for cnt, (Inp,Out) in enumerate(Tqdm(train_loader)):
          if cuda:
              Inp = Inp.cuda()
              Out = Out.cuda()
          G = mesh_list[0][0]
          Inp = Inp.view(-1,1,784)

          Inp_o = Inp[:,:,sol]
          Inp_o = (coords[sol],Inp_o)
          Q = None
          targets = Out
          preds = model(Inp_o, Q, G=G)
          old_cost = loss_fn(preds,targets)
          old_accy =  ((torch.max(preds,1)[1]-targets)==0).sum().item()/preds.shape[0]

          new_sol = neighbour(sol)
          Inp = Inp[:,:,new_sol]
          Inp = (coords[new_sol],Inp)
          Q = None
          targets = Out
          preds = model(Inp, Q, G=G)
          new_cost  = loss_fn(preds,targets)
          new_accy  = ((torch.max(preds,1)[1]-targets)==0).sum().item()/preds.shape[0]
          ap = acceptance_probability(old_cost, new_cost, T)

          print(old_cost.item(),new_cost.item(),ap)
          if ap > torch.rand((1,))[0].item():
                sol = new_sol
                old_cost = new_cost
    T = T*alpha
    val_accy = val(sol)
    
    print('Validation accuracy : ',val_accy)
    if(val_accy > opt_val_accy):
      opt_val_accy = val_accy
      opt_val_sol = sol.clone()
      print(opt_val_sol)

  return(opt_val_sol,opt_val_accy)

  # SA(sol)
  print(val(sol))
  