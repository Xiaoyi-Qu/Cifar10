import sys
sys.path.append('..')
from sanity_check.backends.resnet_cifar10 import resnet18_cifar10
from only_train_once import OTO
import torch
import os

model = resnet18_cifar10()
dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

trainset = CIFAR10(root='cifar10', train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
testset = CIFAR10(root='cifar10', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

trainloader =  torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

optimizer = oto.hessocric(
    variant='sgd',
    lr=0.1,
    weight_decay=0.0,
    first_momentum=0.9,
    target_group_sparsity=0.7,
    start_cric_step=5,
    max_cycle_period=50,
    sampling_steps=5,
    hybrid_training_steps=20,
)

criterion = torch.nn.CrossEntropyLoss()

max_step = 200

for step, (X, y) in enumerate(trainloader):
    if step > max_step:
        break
    X = X.cuda()
    y = y.cuda()
    y_pred = model.forward(X)
    f = criterion(y_pred, y)
    optimizer.zero_grad()
    f.backward()
    optimizer.step(loss=f)

    opt_metric = optimizer.compute_metrics()
    print("S: {step}, loss: {f:.4f}, num_zero_grps: {num_zero_group}, gs: {group_sparsity:.2f}, norm_params: {norm_params:.2f}, norm_import: {norm_import:.2f}, norm_viol: {norm_violating:.2f}, norm_redund: {norm_redund:.2f}, num_grps_import: {num_grps_import}, num_grps_redund: {num_grps_redund}, num_grps_viol: {num_grps_violating}, num_grps_trial_viol: {num_grps_trial_violating}, num_grps_hist_viol: {num_grps_historical_violating}"\
            .format(step=step, f=f.item(), num_zero_group=opt_metric.num_zero_groups, group_sparsity=opt_metric.group_sparsity, norm_params=opt_metric.norm_params, norm_import=opt_metric.norm_important_groups, norm_violating=opt_metric.norm_violating_groups, \
                    norm_redund=opt_metric.norm_redundant_groups, num_grps_import=opt_metric.num_important_groups, num_grps_redund=opt_metric.num_redundant_groups, num_grps_violating=opt_metric.num_violating_groups, num_grps_trial_violating=opt_metric.num_trial_violating_groups, num_grps_historical_violating=opt_metric.num_historical_violating_groups
        ))

oto.construct_subnet(out_dir='./cachee')

full_model = torch.load(oto.full_group_sparse_model_path).cpu()
compressed_model = torch.load(oto.compressed_model_path).cpu()

full_output = full_model(dummy_input)
compressed_output = compressed_model(dummy_input)

max_output_diff = torch.max(torch.abs(full_output - compressed_output))
print("Maximum output difference " + str(max_output_diff.item()))
full_model_size = os.stat(oto.full_group_sparse_model_path)
compressed_model_size = os.stat(oto.compressed_model_path)
print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")