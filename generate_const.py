import os
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import pickle
import argparse
import tempfile
import subprocess
from tqdm import tqdm
from pathlib import Path

import torch
import torchvision.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

from data import get_dataset
from util import set_seed, convert_layout_to_image
from data.util import AddCanvasElement, AddRelation
from model.layoutganpp import Generator, Discriminator

import clg.const
from clg.auglag import AugLagMethod
from clg.optim import AdamOptimizer, CMAESOptimizer
from metric import compute_violation


def save_gif(out_path, j, netG,
             z_hist, label, mask, padding_mask,
             dataset_colors, canvas_size):
    mask = mask[j]
    _j = slice(j, j + 1)

    z_before, z_filtered = None, []
    for z in z_hist:
        if z_before is not None:
            if z_before.eq(z[_j]).all():
                continue
        z_filtered.append(z)
        z_before = z[_j]
    z_filtered += [z] * 2

    with tempfile.TemporaryDirectory() as tempdir:
        for i, z in enumerate(z_filtered):
            bbox = netG(z[_j], label[_j], padding_mask[_j])
            b = bbox[0][mask].cpu().numpy()
            l = label[0][mask].cpu().numpy()

            convert_layout_to_image(
                b, l, dataset_colors, canvas_size
            ).save(tempdir + f'/{j}_{i:08d}.png')

        subprocess.run(['convert', '-delay', '50',
                        tempdir + f'/{j}_*.png', str(out_path)])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('ckpt_path', type=str, help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('-o', '--out_path', type=str,
                        default='output/generated_layouts.pkl',
                        help='output pickle path')
    parser.add_argument('--num_save', type=int, default=0,
                        help='number of layouts to save as images')
    parser.add_argument('--seed', type=int, help='manual seed')

    # CLG specific options
    parser.add_argument('--const_type', type=str,
                        default='beautify', help='constraint type',
                        choices=['beautify', 'relation'])
    parser.add_argument('--optimizer', type=str,
                        default='CMAES', help='inner optimizer',
                        choices=['Adam', 'CMAES'])
    parser.add_argument('--rel_ratio', type=float, default=0.1,
                        help='ratio of relational constraints')

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    out_path = Path(args.out_path)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # load checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = ckpt['args']

    # setup transforms and constraints
    transforms = [AddCanvasElement()]
    if args.const_type == 'relation':
        transforms += [AddRelation(args.seed, args.rel_ratio)]
        constraints = clg.const.relation
    else:
        constraints = clg.const.beautify

    # load test dataset
    dataset = get_dataset(train_args['dataset'], 'test',
                          T.Compose(transforms))
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)
    num_label = dataset.num_classes

    # setup model and load state
    netG = Generator(train_args['latent_size'], num_label,
                     d_model=train_args['G_d_model'],
                     nhead=train_args['G_nhead'],
                     num_layers=train_args['G_num_layers'],
                     ).eval().requires_grad_(False).to(device)
    netG.load_state_dict(ckpt['netG'])

    netD = Discriminator(num_label,
                         d_model=train_args['D_d_model'],
                         nhead=train_args['D_nhead'],
                         num_layers=train_args['D_num_layers'],
                         ).eval().requires_grad_(False).to(device)
    netD.load_state_dict(ckpt['netD'])

    # setup optimizers
    if args.optimizer == 'CMAES':
        inner_optimizer = CMAESOptimizer(seed=args.seed)
    else:
        inner_optimizer = AdamOptimizer()
    optimizer = AugLagMethod(netG, netD, inner_optimizer, constraints)

    results, violation = [], []
    for data in tqdm(dataloader, ncols=100):
        data = data.to(device)
        label_c, mask_c = to_dense_batch(data.y, data.batch)
        label = torch.relu(label_c[:, 1:] - 1)
        mask = mask_c[:, 1:]
        padding_mask = ~mask

        z = torch.randn(label.size(0), label.size(1),
                        train_args['latent_size'],
                        device=device)

        z_hist = [z]
        for z in optimizer.generator(z, data):
            if len(results) < args.num_save:
                z_hist.append(z)

        bbox = netG(z, label, padding_mask)

        if args.const_type == 'relation':
            canvas = optimizer.bbox_canvas.to(bbox)
            canvas = canvas.expand(bbox.size(0), -1, -1)
            bbox_flatten = torch.cat([canvas, bbox], dim=1)[mask_c]
            v = compute_violation(bbox_flatten, data)
            violation += v[~v.isnan()].tolist()

        if len(results) < args.num_save:
            bbox_init = netG(z_hist[0], label, padding_mask)

        for j in range(bbox.size(0)):
            mask_j = mask[j]
            b = bbox[j][mask_j].cpu().numpy()
            l = label[j][mask_j].cpu().numpy()

            if len(results) < args.num_save:
                out_path = out_dir / f'initial_{len(results)}.png'
                convert_layout_to_image(
                    bbox_init[j][mask_j].cpu().numpy(),
                    l, dataset.colors, (120, 80)
                ).save(out_path)

                out_path = out_dir / f'optimized_{len(results)}.png'
                convert_layout_to_image(
                    b, l, dataset.colors, (120, 80)
                ).save(out_path)

                out_path = out_dir / f'optimizing_{len(results)}.gif'
                save_gif(out_path, j, netG,
                         z_hist, label, mask, padding_mask,
                         dataset.colors, (120, 80))

            results.append((b, l))

    if args.const_type == 'relation':
        violation = sum(violation) / len(violation)
        print(f'Relation violation: {violation:.2%}')

    # save results
    with Path(args.out_path).open('wb') as fb:
        pickle.dump(results, fb)
    print('Generated layouts are saved at:', args.out_path)


if __name__ == '__main__':
    main()
