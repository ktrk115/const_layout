import torch
from torch_geometric.utils import to_dense_batch


class AugLagMethod():
    def __init__(self, netG, netD, inner_optimizer, constraints,
                 alpha=3., l0=0., m0=1., iteration=5, tolerance=1e-8,
                 clamp_f=True, raise_error_if_failed=False):
        self.netG = netG
        self.netD = netD
        self.inner_optimizer = inner_optimizer
        self.constraints = constraints

        self.alpha = alpha
        self.l0 = l0
        self.m0 = m0
        self.iteration = iteration
        self.tolerance = tolerance

        self.clamp_f = clamp_f
        self._f0 = None
        self.raise_error = raise_error_if_failed

        # bbox_canvas: [1, 1, 4]
        self.bbox_canvas = torch.tensor(
            [[[.5, .5, 1., 1.]]],
            dtype=torch.float,
        )

    def f(self, bbox, label, padding_mask):
        f = -torch.sigmoid(self.netD(bbox, label, padding_mask))
        return f

    def h(self, bbox, data, mask_c):
        B = bbox.size(0)
        canvas = self.bbox_canvas.to(bbox.device)

        if len(bbox.size()) == 4:
            P = bbox.size(1)
            canvas = canvas.unsqueeze(0).expand(B, P, -1, -1)
            bbox_c_pop = torch.cat([canvas, bbox], dim=2)
            bbox_flatten = bbox_c_pop.transpose(1, 2)[mask_c]
        else:
            canvas = canvas.expand(B, -1, -1)
            bbox_c = torch.cat([canvas, bbox], dim=1)
            bbox_flatten = bbox_c[mask_c]

        return torch.stack([
            const(bbox_flatten, data)
            for const in self.constraints
        ], dim=-1)

    def build_Adam_objective(self, l, m, data, label, padding_mask, mask_c):
        def objective(z):
            bbox = self.netG(z, label, padding_mask)

            f = self.f(bbox, label, padding_mask)
            if self.clamp_f:
                f = torch.relu(f - self._f0[:f.size(0)])
            h = self.h(bbox, data, mask_c)

            h_sqr = h.square().sum(dim=-1)
            L = f + (l * h).sum(dim=-1) + m / 2 * h_sqr

            return L
        return objective

    def build_CMAES_objective(self, l, m, data, label, padding_mask, mask_c):
        def objective(x):
            B, P, ND = x.size()
            N = label.size(1)
            D = int(ND / N)

            z = x.view(-1, N, D).to(data.x.device)
            _label = label.unsqueeze(1)
            _label = _label.expand(-1, P, -1).reshape(-1, N)
            _padding_mask = padding_mask.unsqueeze(1)
            _padding_mask = _padding_mask.expand(-1, P, -1).reshape(-1, N)

            bbox = self.netG(z, _label, _padding_mask)

            f = self.f(bbox, _label, _padding_mask).view(B, P)
            if self.clamp_f:
                f = torch.relu(f - self._f0[:f.size(0)])
            h = self.h(bbox.view(B, P, N, D), data, mask_c)

            h_sqr = h.square().sum(dim=-1)
            lh = (l.unsqueeze(1) * h).sum(dim=-1)
            L = f + lh + m / 2 * h_sqr

            return L.cpu().numpy()

        return objective

    def generator(self, z, data):
        assert data.attr[0]['has_canvas_element']

        C = len(self.constraints)
        l = torch.full((z.size(0), C), self.l0).to(z)
        m = self.m0

        label_c, mask_c = to_dense_batch(data.y, data.batch)
        label = torch.relu(label_c[:, 1:] - 1)
        mask = mask_c[:, 1:]
        padding_mask = ~mask

        bbox = self.netG(z, label, padding_mask)

        if self.clamp_f:
            self._f0 = self.f(bbox, label, padding_mask)
            if 'CMAES' in str(type(self.inner_optimizer)):
                self._f0 = self._f0.unsqueeze(1)

        h = self.h(bbox, data, mask_c)
        h_sqr = h.square().sum(dim=-1)
        stop = m / 2 * h_sqr < self.tolerance

        for _ in range(self.iteration):
            if stop.all():
                break

            if 'CMAES' in str(type(self.inner_optimizer)):
                build = self.build_CMAES_objective
            else:
                build = self.build_Adam_objective
            objective = build(l, m, data, label, padding_mask, mask_c)

            _stop = stop.unsqueeze(-1).unsqueeze(-1)
            iterator = self.inner_optimizer.generator(z, objective, mask=mask)
            for z_opt in iterator:
                _z = torch.where(_stop, z, z_opt)
                yield _z

            z = _z
            bbox = self.netG(z, label, ~mask)

            h = self.h(bbox, data, mask_c)
            h_sqr = h.square().sum(dim=-1)

            l = l + m * h
            m = self.alpha * m
            stop = m / 2 * h_sqr < self.tolerance

        if self.raise_error and not stop.all():
            raise RuntimeError('Failed to find solution')

    def optimize(self, z, data):
        for z_opt in self.generator(z, data):
            pass
        return z_opt
