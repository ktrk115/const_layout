import cma
import torch


class AdamOptimizer():
    def __init__(self, lr=0.01, iteration=200):
        self.lr = lr
        self.iteration = iteration

    def generator(self, z, objective, **kwargs):
        z = z.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=self.lr)
        for _ in range(self.iteration):
            if z.grad is not None:
                z.grad.zero_()
            loss = objective(z)  # [B]
            loss.sum().backward()
            optimizer.step()
            yield z.detach().requires_grad_(False)

    def optimize(self, z, objective, **kwargs):
        for z_opt in self.generator(z, objective):
            pass
        return z_opt


class CMAESOptimizer():
    def __init__(self, sigma0=0.25, iteration=200, seed=None):
        self.sigma0 = sigma0
        self.option = {
            'maxiter': iteration,
            'verbose': -9,
        }

        self.seed = seed
        if seed is not None:
            # pycma issue #111
            self.option['seed'] = seed + 1

    def generator(self, z, objective, mask, **kwargs):
        B, N, D = z.size()
        device = z.device

        es_list = []
        for i in range(B):
            x0 = z[i][mask[i]].flatten().cpu().numpy()
            es = cma.CMAEvolutionStrategy(x0, self.sigma0, self.option)
            es_list.append(es)

        x = torch.zeros(B, max(es.popsize for es in es_list), N * D)
        x_best = torch.zeros(B, N * D)

        while not all(es.stop() for es in es_list):
            x_list = []
            for i, es in enumerate(es_list):
                _x = es.ask()
                x_list.append(_x)
                x[i, :es.popsize, :es.N] = torch.as_tensor(_x)

            with torch.no_grad():
                loss = objective(x)

            for i, es in enumerate(es_list):
                es.tell(x_list[i], loss[i, :es.popsize])
                x_best[i, :es.N] = torch.as_tensor(es.best.x)
            yield x_best.view(B, N, D).to(device)

    def optimize(self, z, objective, mask, **kwargs):
        for z_opt in self.generator(z, objective, mask):
            pass
        return z_opt
