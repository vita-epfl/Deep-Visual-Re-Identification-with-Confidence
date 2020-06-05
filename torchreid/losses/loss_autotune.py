import torch
import numpy as np

class MultiHeadLossAutoTune(torch.nn.Module):
    def __init__(self, losses, lambdas):
        """Auto-tuning multi-head less.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super(MultiHeadLossAutoTune, self).__init__()
        #assert all(l >= 0.0 for l in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.log_sigmas = torch.nn.Parameter(
            torch.zeros((len(lambdas),), dtype=torch.float32),
            requires_grad=True,
        )

        #self.field_names = [n for l in self.losses for n in l.field_names]
        #LOG.info('multihead loss with autotune: %s', self.field_names)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, *args):
        head_fields, head_targets = args
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [l(f, t)
                            for l, f, t in zip(self.losses, head_fields, head_targets)]

        assert len(self.log_sigmas) == len(flat_head_losses)
        loss_values = np.array([lam * l / (2.0 * (log_sigma.exp() ** 2))
                       for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                       if l is not None])
        auto_reg = [lam * log_sigma
                    for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                    if l is not None]
        total_loss = sum(loss_values) + sum(auto_reg) if not(loss_values is None) else None

        return total_loss, flat_head_losses
