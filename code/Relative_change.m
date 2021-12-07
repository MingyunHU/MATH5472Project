function delta = Relative_change(loss_new, loss_old)
delta = abs((loss_new-loss_old)/loss_old);