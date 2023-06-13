import torch
from torch import nn
from torch.optim import SGD
import numpy
import skeletalModel
class BackpropagationBasedFiltering(nn.Module):
    def __init__(self, lines0_values, rootsx0_values, rootsy0_values, rootsz0_values,
                 anglesx0_values, anglesy0_values, anglesz0_values, structure, dtype,
                 learning_rate=0.1, n_cycles=1000, regulator_rates=[0.001, 0.1]):
        super(BackpropagationBasedFiltering, self).__init__()

        self.T = rootsx0_values.shape[0]
        self.n_bones, self.n_points = skeletalModel.structureStats(structure)
        self.n_limbs = len(structure)

        self.structure = structure
        self.learning_rate = learning_rate
        self.n_cycles = n_cycles
        self.regulator_rates = regulator_rates

        self.lines = nn.Parameter(torch.tensor(lines0_values, dtype=dtype))

        self.rootsx = nn.Parameter(torch.tensor(rootsx0_values, dtype=dtype))
        self.rootsy = nn.Parameter(torch.tensor(rootsy0_values, dtype=dtype))
        self.rootsz = nn.Parameter(torch.tensor(rootsz0_values, dtype=dtype))

        self.anglesx = nn.Parameter(torch.tensor(anglesx0_values, dtype=dtype))
        self.anglesy = nn.Parameter(torch.tensor(anglesy0_values, dtype=dtype))
        self.anglesz = nn.Parameter(torch.tensor(anglesz0_values, dtype=dtype))

        self.optimizer = SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, tarx_values, tary_values, w_values):
        epsilon = 1e-10
        x = [None for _ in range(self.n_points)]
        y = [None for _ in range(self.n_points)]
        z = [None for _ in range(self.n_points)]

        # head
        x[0] = self.rootsx
        y[0] = self.rootsy
        z[0] = self.rootsz

        i = 0
        for a, b, l in self.structure:
            L = torch.exp(self.lines[l])
            Ax = self.anglesx[0:self.T, i:(i + 1)]
            Ay = self.anglesy[0:self.T, i:(i + 1)]
            Az = self.anglesz[0:self.T, i:(i + 1)]
            normA = torch.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2) + epsilon

            x[b] = x[a] + L * Ax / normA
            y[b] = y[a] + L * Ay / normA
            z[b] = z[a] + L * Az / normA

            i += 1

        x = torch.cat(x, axis=1)
        y = torch.cat(y, axis=1)
        z = torch.cat(z, axis=1)

        loss = torch.sum(w_values * (x - tarx_values) ** 2 + w_values * (y - tary_values) ** 2) / (
                    self.T * self.n_points)
        reg1 = torch.sum(torch.exp(self.lines))
        dx = x[0:(self.T - 1), 0:self.n_points] - x[1:self.T, 0:self.n_points]
        dy = y[0:(self.T - 1), 0:self.n_points] - y[1:self.T, 0:self.n_points]
        dz = z[0:(self.T - 1), 0:self.n_points] - z[1:self.T, 0:self.n_points]
        reg2 = torch.sum(dx ** 2 + dy ** 2 + dz ** 2) / ((self.T - 1) * self.n_points)
        total_loss = loss + self.regulator_rates[0] * reg1 + self.regulator_rates[1] * reg2

        return x, y, z, total_loss

    def train_model(self, tarx_values, tary_values, w_values):
        for i_cycle in range(self.n_cycles):
            self.optimizer.zero_grad()
            x, y, z, loss = self.forward(tarx_values, tary_values, w_values)
            loss.backward()
            self.optimizer.step()
        
        # 다음 단계에서 detach, cpu, numpy 과정 필요
        return x, y, z

if __name__ == "__main__":
    # debug - don't run it

    #
    #             (0)
    #              |
    #              |
    #              0
    #              |
    #              |
    #     (2)--1--(1)--1--(3)
    #
    structure = (
        (0, 1, 0),
        (1, 2, 1),
        (1, 3, 1),
    )

    T = 3
    nBones, nPoints = skeletalModel.structureStats(structure)
    nLimbs = len(structure)

    dtype = numpy.float32

    lines0_values = numpy.zeros((nBones,), dtype=dtype)
    rootsx0_values = numpy.ones((T, 1), dtype=dtype)
    rootsy0_values = numpy.ones((T, 1), dtype=dtype)
    rootsz0_values = numpy.ones((T, 1), dtype=dtype)
    anglesx0_values = numpy.ones((T, nLimbs), dtype=dtype)
    anglesy0_values = numpy.ones((T, nLimbs), dtype=dtype)
    anglesz0_values = numpy.ones((T, nLimbs), dtype=dtype)

    w_values = numpy.ones((T, nPoints), dtype=dtype)
    w_tensor = torch.tensor(w_values, dtype=torch.float32)
    tarx_values = numpy.ones((T, nPoints), dtype=dtype)
    tarx_values = torch.tensor(tarx_values, dtype=torch.float32)
    tary_values = numpy.ones((T, nPoints), dtype=dtype)
    tary_values = torch.tensor(tary_values, dtype=torch.float32)

    dtype = torch.float32
    test_model = BackpropagationBasedFiltering(
        lines0_values,
        rootsx0_values,
        rootsy0_values,
        rootsz0_values,
        anglesx0_values,
        anglesy0_values,
        anglesz0_values,
        structure,
        dtype)

    result = test_model.train_model(tarx_values, tary_values, w_tensor)

    test = 1