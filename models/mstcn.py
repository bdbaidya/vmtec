import torch.nn as nn

class MSTCN(nn.Module):
    def __init__(self, input_dim=768, num_classes=7, num_stages=4, num_layers=10, hidden_dim=64):
        super(MSTCN, self).__init__()
        self.stage1 = SingleStageModel(input_dim, num_classes, num_layers, hidden_dim)
        self.stages = nn.ModuleList([
            SingleStageModel(num_classes, num_classes, num_layers, hidden_dim) for _ in range(num_stages - 1)
        ])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for stage in self.stages:
            out = stage(torch.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers, hidden_dim):
        super(SingleStageModel, self).__init__()
        self.conv1x1 = nn.Conv1d(input_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(hidden_dim, hidden_dim, dilation=2 ** i) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        out = self.conv1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]


class DilatedResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out
