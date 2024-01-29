import torch
import torch.nn as nn

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions, targets):
        """
        predictions: 模型对整个batch的预测结果
        targets: 整个batch的真实执行时间
        """
        # 使用上三角索引获得任意两个预测结果的组合
        iu = torch.triu_indices(predictions.shape[0], predictions.shape[0], 1)
        pred1, pred2 = predictions[iu[0]], predictions[iu[1]]
        target1, target2 = targets[iu[0]], targets[iu[1]]

        # 计算成对差异
        pred_diff = pred1 - pred2

        # 计算目标标签：1表示第一个配置的执行时间更长，-1表示第二个配置的执行时间更长
        label = torch.sign(target1 - target2)
        
        # 计算成对损失
        loss = self.sigmoid(-label * pred_diff)
        loss = -torch.log(loss)  # 计算交叉熵损失

        return torch.mean(loss)

# 示例使用
ranknet_loss = RankNetLoss()

# 假设batch大小为4
predictions = torch.tensor([3.0, 2.5, 4.0, 1.0], requires_grad=True)  # 模型预测的执行时间
targets = torch.tensor([2.0, 3.0, 4.0, 1.0])  # 真实的执行时间

loss = ranknet_loss(predictions, targets)
print(loss)


class ListMLELoss(nn.Module):
    def forward(self, y_pred, y_true):
        """
        y_pred: 预测值，shape [batch_size, list_size]
        y_true: 真实值，shape [batch_size, list_size]
        """
        _, true_sorted_indices = y_true.sort(descending=True, dim=1)
        pred_sorted_by_true = y_pred.gather(1, true_sorted_indices)

        pred_sorted_by_true = pred_sorted_by_true.exp()
        pred_cumsum = pred_sorted_by_true.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

        listmle_loss = torch.log(pred_cumsum + 1e-10) - pred_sorted_by_true.log()
        return -listmle_loss.sum(dim=1).mean()


class ListNetLoss(nn.Module):
    def forward(self, y_pred, y_true):
        """
        y_pred: 预测值，shape [batch_size, list_size]
        y_true: 真实值，shape [batch_size, list_size]
        """
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        y_true_softmax = torch.softmax(y_true, dim=1)
        return -torch.sum(y_true_softmax * torch.log(y_pred_softmax + 1e-10), dim=1).mean()


# 示例使用ListNetLoss
listnet_loss = ListNetLoss()
y_pred = torch.tensor([[2.0, 1.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float32)
y_true = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=torch.float32)
loss = listnet_loss(y_pred, y_true)
print(loss)

