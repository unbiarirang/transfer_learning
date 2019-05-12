import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(input), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 48 * 4 * 4)

        return x

class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):
        x = F.dropout(F.relu(self.fc1(input)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, 1)

class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        x = F.relu(self.fc1(input))
        x = F.log_softmax(self.fc2(x), 1)

        return x


e = Extractor()
c = Class_classifier()
d = Domain_classifier()
input = torch.randn(1, 3, 28, 28)
feature = e(input)
label = c(feature)
domain = d(feature, 0.01)
print(feature.size())
print(label.size())
print(domain.size())

# Train
src_inputs = torch.randn(10, 3, 28, 28)
tgt_inputs = torch.randn(10, 3, 28, 28)
label = torch.cat([torch.Tensor([1,0,0,0,0,0,0,0,0,0]) for _ in range(10)], out=torch.Tensor(src_inputs.size()[0], 10)).view(-1, 10)
print(label)
src_label = Variable(torch.zeros(src_inputs.size()[0]))
tgt_label = Variable(torch.ones(tgt_inputs.size()[0]))
EPOCH = 10
CONST = 0.01
THETA = 0.01
class_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()
optimizer = optim.Adam([{'params': e.parameters()},
                        {'params': c.parameters()},
                        {'params': d.parameters()}])

for epoch in range(EPOCH):
    print('Epoch: {}'.format(epoch+1))

    # Compute the feature or source domain and target domain
    src_feature = e(src_inputs)
    tgt_feature = e(tgt_inputs)

    # Compute the classification loss of src_feature
    class_pred = c(src_feature)
    class_loss = class_criterion(class_pred, label)

    # Compute the domain loss of src_feature and tgt_feature
    src_pred = d(src_feature, CONST)
    tgt_pred = d(tgt_feature, CONST)
    src_loss = domain_criterion(src_pred, src_label)
    tgt_loss = domain_criterion(tgt_pred, tgt_label)
    domain_loss = src_loss + tgt_loss

    loss = class_loss + THETA * domain_loss
    loss.backward()
    optimizier.step()

    print('Loss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
           loss.item(), class_loss.item(), domain_loss.item()))
