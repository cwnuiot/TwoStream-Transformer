import torch
import time
from torch.autograd import Variable
import argparse
from model import TwoStreamTransformer
import torch.nn as nn
from Data_Preprocessing import savedata
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import MultiStepLR
class Loading_Data(Dataset):
    def __init__(self, dataset):
        self.text = dataset[0]
        self.image = dataset[1]
        self.label = dataset[2]
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx]), self.label[idx]

def train(epoch, model, train_loader, optimizer, criterion, device,lr_scheduler):
    global steps
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        train_txt1,train_txt2,target=Variable(data[0].to(device)),Variable(data[1].to(device)),Variable(target.to(device))
        optimizer.zero_grad()
        train_txt1 = train_txt1.view(-1, 1, 1120)     #8*140=1120
        train_txt2 = torch.unsqueeze(train_txt2, dim=1)
        output = model(train_txt1,train_txt2)
        output = torch.squeeze(output, dim=0)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss
    lr_scheduler.step(epoch)

def test(epoch, model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    if epoch ==20:
      torch.save(model.state_dict(),'tssgcn.mdl')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            test_txt1,test_txt2,target=Variable(data[0].to(device)),Variable(data[1].to(device)),Variable(target.to(device))
            test_txt1 = test_txt1.view(-1, 1, 1120)    #8*140=1120
            test_txt2=torch.unsqueeze(test_txt2,dim=1)
            output = model(test_txt1,test_txt2).to(device)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description="解析命令行参数示例")
    parser.add_argument('--dataset', type=str, default='Couch_GB2_TXT/Couch_Letter_195', help='数据集的地址')
    parser.add_argument('--epochs', type=int, default=35, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--depth', type=int, default=6, help='Num')
    parser.add_argument('--num_features', type=int, default=108, help='Hiden')
    parser.add_argument('--num_heads', type=int, default=12, help='Heads')
    parser.add_argument('--num_class', type=int, default=52, help='class')
    parser.add_argument('--batch_size', type=int, default=64, )
    args = parser.parse_args()

    dataset = args.dataset
    traindata, testdata = savedata(dataset)
    train_dataset = Loading_Data(traindata)
    test_dataset = Loading_Data(testdata)
    # 设置训练参数
    epochs = args.epochs
    lr = args.lr
    depth = args.depth
    num_features = args.num_features
    num_heads = args.num_heads
    num_class = args.num_class
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model = TwoStreamTransformer(depth=depth, num_features=num_features, num_heads=num_heads, num_class=num_class)
    model = model.to(device)
    # 设置优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练和测试循环
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        time_start = time.time()
        train(epoch, model, train_loader, optimizer, criterion, device,lr_scheduler)
        test(epoch, model, test_loader, criterion, device)
        lr_scheduler.step()
        time_end = time.time()
        print(f'训练时间: {time_end - time_start:.2f} 秒')


if __name__ == '__main__':
    main()



