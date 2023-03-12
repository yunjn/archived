import os
import sys
import json
import torch
from tqdm import tqdm
import torch.nn as nn
from model import resnet50
import torch.optim as optim
from torchvision import transforms, datasets
from visdom import Visdom  # 可视化


def train(batch_size=4, epochs=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(
        os.getcwd(), "./.."))
    image_path = os.path.join(data_root, "dataset", "fog_data")
    assert os.path.exists(
        image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    fog_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in fog_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('../resources/out/classes.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet50()
    model_weight_path = "../resources/resnet/resnet50.pth"
    assert os.path.exists(
        model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 3)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = '../resources/out/fog.pth'
    train_steps = len(train_loader)
    # 将窗口类实例化
    #viz = Visdom()
    #viz.line([[0., 0.]], [0], win='train', opts=dict(
    #    title='loss&acc', legend=['loss', 'acc']))
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        # 可视化
        #viz.line([[running_loss / train_steps, val_accurate]],
        #         [epoch], win='train', update='append')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('FINISHED!!!')


if __name__ == '__main__':
    num_argv = len(sys.argv)
    if num_argv == 1:
        train(epochs=5)
    elif num_argv == 2:
        train(16, int(sys.argv[1]))
    elif num_argv == 3:
        train(int(sys.argv[2]), int(sys.argv[1]))
    else:
        print('参数错误！')
