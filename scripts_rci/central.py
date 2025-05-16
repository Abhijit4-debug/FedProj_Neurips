import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(filename="log.txt", 
                    filemode='w', 
                    format='%(asctime)s - %(message)s', 
                    level=logging.INFO)

def log_and_print(message):
    print(message)  
    logging.info(message)  

batch_size = 128
learning_rate = 0.001
num_epochs_teacher = 100
num_epochs_student = 5
distillation_epochs = 10
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])
mode="RGB"

mode = "RGB"

train_ds = datasets.ImageFolder(root='/home/mahdi/codes/data/cinic-10/train', transform=transform)
val_ds = datasets.ImageFolder(root='/home/mahdi/codes/data/cinic-10/valid', transform=transform)
test_ds = datasets.ImageFolder(root='/home/mahdi/codes/data/cinic-10/test', transform=transform)
whole_ds = torch.utils.data.ConcatDataset([train_ds, val_ds, test_ds])

train_size = 100000
length = len(whole_ds)
indices = list(range(length))

# Shuffle indices with fixed seed
np.random.seed(42)
np.random.shuffle(indices)

train_indices = indices[:train_size]
test_indices = indices[train_size:train_size + 30000]


train_ds_global = torch.utils.data.Subset(whole_ds, train_indices)
test_ds_global = torch.utils.data.Subset(whole_ds, test_indices)


# Define batch size
batch_size = 128  # Adjust as needed

# Create DataLoaders
train_loader = DataLoader(train_ds_global, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds_global, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# transform= transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
#         ])

# train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def load_tinyimagenet_data(datadir):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4807, 0.4574, 0.4083), (0.2056, 0.2035, 0.2041)),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4807, 0.4574, 0.4083), (0.2056, 0.2035, 0.2041)),
    ])
    
    train_ds = datasets.ImageNet(root=datadir, split='train', transform=transform_train)
    val_ds = datasets.ImageNet(root=datadir, split='val', transform=transform_val)
    
    labels = [847, 874, 471, 476, 764, 138,  49, 226, 100, 426, 815, 836, 338,
              669, 743, 912, 320, 843, 796, 322, 261, 136, 841, 460, 699, 935,
              949, 877,  61, 332, 416,  35, 227, 493,  32, 478, 660,  13, 451,
              438, 323, 867, 168,  40, 156, 455, 691, 223, 354, 495, 799, 432,
              158, 866, 657, 768, 183, 852, 727, 249, 402, 507,  12, 880, 995,
              233, 176, 776, 830, 586, 865, 475, 610, 534, 953, 490, 160, 386,
              117, 942, 675,  24, 538, 494, 266, 295, 272,  11,   9, 154, 967,
              901, 123, 649, 737, 121,  20, 439, 641, 398]

    subset_train = np.array([], dtype='int')
    train_targets = np.array(train_ds.targets)
    for label in labels:
        subset_train = np.hstack([subset_train, np.where(train_targets == label)[0][0:500]])
    public_ds_train = torch.utils.data.Subset(train_ds, subset_train)
    
    subset_val = np.array([], dtype='int')
    val_targets = np.array(val_ds.targets)
    for label in labels:
        subset_val = np.hstack([subset_val, np.where(val_targets == label)[0]])
    public_ds_val = torch.utils.data.Subset(val_ds, subset_val)
    
    return public_ds_train, public_ds_val

public_ds_train, _ = load_tinyimagenet_data("/home/mahdi/codes/data/imagenet_resized/")
public_loader = torch.utils.data.DataLoader(public_ds_train, batch_size=batch_size, shuffle=True)

# Define ResNet18 model
def get_resnet18(num_classes=100):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    log_and_print(f"Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
    return avg_loss, accuracy

def train_teacher(model, train_loader, test_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(num_epochs_teacher):
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        train_loss = total_loss / total
        train_acc = 100 * correct / total
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        log_and_print(f"Epoch [{epoch + 1}/{num_epochs_teacher}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        log_and_print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

def train_student(student, train_loader, test_loader, criterion, optimizer, device):
    student.train()
    for epoch in range(num_epochs_student):
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = student(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        train_loss = total_loss / total
        train_acc = 100 * correct / total
        test_loss, test_acc = evaluate_model(student, test_loader, criterion, device)

        log_and_print(f"Epoch [{epoch + 1}/{num_epochs_student}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        log_and_print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

def distill_with_kl(student, teacher, public_loader, test_loader, kl_div_loss, criterion, optimizer, device):
    teacher.eval()
    for epoch in range(distillation_epochs):
        student.train()
        total_kl_loss = 0.0
        correct = 0
        total = 0

        for inputs, _ in public_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            student_outputs = student(inputs)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            kl_loss = kl_div_loss(F.log_softmax(student_outputs / 2, dim=1), F.softmax(teacher_outputs / 2, dim=1))
            kl_loss.backward()
            optimizer.step()

            total_kl_loss += kl_loss.item() * inputs.size(0)
            _, predicted = student_outputs.max(1)
            correct += predicted.eq(teacher_outputs.argmax(dim=1)).sum().item()
            total += inputs.size(0)

        train_kl_loss = total_kl_loss / total
        train_acc = 100 * correct / total
        test_loss, test_acc = evaluate_model(student, test_loader, criterion, device)

        log_and_print(f"Distillation Round [{epoch + 1}/{distillation_epochs}], KL Loss: {train_kl_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        log_and_print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_model = get_resnet18(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=learning_rate)

log_and_print("Training Teacher Model...")
train_teacher(teacher_model, train_loader, test_loader, criterion, optimizer_teacher, device)

student_model = get_resnet18(num_classes=100).to(device)
optimizer_student = optim.Adam(student_model.parameters(), lr=learning_rate)

log_and_print("\nTraining Student Model...")
train_student(student_model, train_loader, test_loader, criterion, optimizer_student, device)
optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.00005)

kl_div_loss = nn.KLDivLoss(reduction='batchmean')
log_and_print("\nDistilling Knowledge from Teacher to Student...")
distill_with_kl(student_model, teacher_model, public_loader, test_loader, kl_div_loss, criterion, optimizer, device)

log_and_print("\nEvaluating Student Model...")
evaluate_model(student_model, test_loader, criterion, device)
