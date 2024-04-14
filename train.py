from clearml import Task
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from engine import train_step, valid_step
from data_setup import create_dataloaders
from config import DATA_DIR

torch.manual_seed(42)
torch.cuda.manual_seed(42)

from tqdm.auto import tqdm

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          params: dict,
          task_name: str,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 15):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Always initialize ClearML before anything else. Automatic hooks will track as
    # much as possible for you (such as in this case TensorBoard logs)!
    task = Task.init(project_name="Leukocytes Classification", task_name=task_name)
    writer = SummaryWriter()
    task.connect(params)
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = valid_step(model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {test_loss:.4f} | "
            f"val_acc: {test_acc:.4f}"
        )
        writer.add_scalar("Training Loss", train_loss, epoch)
        writer.add_scalar("Validataion Loss", test_loss, epoch)
        writer.add_scalar("Training acccuracy", train_acc, epoch)
        writer.add_scalar("Validation accuraccy", test_acc, epoch)

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(test_loss)
        results["val_acc"].append(test_acc)

    writer.close()
    task.close()
    torch.save(model.state_dict(), "models/model.pth")
    # 6. Return the filled results at the end of the epochs
    return results


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    train_dataloader, val_dataloader = create_dataloaders(DATA_DIR,
                                      bs_train=32,
                                      bs_val=1,
                                      transforms=transform)
    device = "cuda" if torch.cuda.is_available else "cpu"
    mobile_net = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    num_classes = 5
    mobile_net.classifier = nn.Linear(mobile_net.last_channel, num_classes)
    mobile_net = mobile_net.to(device)
    
    NUM_EPOCHS = 15
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = mobile_net.parameters(), lr=0.001)

    from timeit import default_timer as timer
    start_time = timer()
    params = {
        "dataset":"PBC",
        "model":"MobileNetV2",
        "optimizer":"Adam",
        "batch_size":32,
        "learning_rate":0.001
    }

    # Train model_1
    mobilenet_results = train(model=mobile_net,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            optimizer=optimizer,
                            params = params,
                            task_name = "MobileNetV2 exp-1",
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")