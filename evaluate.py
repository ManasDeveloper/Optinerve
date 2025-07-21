import torch
from tqdm import tqdm

def evaluate(model, test_loader, criterion, device='cuda', label_encoder=None):
    model.eval()
    model.to(device)

    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating", leave=False)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            loop.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Accuracy": f"{100 * correct / total:.2f}%"
            })

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f"\n📊 Test Set — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
