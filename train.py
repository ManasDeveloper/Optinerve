import torch
from tqdm import tqdm  # progress bar

def train(model, criterion, optimizer, epochs, train_loader, device='cuda'):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        # tqdm progress bar
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # Update tqdm progress bar with batch loss/acc
            loop.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Accuracy": f"{100 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"\n✅ Epoch {epoch+1} Summary: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%\n")


