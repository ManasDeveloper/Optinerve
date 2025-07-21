from PIL import Image
import torchvision.transforms as transforms
import torch

def predict_single_image(model, image_path, transform, device='cuda', label_encoder=None):
    model.eval()
    model.to(device)

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    if label_encoder:
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    else:
        predicted_label = str(predicted_idx)

    print(f"🖼️ Prediction for {image_path}: {predicted_label}")
    return predicted_label


def predict_pil_image(model, image_pil, transform, device='cuda', label_encoder=None):
    model.eval()
    model.to(device)

    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    return (
        label_encoder.inverse_transform([predicted_idx])[0]
        if label_encoder else str(predicted_idx)
    )