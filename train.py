
import torch.nn as nn
import torch.optim as optim
import torch
from metrics import calculate_eer

# Function to validate the model
def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for spectrograms, labels in dataloader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs,embeddings = model(spectrograms)
            y_pred = torch.round(torch.sigmoid(outputs))
            loss = criterion(outputs, labels.unsqueeze(1).float())
            #loss = criterion(outputs, labels.unsqueeze(1).float())

            running_loss += loss.item() #* spectrograms.size(0)
            #running_corrects += (torch.max(outputs, 1)[1] == labels).sum().item()
            running_corrects += (y_pred == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)
            
            all_outputs.append(outputs)
            all_labels.append(labels)

    avg_loss = running_loss / total_samples
    avg_accuracy = running_corrects / total_samples
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    avg_eer = calculate_eer(all_outputs, all_labels)
    
    return avg_loss, avg_accuracy, avg_eer

# Training loop with accuracy metric
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu',checkpoint_path='model_checkpoint.pth'):
    model.to(device)
    best_eer = float('inf')
    patience = 15
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        all_outputs = []
        all_labels = []
        
        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs,embeddings = model(spectrograms)
            y_pred = torch.round(torch.sigmoid(outputs))
            loss = criterion(outputs, labels.unsqueeze(1).float())
            #loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() #* spectrograms.size(0)
            #running_corrects += (torch.max(outputs, 1)[1] == labels).sum().item()
            running_corrects += (y_pred == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        # Concatenate all outputs and labels for EER calculation
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        epoch_eer = calculate_eer(all_outputs, all_labels)
        
        # Validate the model
        val_loss, val_acc, val_eer = validate_model(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Train EER: {epoch_eer:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val EER: {val_eer:.4f}')
        
        # Checkpoint the model if the validation EER is improved
        if val_eer < best_eer:
            patience = 15 # reset patience counter for early stopping
            best_eer = val_eer
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1} with EER: {best_eer:.4f}')
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping criterion has met!!!')
                break
            
        # Step the scheduler
        scheduler.step(val_loss)
