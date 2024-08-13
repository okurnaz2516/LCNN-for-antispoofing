
import torch 
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from dataio import AudioDataset
from feature_extraction import SpectrogramTransform
from torch.utils.data import Dataset, DataLoader
from lcnn_model import LCNN
from metrics import calculate_eer
from train import train_model, validate_model
from save_probabilities import save_output_probabilities
# Main function
def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    fixed_num_frames = 400  # Define a fixed number of frames or set to None to use protocol file value
    input_dim = (257, fixed_num_frames)
    num_classes = 2
    batch_size = 8
    num_epochs = 20
    
    # Determine if GPU is available
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Create the model
    model = LCNN(input_dim=input_dim, num_classes=num_classes)

    #model = LCNN(data_shape=input_dim, LDO_p1=0.75, LDO_p2=0.00)
    # Create the training and validation datasets and dataloaders
    train_subset = 'train'
    val_subset = 'dev'
    Train_protocol_file = './asvspoof2017/protocol_V2/ASVspoof2017_V2_train.trn.txt'  # Path to your protocol file
    Train_AudioPath = './asvspoof2017/ASVspoof2017_V2_' + train_subset + '/'

    Val_protocol_file = './asvspoof2017/protocol_V2/ASVspoof2017_V2_dev.trl.txt'  # Path to your protocol file
    Val_AudioPath = './asvspoof2017/ASVspoof2017_V2_' + val_subset + '/'
    eval_subset = 'eval'
    
    Eval_protocol_file = './asvspoof2017/protocol_V2/ASVspoof2017_V2_eval.trl.txt'  # Path to your protocol file
    Eval_AudioPath = './asvspoof2017/ASVspoof2017_V2_' + eval_subset + '/'
 
    # Create the transform
    transform = SpectrogramTransform(n_fft=512, win_length=480, hop_length=160)

    # Create the dataset
    TrainDataset = AudioDataset(protocol_file=Train_protocol_file, AudioPath=Train_AudioPath, transform=transform, fixed_num_frames=fixed_num_frames)

    # Create the dataloader
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)

    # Create the dataset
    DevDataset = AudioDataset(protocol_file=Val_protocol_file, AudioPath=Val_AudioPath, transform=transform, fixed_num_frames=fixed_num_frames)

    # Create the dataloader
    DevDataLoader = DataLoader(DevDataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Checkpoint path
    chkpt_path = 'lcnn_spectrogram_checkpoint.pt'
    # Train the model
    train_model(model, TrainDataLoader, DevDataLoader, criterion, optimizer, num_epochs=num_epochs, device=device, checkpoint_path=chkpt_path)
    
    # Load the best model weigths
    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint)
    
    # Save output probabilities of the validation dataset
    save_output_probabilities(model, DevDataLoader, output_file='lcnn_dev_scores.txt', device='cuda')

    EvalDataset = AudioDataset(protocol_file=Eval_protocol_file , AudioPath=Eval_AudioPath, transform=transform, fixed_num_frames=fixed_num_frames)
    EvalDataLoader = DataLoader(EvalDataset, batch_size=batch_size, shuffle = False)
    
    # Save output probabilities of the validation dataset
    save_output_probabilities(model, EvalDataLoader, output_file='lcnn_eval_scores.txt', device='cuda')

if __name__ == '__main__':
    main()
