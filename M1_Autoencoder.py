"""
Neural Networks - Deep Learning
Autoencoder (predict face behind mask)
Author: Dimitrios Spanos Email: dimitrioss@ece.auth.gr
"""

from M1 import M1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import time, os, warnings
import numpy as np

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "resized_dataset/dataset/"
img_size = 256
mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
name_of_model = 'M1_MSE_lr_1e-3'
model_file = "autoencoders/"+name_of_model+".pt"
learning_rate = 1e-3
p = 0.2
batch_size = 32
epochs = 100
batches = 375 # batches * batch_size = training_size, with batches=375 the whole dataset is iterated

def main():

    # Creation of Data Loaders
    trn_no_mask, trn_mask, tst_no_mask, tst_mask  = getData()

    print(f"Shape of trn_no_mask: ({len(trn_no_mask.dataset)}, {trn_no_mask.dataset[0][0].shape})")
    print(f"Shape of trn_mask:    ({len(trn_mask.dataset)}, {trn_mask.dataset[0][0].shape})")
    print(f"Shape of tst_no_mask: ({len(tst_no_mask.dataset)}, {tst_no_mask.dataset[0][0].shape})")
    print(f"Shape of tst_mask:    ({len(tst_mask.dataset)}, {tst_mask.dataset[0][0].shape})\n")

    # Creation of the model
    model = M1(dropout_probability = p)

    # Train the model
    trained_model,train_losses = train(model, trn_no_mask, trn_mask)

    # Visualize 5 examples of training set
    visualize_results(trained_model, trn_no_mask, trn_mask, name_of_model+'_train_examples')

    # Test the model on 4000 images and find the average loss
    test_loss = test(trained_model, tst_no_mask, tst_mask)

    # Draw the learning curve along with the test loss
    draw_learning_curve(train_losses, test_loss)

    # Visualize 5 examples of testing set
    visualize_results(trained_model, tst_no_mask, tst_mask, name_of_model+'_test_examples')


def train(model, no_mask_loader, mask_loader):
    """
    :param model: The untrained model
    :param no_mask_loader: The train loader containing the faces without mask
    :param mask_loader: The train loader containing the faces with mask
    :return: The trained model
    """
    if device == torch.device("cuda"):
        model.cuda()
    model.train()
    criterion = nn.MSELoss()
    if device == torch.device("cuda"):
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    start_time = time.time()
    losses = []
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    # Training
    for epoch in range(epochs):

        epoch_start = time.time()
        no_mask_iter = iter(no_mask_loader)
        b,mean_loss = 0,0
        for batch, (mask_train, _) in enumerate(mask_loader):
            if batch == batches: # 200 * 16 = 3200 images for training
                break
            b += 1

            try:
                (no_mask_train,_) = next(no_mask_iter)
            except StopIteration:
                no_mask_iter = iter(no_mask_iter)
                (no_mask_train,_) = next(no_mask_iter)

            mask_train,no_mask_train = mask_train.to(device),no_mask_train.to(device)
            prediction = model(mask_train)

            loss = criterion(prediction, no_mask_train)
            mean_loss += (1/b) * (loss.item() - mean_loss)

            # Update Parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        losses.append(mean_loss)
        # Results
        print(f"Epoch {(epoch+1):02}/{epochs} => {int(time.time()-epoch_start)}s - train_loss: {mean_loss:.5f}")

    torch.save(model.state_dict(), model_file)
    print(f"\nFinal training took {((time.time() - start_time) / 60):.2f} minutes.")
    return model, losses


def visualize_results(trained_model, trn_no_mask, trn_mask,name):
    
    unnormalize= transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    for i in range(5):
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title("No Mask", fontsize=16)
        no_mask_image = unnormalize(trn_no_mask.dataset[i][0]).permute(1,2,0).detach().numpy()
        plt.imshow((no_mask_image * 255).astype(np.uint8))

        plt.subplot(1, 3, 2)
        plt.title("Mask", fontsize=16)
        mask_image = unnormalize(trn_mask.dataset[i][0]).permute(1,2,0).detach().numpy()
        plt.imshow((mask_image * 255).astype(np.uint8))

        plt.subplot(1, 3, 3)
        plt.title("Prediction", fontsize=16)
        prediction = unnormalize(trained_model(trn_mask.dataset[i][0].reshape(1,3,img_size,img_size).cuda()))
        prediction = (prediction.reshape(3,img_size,img_size)).permute(1,2,0)
        prediction = prediction.cpu().detach().numpy()
        plt.imshow(prediction)

        plt.savefig('examples/' + name + f'_{i+1}.png', dpi=400, bbox_inches='tight')
        plt.close()


def draw_learning_curve(train_losses, test_loss):
    # just so that I can visualize test in respect to train loss
    test_losses = list(np.full(epochs, test_loss))

    plt.plot(train_losses, color='b')
    plt.plot(test_losses, color='r')
    plt.legend(['training losss', 'testing loss'], loc='upper right')
    plt.title('Learning Curve - MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('Learning_Curve_' +name_of_model+'.png', dpi=400, bbox_inches='tight')
    plt.close()


def test(trained_model, no_mask_loader, mask_loader):
    """
    :param trained_model: The trained model
    :param no_mask_loader: The test data without mask (y)
    :param mask_loader: The test data with mask (X)
    """

    if device == torch.device("cuda"):
        trained_model.cuda()
    trained_model.eval()
    criterion = nn.MSELoss()
    if device == torch.device("cuda"):
        criterion = criterion.cuda()
    start_time = time.time()
    test_loss = 0
    with torch.no_grad():
        no_mask_iter = iter(no_mask_loader)
        for batch, (mask_test, _) in enumerate(mask_loader):
            batch+=1
            try:
                (no_mask_test, _) = next(no_mask_iter)
            except StopIteration:
                no_mask_iter = iter(no_mask_iter)
                (no_mask_test, _) = next(no_mask_iter)

            mask_test, no_mask_test = mask_test.to(device=device), no_mask_test.to(device=device)
            prediction = trained_model(mask_test)
            loss = criterion(prediction, no_mask_test)
            test_loss += (1/batch) * (loss.item()-test_loss)

    total = time.time() - start_time
    # 4000 images on testing dataset
    print(f"Testing took {total:.5f}secs.\n{(total/4000*1000):.5f}ms per image on average.")

    return test_loss


def getData():
    """
    :return: The train, test loaders
    """
    tranforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    trn_without_mask_data = datasets.ImageFolder(os.path.join(root, "Training/without_mask"), transform=tranforms)
    trn_with_mask_data = datasets.ImageFolder(os.path.join(root, "Training/with_mask"), transform=tranforms)

    trn_without_mask = DataLoader(trn_without_mask_data, batch_size=batch_size, shuffle=False, num_workers=4)
    trn_with_mask = DataLoader(trn_with_mask_data, batch_size=batch_size, num_workers=4)

    tst_without_mask_data = datasets.ImageFolder(os.path.join(root, "Testing/without_mask"), transform=tranforms)
    tst_with_mask_data = datasets.ImageFolder(os.path.join(root, "Testing/with_mask"), transform=tranforms)

    tst_without_mask = DataLoader(tst_without_mask_data, batch_size=batch_size, shuffle=False, num_workers=4)
    tst_with_mask = DataLoader(tst_with_mask_data, batch_size=batch_size, num_workers=4)

    return trn_without_mask, trn_with_mask,tst_without_mask,tst_with_mask


if __name__ == '__main__':
    main()


