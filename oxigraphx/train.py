import matplotlib.pyplot as plt
import time

import os.path as osp
from sklearn.metrics import pairwise_distances
from model.atomsight_ceal import *
from utils import *


# Define ONE training epoch
def train_step(model, train_data_loader, optimizer, device):
   # Step 1: set model to training mode
   model.train()
   # Step 2: Iterate through batches returned by the train_data_loader
   total_loss = 0.0
   for batch_i, data in enumerate(train_data_loader):
      # Step 3: put data to device. eg. 'cuda'
      data = data.to(device)
      # Step 4: zero out gradient
      optimizer.zero_grad()
      # Run data through model
      out = model(data)

      if torch.isnan(data.x).any():
          raise ValueError(f"NaN in input features at batch {batch_i}")
      if torch.isnan(out).any():
          raise ValueError(f"NaN in model output at batch {batch_i}. Check model init or learning rate.")


      # Step 5: Calculate loss: Mean Absolute Error
      loss = (out.squeeze() - data.y).abs().mean()
      # Step 6: Back propogation
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


      # Step 7: adjust the parameters by the gradients collected in the backward pass
      optimizer.step()
      #Step 8: update total_loss
      total_loss += loss.item() * data.num_graphs

   # Step 8: calculate step loss
   loss = total_loss / len(train_data_loader.dataset)

   return model, loss

# Define the testing/validating step
@torch.no_grad()
def test(model, data_loader, device):
   # Step 1: Sett model to testing mode
   model.eval()
   total_error = 0

   """
   Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode.
   Also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
   """
   my_out = torch.empty(0).to(device)
   my_y = torch.empty(0).to(device)
   # my_f = torch.empty(0).to(device)

   with torch.no_grad():
      for i, data in enumerate(data_loader):
         #for data in data_loader:
         # Step 2: put data to device   
         data = data.to(device)
         # Step 3: run the data through model
         out = model(data)
         # Step 4: update error
         total_error += (out.squeeze() - data.y).abs().sum().item()
       
         my_out = torch.cat((my_out, out), 0)
         my_y = torch.cat((my_y, data.y), 0)

   # Step: 5: calculate mean step testing loss
   loss = total_error / len(data_loader.dataset)
   return loss, my_out, my_y


# Define the training process
def train(epochs, model, train_data_loader, val_data_loader, test_data_loader, optimizer, scheduler, device, disp_step, batch_size, learning_rate, numLayers, target_scaler=None, dispProgress=True, model_config=None, feature_names=None):
   if dispProgress:
      # Prepare plotting
      plt.figure(figsize=(8, 6))
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.title('Loss vs. Epoch during Training')
      plt.grid(True)

   train_losses = []
   val_losses = []
   test_losses = []
   epoch_loss_list = []
   epochs_list = []

   prefix = 'batch_size:'+ str(batch_size)+'learning_rate:'+str(learning_rate)+'num layers:'+str(numLayers)
   for epoch in range(1, epochs+1):
      model, train_loss = train_step(model, train_data_loader, optimizer, device)
      val_loss, val_out, val_y = test(model, val_data_loader, device)
      test_loss, test_out, test_y= test(model, test_data_loader, device)
      # Use val_loss to scheduler

      scheduler.step(val_loss) # This is for the ReduceLROnPlateau scheduler
      #scheduler.step()  # This is for the annealing scheduler
      epoch_loss_list.append(train_loss)

      # Display training progress
      if epoch % disp_step == 0:
         current_lr = optimizer.param_groups[0]['lr']
         progress_msg = 'Epoch '+ str(epoch) + \
                        ', training loss(MAE)=' + str(round(train_loss,4)) + \
                        ', Validating loss(MAE)=' + str(round(val_loss,4)) +  \
                        ', testing loss(MAE)=' + str(round(test_loss,4)) + \
                        ', lr=' + str(round(current_lr,8))
         print(progress_msg)
         train_losses.append(train_loss)
         val_losses.append(val_loss)
         test_losses.append(test_loss)

         epochs_list.append(epoch) 
         if dispProgress:
            plot_title = prefix 
            plot_training_progress(epoch, train_losses, val_losses, test_losses, title=plot_title)

   
   # Test the model after it's trained
   test_loss, test_out, test_y = test(model, test_data_loader, device)

    # Initialize variables for saving - handle both normalized and unnormalized cases
   test_out_to_save = test_out
   test_y_to_save = test_y

   if target_scaler is not None and 'y_scaler' in target_scaler:
       print(f"\nInverse transforming predictions to original scale...")
       y_scaler = target_scaler['y_scaler']
    
       # Convert to numpy, inverse transform, convert back to tensor
       test_out_original = y_scaler.inverse_transform(test_out.cpu().numpy().reshape(-1, 1))
       test_y_original = y_scaler.inverse_transform(test_y.cpu().numpy().reshape(-1, 1))
    
       test_out_original = torch.tensor(test_out_original, dtype=torch.float32)
       test_y_original = torch.tensor(test_y_original, dtype=torch.float32)
    
       print("Sample predictions (original scale):")
       print(test_out_original[:5])
       print("Sample targets (original scale):")
       print(test_y_original[:5])
    
       # Recalculate MAE on original scale
       test_loss = (test_out_original - test_y_original).abs().mean().item()
       print(f"Test MAE on original scale: {test_loss:.4f}")
    
       # Use original scale for saving
       test_out_to_save = test_out_original
       test_y_to_save = test_y_original
   else:
       # No normalization - reshape test_out to match test_y dimensions
       print(f"\nNo normalization applied.")
       print(f"test_out shape: {test_out.shape}")
       print(f"test_y shape: {test_y.shape}")
    
       # Ensure both have shape [N, 1] for consistency
       if test_out.dim() == 1:
           test_out_to_save = test_out.unsqueeze(1)
       else:
            test_out_to_save = test_out
        
       if test_y.dim() == 1:
           test_y_to_save = test_y.unsqueeze(1)
       else:
           test_y_to_save = test_y
    
       print(f"Reshaped test_out shape: {test_out_to_save.shape}")
       print(f"Reshaped test_y shape: {test_y_to_save.shape}")
       print(f"Test MAE: {test_loss:.4f}")

    # Save training results and information for later processing
   timestr = str(int(time.time()*1000000))
   folder = './Results/8/21'
   create_folder_if_not_exists(folder)

   maeFile = prefix + '_MAE.txt'
   save_mae(folder=folder, maeFile=maeFile, test_loss=str(test_loss) + '_' + timestr)

   progress_filename = prefix + '_' + timestr + '_training_progress.txt'
   save_train_progress(folder, progress_filename, epochs=epochs_list, train_losses=train_losses, val_losses=val_losses, test_losses=test_losses)

   plotfilename = prefix + '_' + timestr + '_progress_plotting.png'
   plot_training_from_file(folder, filename=progress_filename, plotfilename=plotfilename, disp=False, disp_step=disp_step)

   agg_weight_filename = prefix + '_' + timestr + '_agg_weights.txt'
   save_agg_weights(folder, filename=agg_weight_filename, model=model)

   result_file = prefix + '_' + timestr + '_y.txt'
   # Use the appropriately shaped tensors for saving
   save_result(folder, result_file, test_out=test_out_to_save, test_y=test_y_to_save)

   plot_regression(folder, result_file)

   model_file = prefix + '_' + timestr +'.pth'
   model_path = osp.join(folder, model_file)
   checkpoint = {
    'model_state_dict': model.state_dict(),
    'target_scaler': target_scaler,   # x_scaler / y_scaler
    'model_config': model_config     # architecture
   }

   torch.save(checkpoint, model_path)

   # ── Feature importance ranking ────────────────────────────────────────────
   if feature_names is not None:
      ranker = FeatureImportanceRanker(model=model, feature_names=feature_names)
      report = ranker.rank()

      print('\nTop-10 most important atomic features:')
      print(report.head(10).to_string(index=False))

      importance_csv = osp.join(folder, prefix + '_' + timestr + '_feature_importance.csv')
      ranker.save_report(importance_csv)

      importance_png = osp.join(folder, prefix + '_' + timestr + '_feature_importance.png')
      ranker.plot(
         report,
         top_k=20,
         title='AtomSight-GNN: Learned Atomic Feature Importance',
         save_path=importance_png,
      )
      print(f'Feature importance saved to {importance_csv}')

   return model, timestr