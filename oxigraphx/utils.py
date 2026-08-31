import os
import os.path as osp
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math
import torch
from torch_geometric.utils import degree
from scipy.stats import gaussian_kde
from numpy.polynomial.polynomial import polyfit
import ast
import re
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
from matplotlib import cm
from sklearn.metrics import r2_score

###############################################
def worker_init_fn(worker_id):
    #random.seed(seed + worker_id)
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(np.random.get_state()[1][0] + worker_id)

# convert list in string to list
def get_list(list_str):
   output_list = ast.literal_eval(list_str)
   return output_list   


# Need to find out what this function does
def generate_deg(dataset):
    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        # find the max degree in the whole dateset
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


# Save regression result to a file
def save_result(folder, filename, test_out, test_y):
   """
   test_out: predicted values
   test_y: truth values

   """
   result_file = osp.join(folder, filename)
   result = torch.cat((test_out, test_y), dim=1)
   result = result.cpu().numpy()
   with open(result_file, 'w') as file:
      for row in result:
         # Convert each row of the NumPy array to a space-separated string
         row_str = ' '.join(str(element) for element in row)
         file.write(row_str + '\n')
      file.close()

# Save the test loss. 
def save_mae(folder, maeFile, test_loss):
   # Save test_loss to maeFile
   result_file = osp.join(folder, maeFile)
   test_loss = test_loss
   if osp.exists(result_file):
      with open(result_file, "a") as file:
         file.write(str(test_loss) + '\n')
         file.close()
   else:
      with open(result_file, "w") as file:
         file.write(str(test_loss) + '\n')
         file.close()      

# Save the training progress info to a file
def save_train_progress(folder, filename, epochs, train_losses, val_losses, test_losses):
   # epochs, train_losses, val_losses, test_losses ALL are lists of numbers

   result_file = osp.join(folder, filename)
   with open(result_file, 'w') as file:
      row_str = " ".join(map(str, epochs))
      file.write(row_str + "\n")
      row_str = " ".join(map(str, train_losses))
      file.write(row_str + "\n")
      row_str = " ".join(map(str, val_losses))
      file.write(row_str + "\n")
      row_str = " ".join(map(str, test_losses))
      file.write(row_str + "\n")
      file.close()


# Get the prefix from the training_progress filename
def extract_prefix(filename):
   match = re.match(r'^([^_]+_[^_]+_[^_]+_[^_]+)_.+', filename)
   if match:
       return match.group(1)
   else:
      return None



# Plot training progress from saved progress file
def plot_training_from_file(folder, filename, plotfilename=None, disp=True, disp_step=1):

   prefix = extract_prefix(filename)

   """
   train_sets = ''
   test_sets = ''
   match = re.search(r'\d+_\d+', filename)
   if match:
      datasets = match.group()
   datasets = datasets.split('_')
   train_sets = datasets[0]
   test_sets = datasets[1]
   title = prefix + ', Training Sets: ' + train_sets + ' Testing Sets: ' + test_sets
   """

   result_file = osp.join(folder, filename)
   matrix = []
   with open(result_file, 'r') as file:
      for line in file:
         numbers = list(map(float, line.split()))
         matrix.append(numbers)
      file.close()

   epochs = list(map(int, matrix[0]))
   train_losses = matrix[1]
   val_losses = matrix[2]
   test_losses = matrix[3]
   maxepoch = max(epochs)

   threshold = 1
   for i in range(len(epochs)):
      if train_losses[i]>threshold:
         train_losses[i] = threshold
      if val_losses[i]>threshold:
         val_losses[i] = threshold
      if test_losses[i]>threshold:
         test_losses[i] = threshold

   # Prepare plotting
   plt.figure(figsize=(8, 6))
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Loss vs. Epoch during Training')
   plt.grid(True)
   lw = 0.7
   ms = 0.7

   x = range(disp_step, maxepoch+1, disp_step)
   plt.plot(x, train_losses, marker='o', linestyle='-', color='b', lw=lw, ms=ms)
   plt.plot(x, val_losses, marker='s', linestyle='-', color='g', lw=lw, ms=ms)
   # plt.plot(range(10, maxepoch+1, 10), test_losses, marker='*', linestyle='-', color='r', lw=lw, ms=ms)
   legend_entries = [Line2D([0], [0], color='blue', label='train_loss (Blue)'),
                     Line2D([0], [0], color='green', label='validate_loss (Green)')]
                    # Line2D([0], [0], color='green', label='test_loss (Red)')
   plt.title("Training Progress")
   plt.legend(handles=legend_entries, loc='upper right')

   if disp:
      plt.show()

   if plotfilename != None:
      plotfilename = osp.join(folder, plotfilename)
      plt.savefig(plotfilename)





# Plot the training progress
def plot_training_progress(epoch, train_losses, val_losses, test_losses, title, disp_step=1):
   lw = 0.7
   ms = 0.7

   threshold = 1
   for i in range(len(train_losses)):
      if train_losses[i]>threshold:
         train_losses[i] = threshold
      if val_losses[i]>threshold:
         val_losses[i] = threshold
      if test_losses[i]>threshold:
         test_losses[i] = threshold

   x = range(disp_step, epoch+1, disp_step)
   plt.plot(x, train_losses, marker='o', linestyle='-', color='b', lw=lw, ms=ms)
   plt.plot(x, val_losses, marker='s', linestyle='-', color='r', lw=lw, ms=ms)
   plt.plot(x, test_losses, marker='*', linestyle='-', color='g', lw=lw, ms=ms)

   legend_entries = [Line2D([0], [0], color='blue', label='train_loss (Blue)'),
                     Line2D([0], [0], color='red', label='validate_loss (Red)'),
                     Line2D([0], [0], color='green', label='test_loss (Green)')]
   plt.title(title)

   plt.legend(handles=legend_entries, loc='upper right')
   plt.pause(0.001)

# Find the most unique vector of aggregator weight in a group of them
def get_most_unique_agg_weights(vectors):
   # vectors is a list of aggregator weight vectors in nparray form
   max_distance_sum = -1
   most_different_vector = None
   most_different_index = -1

   for i, vector in enumerate(vectors):
       distance_sum = sum(np.sqrt(np.sum((vector - other_vector)**2)) for j, other_vector in enumerate(vectors) if i != j)
        
       if distance_sum > max_distance_sum:
          max_distance_sum = distance_sum
          most_different_vector = vector
          most_different_index = i
   return most_different_vector, most_different_index
   
   
# get and print the aggregator weights of CEAL layers
def get_print_agg_weights(model):
   for name, param in model.named_parameters():
      #if 'agg_weights' in name :
      x = param.data.clone()
      print(f"Parameter name: {name}, Values: {x}")
      #print(f"Parameter name: {name}, Values sftmx: {torch.nn.functional.softmax(ceal_1_weight, dim=-1)}") 

# Save aggregator weights of trained model
def save_agg_weights(folder, filename, model):
   result_file = osp.join(folder, filename)
   with open(result_file, 'w') as file:
      for name, param in model.named_parameters():
         #if 'agg_weights' in name or 'aggregator_weights' in name:
         x = param.data.clone()
         x = torch.nn.functional.softmax(x)
         x = x.cpu().numpy()
         msg = 'Parameter name: ' + name + ', '
         file.write(f"{msg}{x}" + '\n')
            
      file.close()


# Load regression results
def load_regression_results(folder, filename):

   filename = osp.join(folder, filename)

   with open(filename, "r") as f:
      data = f.readlines()
      f.close()
   data = [line.split() for line in data]
   data = np.asarray(data, dtype=float)
   data = torch.tensor(data, dtype=torch.float32)
   y = data[:,0] # predicted
   x = data[:,1]
   return y, x


# Calculate the point density
def get_density(x, y):
    """Get kernal density estimate for each (x, y) point."""
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    density = kernel(values)
    return density


# Plot the regression result
def plot_regression(folder, filename):

    result_file = osp.join(folder, filename)
    if osp.exists(result_file):
            with open(result_file, "r") as f:
                data = f.readlines()
                
    data = [line.split() for line in data]
    train_pred_list = [l[0] for l in data]
    train_true_list = [l[1] for l in data]
    train_true_list=np.asarray(train_true_list, dtype = float)
    train_pred_list = np.asarray(train_pred_list, dtype = float)
    
    r2 = r2_score(train_true_list, train_pred_list)
    
    error = []
    for i in range(len(train_true_list)):
        dif=abs(train_true_list[i]-train_pred_list[i])
        error.append(dif)
    loss=np.mean(error)

    max_true=5 #max(train_true_list)
    min_true=-5 #min(train_true_list)
    fig,ax=plt.subplots(figsize=(14,9))
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': '18'}
    #fit = np.polyfit(train_true_list, train_pred_list, 1)
    #line_fn = np.poly1d(fit)
    #y_line = train_true_list
    xy=np.vstack([train_true_list,train_pred_list])
    z=gaussian_kde(xy)(xy)
    idx=z.argsort()
    train_true_list=train_true_list[idx]
    train_pred_list=train_pred_list[idx]
    z=z[idx]
    Colors=['lightsteelblue','b','crimson']
    Cmap=LinearSegmentedColormap.from_list('mycmap',Colors)
    sm=ax.scatter(train_true_list, train_pred_list,c=z,s=40,cmap=Cmap)
    #ax.scatter(train_true_H, train_pred_H,s=40,c='b',label='H')
    # v=[0,10,20,30,40,50,60,70,80,90,100]
    # print(v)
    cm_1=plt.colorbar(sm)
    # cm_1.set_ticks([0, 2, 4, 6, 8, 10])
    cm_1.ax.tick_params(labelsize=20)
    #ax.scatter(train_true_list_B, train_pred_list_B,s=40, label='BETA')
    #plt.plot(train_true_list, y_line, linewidth=1, c="black", linestyle="--")
    ax.plot((0,1),(0,1),transform=ax.transAxes,linestyle='--',linewidth=1,c='black')
    #x_major_locator=MultipleLocator(0.5)
    #y_major_locator=MultipleLocator(0.5)
    #x_minor_locator=MultipleLocator(0.1)
    #y_minor_locator=MultipleLocator(0.1)
    ax1=plt.gca()
    #ax1.xaxis.set_major_locator(x_major_locator)
    #ax1.yaxis.set_major_locator(y_major_locator)
    #ax1.xaxis.set_minor_locator(x_minor_locator)
    #ax1.yaxis.set_minor_locator(y_minor_locator)
    ax.set_xlim(min_true-(max_true-min_true)/10, max_true+(max_true-min_true)/10)
    ax.set_ylim(min_true-(max_true-min_true)/10, max_true+(max_true-min_true)/10)
    ax.set_xlabel("True Oxygen Vacancy Formation Energy(eV)", font)
    ax.set_ylabel("Predicted Oxygen Vacancy Formation Energy(eV)", font)
    ax.tick_params(axis='x', which='major', direction='in', labelsize=20,length=10)
    ax.tick_params(axis='y', which='major', direction='in', labelsize=20,length=10)
    ax.tick_params(axis='x', which='minor', direction='in', labelsize=20,length=5)
    ax.tick_params(axis='y', which='minor', direction='in', labelsize=20,length=5)
    #plt.legend({"LOSS:{0:.4f}".format(np.mean(loss))},loc= 'upper left',  prop = font,markerscale=2,frameon=False)
    # plt.title('Y_True vs. Y_Pred Regression', fontsize=30)
    plt.suptitle("R^2={0:.4f},MAE:{1:.4f}(eV)".format(r2, loss),fontsize=25,x=0.5, y=0.2)
    #plt.suptitle("Regression: R^2={0:.4f}".format(r_train ** 2.0), fontsize=25,x=0.6, y=0.2)
    #plt.title('global/ceal', size=25, x=0.2, y=0.9)
    #plt.suptitle('testing set', size=25, x=0.75, y=0.2)
    #plt.legend(fontsize=25,frameon=False,loc='center right',bbox_to_anchor=(0.4,0.8))
    plot_file_name = osp.join(folder, filename+'.png')
    plt.savefig(plot_file_name)




# R^2 of predicted vs true values
def calculate_r_squared(x, y):
    # x: predicted values
    # y: true values

    x = list(x)
    y = list(y)

    n = len(y)

    # Calculate the mean of the true values (y_bar)
    y_bar = sum(y) / n

    # Calculate the total sum of squares (TSS)
    tss = sum((yi - y_bar) ** 2 for yi in y)

    # Calculate the residual sum of squares (RSS)
    rss = sum((xi - yi) ** 2 for xi, yi in zip(x, y))

    # Calculate R^2
    r_squared = 1 - (rss / tss)
    
    return r_squared


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


# Prepare aggregator weight list for batch data
def prep_ws(w, batch):
   """
   This function expands graph-wise weights to match batch data
   w: graph-wise weights
   batch: a list of index to indicate which graph nodes belong to

   """

   # Find the maximum value
   max_value = batch.max().item()

   N = []
   for i in range(max_value+1):
      seg_len = (batch == i).sum().item()
      N.append(seg_len)

   result = torch.tensor([]).to(w.device)
   for i in range(len(N)):
      repeated_value = w[i].repeat(N[i])
      result = torch.cat((result, repeated_value))

   w_size = w.size()
   result = result.view(len(batch), w_size[1])
   return result




#################################################



































































































