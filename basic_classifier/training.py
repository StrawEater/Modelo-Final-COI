from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def translate_label(dict_label):
    # Extract values preserving the rank order
  
  ranks = list(dict_label.keys())
  # Stack tensors directly
  tensor = torch.stack([dict_label[r] for r in ranks], dim=0).T
  return tensor


def do_criterion_by_rank( predictions, labels, criterion, device):
  
  loss = torch.tensor(0.0, device=device)
  loss_by_rank = np.zeros(len(predictions))

  for idx, rank_prediction in enumerate(predictions):
      
    rank_labels = labels[:, idx]
    loss_rank = criterion(rank_prediction, rank_labels) 
    loss_by_rank[idx] = loss_rank.item()
    
    loss += loss_rank
  
  return loss, loss_by_rank

def train_epoch(model, dataloader, criterion, optimizer, device):
  """Train for one epoch"""
  
  model.train()
  
  total = 0

  top1_correct = 0
  top5_correct = 0
  
  top1_by_rank = np.zeros(model.get_ranks())
  top5_by_rank = np.zeros(model.get_ranks())

  total_loss = 0
  total_loss_by_rank = np.zeros(model.get_ranks())

  pbar = tqdm(dataloader, desc="Training")
  for sequences, labels in pbar:
    
    labels = translate_label(labels).to(device)
    
    optimizer.zero_grad()

    outputs = model(list(sequences))

    loss, loss_by_rank = do_criterion_by_rank(outputs, labels, criterion, device)
    
    total_loss += loss.item()
    total_loss_by_rank += loss_by_rank

    loss.backward()
    optimizer.step()
    
    for idx, out in enumerate(outputs):
      batch_size = out.size(0)
      total += batch_size
      
      # Top-1
      preds = torch.argmax(out, dim=1)
      correctas_top1 = (preds == labels[:, idx]).sum().item() 
      top1_correct += correctas_top1 
      top1_by_rank[idx] += correctas_top1

      # Top-5
      top5_preds = torch.topk(out, k=5, dim=1).indices
      correctas_top5 = (top5_preds == labels[:, idx].unsqueeze(1)).any(dim=1).sum().item() 
      top5_correct += correctas_top5
      top5_by_rank[idx] += correctas_top5
    

  top1_acc = top1_correct / total
  top5_acc = top5_correct / total
  
  top1_rank_acc = top1_by_rank / (total // model.get_ranks())
  top5_rank_acc = top5_by_rank / (total // model.get_ranks())
  
  loss_avg = total_loss / len(dataloader)
  loss_rank_avg = total_loss_by_rank / len(dataloader)

  results = {
              "top1_acc": top1_acc,
              "top5_acc": top5_acc,
              "top1_rank_acc": top1_rank_acc,
              "top5_rank_acc": top5_rank_acc,
              "loss_avg": loss_avg,
              "loss_rank_avg": loss_rank_avg
            }

  return results

def evaluate(model, dataloader, criterion, device):
  """Evaluate model"""
  model.eval()
  
  total = 0

  top1_correct = 0
  top5_correct = 0
  
  top1_by_rank = np.zeros(model.get_ranks())
  top5_by_rank = np.zeros(model.get_ranks())

  total_loss = 0
  total_loss_by_rank = np.zeros(model.get_ranks())
  
  with torch.no_grad():
    
    pbar = tqdm(dataloader, desc="Evaluation")
    
    for inputs, labels in pbar:
    
      labels = translate_label(labels).to(device)
      outputs = model(inputs)
      
      loss, loss_by_rank = do_criterion_by_rank(outputs, labels, criterion, device)
    
      total_loss += loss.item()
      total_loss_by_rank += loss_by_rank
    
      for idx, out in enumerate(outputs):
        batch_size = out.size(0)
        total += batch_size
        
        # Top-1
        preds = torch.argmax(out, dim=1)
        correctas_top1 = (preds == labels[:, idx]).sum().item() 
        top1_correct += correctas_top1 
        top1_by_rank[idx] += correctas_top1

        # Top-5
        top5_preds = torch.topk(out, k=5, dim=1).indices
        correctas_top5 = (top5_preds == labels[:, idx].unsqueeze(1)).any(dim=1).sum().item() 
        top5_correct += correctas_top5
        top5_by_rank[idx] += correctas_top5
    

  top1_acc = top1_correct / total
  top5_acc = top5_correct / total
  
  top1_rank_acc = top1_by_rank / (total // model.get_ranks())
  top5_rank_acc = top5_by_rank / (total // model.get_ranks())
  
  loss_avg = total_loss / len(dataloader)
  loss_rank_avg = total_loss_by_rank / len(dataloader)

  results = {
              "top1_acc": top1_acc,
              "top5_acc": top5_acc,
              "top1_rank_acc": top1_rank_acc,
              "top5_rank_acc": top5_rank_acc,
              "loss_avg": loss_avg,
              "loss_rank_avg": loss_rank_avg
            }

  return results

def save_best_model(accuracy, best_val_acc, model, best_model):
  if accuracy > best_val_acc:
    return accuracy, model.state_dict().copy(), True 
  else:
    return best_val_acc, best_model, False 