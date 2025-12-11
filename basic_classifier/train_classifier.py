
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn as nn
from src.basic_classifier.training import train_epoch, evaluate, save_best_model



def train_basic_classifier(model, train_loader, val_loader, training_config, optimizer_config):

    NUM_EPOCHS = training_config["num_epochs"]
    PATIENCE = training_config["patience"]
    DEVICE = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay']
    )

    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=training_config['warmup_epochs']
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=training_config['num_epochs'] - training_config['warmup_epochs']
    )

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[training_config['warmup_epochs']]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=training_config['label_smoothing'])

    # Best Top5 Accuracy
    best_val_acc = 0
    last_improved = 0
    best_model_state = None
    epochs_without_improvement = 0
        
    history = {
        'train': [],
        'val': []
    }

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
        
    # Training loop
    for epoch in range(NUM_EPOCHS):
        
        loss_weights = model.get_dynamic_loss_weights(epoch, NUM_EPOCHS)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, loss_weights)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE, loss_weights)

        # Learning rate scheduling
        scheduler.step()
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Weight- {loss_weights}")
        print(f"  Train - Loss: {train_metrics['loss_avg']:.4f}, Rank Loss: {train_metrics['loss_rank_avg']}")
        print(f"          Top-1: {train_metrics['top1_acc']:.4f}, Top-5: {train_metrics['top5_acc']:.4f}")
        print(f"          Rank Top-1: {train_metrics['top1_rank_acc']}, Rank Top-5: {train_metrics['top5_rank_acc']}")
        print(f"  Val   - Loss: {val_metrics['loss_avg']:.4f}, Rank Loss: {val_metrics['loss_rank_avg']}")
        print(f"          Top-1: {val_metrics['top1_acc']:.4f}, Top-5: {val_metrics['top5_acc']:.4f}")
        print(f"          Rank Top-1: {val_metrics['top1_rank_acc']}, Rank Top-5: {val_metrics['top5_rank_acc']}")
        
        best_val_acc, best_model_state, improved = save_best_model(
            val_metrics["top5_acc"], best_val_acc, model, best_model_state
        )

        if improved:
            last_improved = epoch
            epochs_without_improvement = 0 
        else:
            epochs_without_improvement += 1
            

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    return best_val_acc, best_model_state, history, last_improved