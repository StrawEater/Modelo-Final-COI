# runner.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

class ExperimentRunner:

    def __init__(self, model, device="cuda"):
        """
        model: CombinedModelWithDecoder multi-head (tiene .decoders, .classifiers, .global_decoder)
        """
        self.device = device
        self.model = model.to(device)

        # Detect multitask: requiere ModuleDicts 'decoders' and 'classifiers' and 'global_decoder'
        self.multi_task = hasattr(model, "decoders") and hasattr(model, "classifiers") and hasattr(model, "global_decoder")
        if self.multi_task:
            print("→ Modo MULTITASK ACTIVADO (clasificación + reconstrucción)")
        else:
            print("→ Modo LINEAR PROBE activado")

    # ---------------------------------------------------------
    def train_probe(self, train_loader, val_loader, num_epochs=10, lr=1e-3, train_encoder=False):
        """
        Entrena solo un probe (legacy). Asume self.model(seqs) -> logits (single head).
        """
        if train_encoder:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            print("Entrenando encoder + probe...")
        else:
            # asumimos que el modelo tiene .probe
            optimizer = optim.Adam(self.model.probe.parameters(), lr=lr)
            print("Entrenando solo el linear probe (encoder congelado)...")

        criterion = nn.CrossEntropyLoss()
        train_losses, val_losses, val_accuracies = [], [], []

        for epoch in range(num_epochs):
            self.model.train()
            running = 0.0
            for sequences, labels in train_loader:
                # sequences can be list of strings or token tensors; adapt según tu DNABERT wrapper
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(sequences)  # legacy single-head forward
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running += loss.item()
            train_losses.append(running / len(train_loader))

            # validation
            self.model.eval()
            val_running = 0.0
            preds, gts = [], []
            with torch.no_grad():
                for sequences, labels in val_loader:
                    labels = labels.to(self.device)
                    logits = self.model(sequences)
                    loss = criterion(logits, labels)
                    val_running += loss.item()
                    _, pred = torch.max(logits, 1)
                    preds.extend(pred.cpu().tolist())
                    gts.extend(labels.cpu().tolist())
            val_losses.append(val_running / len(val_loader))
            val_acc = accuracy_score(gts, preds)
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch+1}/{num_epochs} TrainLoss={train_losses[-1]:.4f} ValLoss={val_losses[-1]:.4f} ValAcc={val_acc:.4f}")

        return train_losses, val_losses, val_accuracies

    # ---------------------------------------------------------
    def train_multitask(self, train_loader, val_loader=None, num_epochs=20, 
                    alpha=1.0, beta=1.0, lr=1e-3, 
                    taxon_weights=None, mixed_precision=True,
                    early_stopping_patience=None, save_best=True):
        """
        Entrena el modelo multitarea con validación por época.
        
        Args:
            early_stopping_patience: Si no mejora en N épocas, para el entrenamiento
            save_best: Si True, guarda el mejor modelo según val_loss
        """
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evitar warning
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        
        if taxon_weights is None:
            taxon_weights = {taxon: 1.0 for taxon in self.model.classifiers.keys()}
        
        print(f"📊 Pesos por taxón: {taxon_weights}")
        print(f"🚀 Mixed Precision: {mixed_precision}")
        print(f"📦 Batches por época: {len(train_loader)}")
        
        # Historial expandido
        history = {
            "train_total": [], "train_cls": [], "train_rec": [], "train_global_rec": [],
            "val_total": [], "val_cls": [], "val_rec": [], "val_global_rec": [],
            "val_acc": {taxon: [] for taxon in self.model.classifiers.keys()},
            "val_f1": {taxon: [] for taxon in self.model.classifiers.keys()}
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # ==================== ENTRENAMIENTO ====================
            self.model.train()
            epoch_train_total = 0.0
            epoch_train_cls = 0.0
            epoch_train_rec = 0.0
            epoch_train_global = 0.0
            n_batches = 0
    
            pbar = tqdm(train_loader, 
                        desc=f"[TRAIN] Epoch {epoch+1}/{num_epochs}",
                        ncols=130,
                        mininterval=2.0)  # Actualizar cada 2s para evitar rate limit
            
            for batch in pbar:
                if len(batch) == 2:
                    seqs, labels = batch
                    labels_dict = {"label": labels.to(self.device)}
                    recon_targets_dict = None
                    true_tokens = None
                else:
                    seqs, labels_dict, recon_targets_dict, true_tokens = batch
                    true_tokens = true_tokens.to(self.device) if true_tokens is not None else None
                    for k in list(labels_dict.keys()):
                        labels_dict[k] = labels_dict[k].to(self.device)
    
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=mixed_precision):
                    out = self.model(seqs, use_hierarchy=False)
                    logits_dict = out["logits"]
                    recon_dict = out["recon"]
                    global_recon = out["global_recon"]
                    emb = out.get("emb", None)
    
                    total_loss = 0.0
                    sum_cls = 0.0
                    sum_rec = 0.0
    
                    for taxon, logits in logits_dict.items():
                        taxon_weight = taxon_weights.get(taxon, 1.0)
                        
                        if isinstance(labels_dict, dict) and taxon in labels_dict:
                            labels_t = labels_dict[taxon]
                        elif isinstance(labels_dict, dict) and len(labels_dict) == 1:
                            labels_t = next(iter(labels_dict.values()))
                        else:
                            raise ValueError(f"Labels for taxon '{taxon}' not found")
    
                        loss_cls = F.cross_entropy(logits.to(self.device), labels_t)
    
                        if isinstance(recon_targets_dict, dict) and recon_targets_dict is not None and taxon in recon_targets_dict and recon_targets_dict[taxon] is not None:
                            target_tokens = recon_targets_dict[taxon].to(self.device)
                            recon_logits = recon_dict[taxon].to(self.device)
                            loss_rec = F.cross_entropy(recon_logits.view(-1, recon_logits.size(-1)),
                                                       target_tokens.view(-1))
                        else:
                            if true_tokens is not None:
                                recon_logits = recon_dict[taxon].to(self.device)
                                loss_rec = F.cross_entropy(recon_logits.view(-1, recon_logits.size(-1)),
                                                           true_tokens.view(-1))
                            elif emb is not None:
                                recon_pred = recon_dict[taxon].to(self.device)
                                loss_rec = F.mse_loss(recon_pred, emb.to(self.device))
                            else:
                                raise ValueError(f"No reconstruction target for '{taxon}'")
    
                        total_loss = total_loss + taxon_weight * (alpha * loss_cls + beta * loss_rec)
                        sum_cls = sum_cls + loss_cls.item()
                        sum_rec = sum_rec + loss_rec.item()
    
                    if true_tokens is not None:
                        glob = global_recon.to(self.device)
                        loss_glob = F.cross_entropy(glob.view(-1, glob.size(-1)), true_tokens.view(-1))
                    elif emb is not None:
                        glob = global_recon.to(self.device)
                        loss_glob = F.mse_loss(glob, emb.to(self.device))
                    else:
                        raise ValueError("No global reconstruction target")
    
                    total_loss = total_loss + beta * loss_glob
    
                if mixed_precision:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
    
                epoch_train_total += total_loss.item()
                epoch_train_cls += sum_cls
                epoch_train_rec += sum_rec
                epoch_train_global += loss_glob.item()
                n_batches += 1
                
                # Actualizar cada 10 batches para evitar rate limit
                if n_batches % 10 == 0:
                    pbar.set_postfix({
                        'Loss': f'{total_loss.item():.2f}',
                        'CE': f'{sum_cls/len(logits_dict):.2f}',
                    })
    
            # Guardar métricas de entrenamiento
            history["train_total"].append(epoch_train_total / n_batches)
            history["train_cls"].append(epoch_train_cls / n_batches)
            history["train_rec"].append(epoch_train_rec / n_batches)
            history["train_global_rec"].append(epoch_train_global / n_batches)
    
            # ==================== VALIDACIÓN ====================
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, alpha, beta, taxon_weights)
                
                history["val_total"].append(val_metrics["total_loss"])
                history["val_cls"].append(val_metrics["cls_loss"])
                history["val_rec"].append(val_metrics["rec_loss"])
                history["val_global_rec"].append(val_metrics["global_loss"])
                
                for taxon in self.model.classifiers.keys():
                    history["val_acc"][taxon].append(val_metrics["accuracy"][taxon])
                    history["val_f1"][taxon].append(val_metrics["f1"][taxon])
                
                # Imprimir resumen
                print(f"\n{'='*100}")
                print(f"📊 EPOCH {epoch+1}/{num_epochs} SUMMARY:")
                print(f"{'='*100}")
                print(f"  TRAIN → Loss: {history['train_total'][-1]:.4f} | "
                      f"CE: {history['train_cls'][-1]:.4f} | "
                      f"Rec: {history['train_rec'][-1]:.4f} | "
                      f"GRec: {history['train_global_rec'][-1]:.4f}")
                print(f"  VAL   → Loss: {history['val_total'][-1]:.4f} | "
                      f"CE: {history['val_cls'][-1]:.4f} | "
                      f"Rec: {history['val_rec'][-1]:.4f} | "
                      f"GRec: {history['val_global_rec'][-1]:.4f}")
                print(f"\n  📈 VALIDATION ACCURACY & F1:")
                for taxon in self.model.classifiers.keys():
                    acc = history["val_acc"][taxon][-1]
                    f1 = history["val_f1"][taxon][-1]
                    print(f"     {taxon:10s} → Acc: {acc:6.2%} | F1: {f1:.4f}")
                print(f"{'='*100}\n")
                
                # Early stopping y guardar mejor modelo
                current_val_loss = val_metrics["total_loss"]
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    
                    if save_best:
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'history': history
                        }, 'best_model.pt')
                        print(f"💾 Mejor modelo guardado (val_loss={best_val_loss:.4f})\n")
                else:
                    patience_counter += 1
                    if early_stopping_patience and patience_counter >= early_stopping_patience:
                        print(f"\n🛑 Early stopping: no mejora en {early_stopping_patience} épocas")
                        break
            else:
                # Sin validación, solo mostrar train
                print(f"\n📊 Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {history['train_total'][-1]:.4f} | "
                      f"CE: {history['train_cls'][-1]:.4f}\n")
    
        return history

    def _validate_epoch(self, val_loader, alpha, beta, taxon_weights):
        """Valida el modelo en val_loader y retorna métricas completas"""
        self.model.eval()
        
        total_loss = 0.0
        total_cls = 0.0
        total_rec = 0.0
        total_global = 0.0
        n_batches = 0
        
        # Para calcular accuracy/F1
        predictions = {taxon: [] for taxon in self.model.classifiers.keys()}
        ground_truth = {taxon: [] for taxon in self.model.classifiers.keys()}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[VAL]  ", ncols=130, mininterval=2.0):
                if len(batch) == 2:
                    seqs, labels = batch
                    labels_dict = {"label": labels.to(self.device)}
                    recon_targets_dict = None
                    true_tokens = None
                else:
                    seqs, labels_dict, recon_targets_dict, true_tokens = batch
                    true_tokens = true_tokens.to(self.device) if true_tokens is not None else None
                    for k in list(labels_dict.keys()):
                        labels_dict[k] = labels_dict[k].to(self.device)
                
                out = self.model(seqs, use_hierarchy=False)
                logits_dict = out["logits"]
                recon_dict = out["recon"]
                global_recon = out["global_recon"]
                emb = out.get("emb", None)
                
                batch_loss = 0.0
                batch_cls = 0.0
                batch_rec = 0.0
                
                for taxon, logits in logits_dict.items():
                    taxon_weight = taxon_weights.get(taxon, 1.0) if taxon_weights else 1.0
                    
                    labels_t = labels_dict.get(taxon, next(iter(labels_dict.values())))
                    
                    loss_cls = F.cross_entropy(logits, labels_t)
                    
                    if isinstance(recon_targets_dict, dict) and recon_targets_dict is not None and taxon in recon_targets_dict:
                        target_tokens = recon_targets_dict[taxon].to(self.device)
                        recon_logits = recon_dict[taxon]
                        loss_rec = F.cross_entropy(recon_logits.view(-1, recon_logits.size(-1)),
                                                   target_tokens.view(-1))
                    elif true_tokens is not None:
                        recon_logits = recon_dict[taxon]
                        loss_rec = F.cross_entropy(recon_logits.view(-1, recon_logits.size(-1)),
                                                   true_tokens.view(-1))
                    else:
                        loss_rec = torch.tensor(0.0)
                    
                    batch_loss += taxon_weight * (alpha * loss_cls + beta * loss_rec)
                    batch_cls += loss_cls.item()
                    batch_rec += loss_rec.item()
                    
                    # Guardar predicciones para accuracy/F1
                    _, preds = torch.max(logits, 1)
                    predictions[taxon].extend(preds.cpu().numpy())
                    ground_truth[taxon].extend(labels_t.cpu().numpy())
                
                # Global reconstruction
                if true_tokens is not None:
                    loss_global = F.cross_entropy(global_recon.view(-1, global_recon.size(-1)),
                                                   true_tokens.view(-1))
                else:
                    loss_global = torch.tensor(0.0)
                
                batch_loss += beta * loss_global
                
                total_loss += batch_loss.item()
                total_cls += batch_cls
                total_rec += batch_rec
                total_global += loss_global.item()
                n_batches += 1
        
        # Calcular accuracy y F1
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = {}
        f1 = {}
        for taxon in self.model.classifiers.keys():
            accuracy[taxon] = accuracy_score(ground_truth[taxon], predictions[taxon])
            f1[taxon] = f1_score(ground_truth[taxon], predictions[taxon], average='weighted')
        
        self.model.train()
        
        return {
            "total_loss": total_loss / n_batches,
            "cls_loss": total_cls / n_batches,
            "rec_loss": total_rec / n_batches,
            "global_loss": total_global / n_batches,
            "accuracy": accuracy,
            "f1": f1
        }

    # ---------------------------------------------------------
    def evaluate(self, test_loader, head_name=None):
        """
        Legacy evaluate: single-head models.
        If head_name is provided for multihead models, evaluate that head.
        """
        self.model.eval()
        preds, gts, probs = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    seqs, labels = batch
                    labels = labels.to(self.device)
                    out = self.model(seqs)
                    # out may be dict if model is multihead
                    if isinstance(out, dict) and head_name is not None:
                        logits = out["logits"][head_name]
                    elif not isinstance(out, dict):
                        logits = out
                    else:
                        raise ValueError("Provide head_name for multihead model evaluation.")
                else:
                    # recommended format: (seqs, labels_dict, recon_targets_dict, true_tokens)
                    seqs, labels_dict, *_ = batch
                    labels = labels_dict[head_name].to(self.device)
                    out = self.model(seqs)
                    logits = out["logits"][head_name]

                soft = torch.softmax(logits, dim=1)
                _, pred = torch.max(logits, 1)
                preds.extend(pred.cpu().tolist())
                gts.extend(labels.cpu().tolist())
                probs.extend(soft.cpu().tolist())

        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds, average="weighted")
        return acc, f1, preds, gts, probs

    # ---------------------------------------------------------
    def evaluate_multitask(self, test_loader, heads=None):
        """
        Evalúa varias cabezas EN UNA SOLA PASADA (mucho más rápido).
        """
        self.model.eval()
        
        if heads is None:
            heads = list(self.model.classifiers.keys())
        
        # ✅ Almacenar TODAS las predicciones en una sola pasada
        all_preds = {head: [] for head in heads}
        all_gts = {head: [] for head in heads}
        
        print(f"🔍 Evaluating {len(heads)} taxonomic levels...")
        
        with torch.no_grad():
            # ✅ RECORRER TEST_LOADER UNA SOLA VEZ
            for batch in tqdm(test_loader, desc="Evaluating", ncols=100):
                if len(batch) == 2:
                    seqs, labels = batch
                    raise ValueError("To evaluate multiple heads, test_loader must provide labels_dict.")
                else:
                    seqs, labels_dict, *_ = batch
                    
                    # ✅ UN SOLO FORWARD PASS
                    out = self.model(seqs)
                    
                    # ✅ GUARDAR PREDICCIONES DE TODOS LOS TAXONES
                    for head in heads:
                        labels = labels_dict[head].to(self.device)
                        logits = out["logits"][head]
                        _, pred = torch.max(logits, 1)
                        
                        all_preds[head].extend(pred.cpu().tolist())
                        all_gts[head].extend(labels.cpu().tolist())
        
        # Calcular métricas para cada taxón
        results = {}
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        for head in heads:
            acc = accuracy_score(all_gts[head], all_preds[head])
            f1 = f1_score(all_gts[head], all_preds[head], average="weighted")
            results[head] = {"acc": acc, "f1": f1, "preds": all_preds[head], "gts": all_gts[head]}
            print(f"{head:12s} → Acc: {acc:6.2%} | F1: {f1:.4f}")
        
        print("="*60)
        return results

    # ---------------------------------------------------------
    def plot_confusion(self, y_true, y_pred, label_to_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[label_to_name[i] for i in sorted(label_to_name.keys())],
                    yticklabels=[label_to_name[i] for i in sorted(label_to_name.keys())])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()



import torch
import torch.nn as nn
from typing import Dict, List
import json

class AdaptiveCurriculumTrainer:
    """
    Entrena el modelo con curriculum learning adaptativo:
    - Empieza enfocándose en phylum
    - Cuando phylum alcanza 85% accuracy (o después de 3 épocas), activa order
    - Cuando order alcanza 85% (o después de 3 épocas), activa family
    - Y así sucesivamente
    """
    
    def __init__(self, model, runner, device='cuda', 
                 target_accuracy=0.85, max_epochs_per_level=3):
        """
        Args:
            model: El modelo a entrenar
            runner: ExperimentRunner
            device: cuda o cpu
            target_accuracy: Accuracy objetivo para pasar al siguiente nivel (default: 85%)
            max_epochs_per_level: Máximo de épocas antes de forzar el cambio
        """
        self.model = model
        self.runner = runner
        self.device = device
        self.threshold = 0.85
        self.target_accuracy = target_accuracy
        self.max_epochs_per_level = max_epochs_per_level
        
        # Orden de aprendizaje (de fácil a difícil)
        self.taxon_order = ['phylum', 'class','order', 'family', 'genus', 'species']
        
        # Estado del curriculum
        self.current_level = 0  # Empezar con phylum
        self.epochs_in_current_level = 0
        self.level_history = []
        
        print("🎓 Curriculum Learning Adaptativo Inicializado")
        print(f"   Target accuracy: {target_accuracy*100:.0f}%")
        print(f"   Max epochs por nivel: {max_epochs_per_level}")
    
    def get_current_weights(self):
        """
        Genera pesos según el nivel actual del curriculum
        
        Estrategia:
        - Niveles activos: peso normal (1.0)
        - Nivel en entrenamiento: peso alto (2.0)
        - Niveles futuros: peso muy bajo (0.1)
        """
        weights = {}
        
        for i, taxon in enumerate(self.taxon_order):
            if i < self.current_level:
                # Niveles ya aprendidos: peso normal
                weights[taxon] = 1.0
            elif i == self.current_level:
                # Nivel actual: peso alto
                weights[taxon] = 5.0
            else:
                # Niveles futuros: peso bajo
                weights[taxon] = 1.0
        
        return weights
    
    def should_advance(self, val_accuracy):
        """
        Decide si avanzar al siguiente nivel
        
        Args:
            val_accuracy: Accuracy de validación del nivel actual
            
        Returns:
            True si debe avanzar, False si debe continuar
        """
        current_taxon = self.taxon_order[self.current_level]
        current_acc = val_accuracy[current_taxon]

        if isinstance(current_acc, list):
            current_acc = current_acc[-1]
        
        # Criterio 1: Alcanzó el target accuracy
        if current_acc >= self.target_accuracy:
            print(f"\n✅ {current_taxon} alcanzó {current_acc*100:.1f}% (objetivo: {self.target_accuracy*100:.0f}%)")
            return True
        
        # Criterio 2: Pasaron max_epochs sin alcanzar target
        if self.epochs_in_current_level >= self.max_epochs_per_level:
            print(f"\n⏭️  {current_taxon} no alcanzó objetivo ({current_acc*100:.1f}%), pero pasaron {self.max_epochs_per_level} épocas")
            print(f"   Avanzando al siguiente nivel...")
            return True
        
        return False
    
    def advance_level(self):
        """Avanza al siguiente nivel del curriculum"""
        if self.current_level < len(self.taxon_order) - 1:
            # Guardar historial
            self.level_history.append({
                'level': self.current_level,
                'taxon': self.taxon_order[self.current_level],
                'epochs': self.epochs_in_current_level
            })
            
            self.current_level += 1
            self.epochs_in_current_level = 0
            
            current_taxon = self.taxon_order[self.current_level]
            print(f"\n📚 Avanzando a nivel {self.current_level + 1}/{len(self.taxon_order)}: {current_taxon}")
            print(f"   Pesos actualizados: {self.get_current_weights()}")
        else:
            print(f"\n🎉 Curriculum completo! Todos los niveles activados")
    
    def train(self, train_loader, val_loader, total_epochs=20, 
          alpha=1.0, beta=1.0, lr=1e-3, mixed_precision=True,
          enable_early_stop=True):
        """
        Entrena con curriculum learning adaptativo con early stopping
        
        SE DETIENE cuando:
        1. Todos los taxones llegan a 85% de accuracy, O
        2. Todos los taxones completaron sus épocas asignadas (2-3 épocas cada uno)
        
        Args:
            train_loader, val_loader: DataLoaders
            total_epochs: Total MÁXIMO de épocas a entrenar
            alpha, beta: Pesos para classification y reconstruction
            lr: Learning rate
            mixed_precision: Usar mixed precision
            enable_early_stop: Habilitar early stopping automático
        """
        print("\n" + "="*70)
        print("🎓 INICIANDO CURRICULUM LEARNING ADAPTATIVO")
        print("="*70)
        print(f"🎯 Threshold objetivo: {self.threshold * 100:.1f}%")
        print(f"⏱️  Épocas máximas por nivel: {self.max_epochs_per_level}")
        print(f"📊 Épocas máximas totales: {total_epochs}")
        print(f"🛑 Early stopping: {'Activado' if enable_early_stop else 'Desactivado'}")
        print("="*70)
        
        all_history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': {taxon: [] for taxon in self.taxon_order},
            'weights_history': [],
            'level_changes': []
        }
        
        # Tracking de cumplimiento de objetivos
        taxon_goals = {
            taxon: {
                'reached_threshold': False,  # ¿Llegó a 85%?
                'completed_epochs': False,   # ¿Completó 2-3 épocas?
                'epochs_trained': 0,         # Épocas entrenadas
                'best_acc': 0.0             # Mejor accuracy alcanzado
            } 
            for taxon in self.taxon_order
        }
        
        for epoch in range(total_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{total_epochs}")
            print(f"{'='*70}")
            
            current_taxon = self.taxon_order[self.current_level]
            print(f"📚 Nivel actual: {self.current_level + 1}/{len(self.taxon_order)} - {current_taxon}")
            print(f"   Épocas en este nivel: {self.epochs_in_current_level + 1}/{self.max_epochs_per_level}")
            
            # Obtener pesos actuales
            current_weights = self.get_current_weights()
            print(f"   Pesos: {current_weights}")
            all_history['weights_history'].append(current_weights.copy())
            
            # Entrenar una época
            epoch_history = self.runner.train_multitask(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=1,  # Una época a la vez
                alpha=alpha,
                beta=beta,
                lr=lr,
                taxon_weights=current_weights,
                mixed_precision=mixed_precision,
                early_stopping_patience=None,  # Desactivar early stopping interno
                save_best=False  # No guardar cada época
            )
            
            # Guardar métricas
            all_history['train_loss'].append(epoch_history['train_total'][-1])
            all_history['val_loss'].append(epoch_history['val_total'][-1])
            
            for taxon in self.taxon_order:
                all_history['val_acc'][taxon].append(epoch_history['val_acc'][taxon][-1])
            
            # Actualizar tracking de objetivos
            for taxon in self.taxon_order:
                acc = epoch_history['val_acc'][taxon][-1]
                
                # Actualizar mejor accuracy
                if acc > taxon_goals[taxon]['best_acc']:
                    taxon_goals[taxon]['best_acc'] = acc
                
                # Verificar si llegó al threshold
                if acc >= self.threshold and not taxon_goals[taxon]['reached_threshold']:
                    taxon_goals[taxon]['reached_threshold'] = True
                    print(f"   ✅ {taxon} alcanzó {acc*100:.1f}% (objetivo: {self.threshold*100:.1f}%)")
            
            # Incrementar contador del nivel actual
            self.epochs_in_current_level += 1
            taxon_goals[current_taxon]['epochs_trained'] += 1
            
            # Verificar si el nivel actual completó sus épocas
            if taxon_goals[current_taxon]['epochs_trained'] >= self.max_epochs_per_level:
                taxon_goals[current_taxon]['completed_epochs'] = True
            
            # Verificar si avanzar al siguiente nivel
            if self.should_advance(epoch_history['val_acc']):
                all_history['level_changes'].append(epoch + 1)
                
                # Antes de avanzar, verificar early stopping
                if enable_early_stop:
                    all_reached = all(goal['reached_threshold'] for goal in taxon_goals.values())
                    all_completed = all(goal['completed_epochs'] for goal in taxon_goals.values())
                    
                    if all_reached:
                        print(f"\n{'='*70}")
                        print(f"🏁 EARLY STOPPING: TODOS LOS TAXONES ALCANZARON {self.threshold*100:.1f}%")
                        print(f"{'='*70}")
                        self._print_final_summary(taxon_goals, epoch + 1, total_epochs, all_history)
                        return all_history
                    
                    if all_completed:
                        print(f"\n{'='*70}")
                        print(f"🏁 EARLY STOPPING: TODOS LOS TAXONES COMPLETARON SUS ÉPOCAS")
                        print(f"{'='*70}")
                        self._print_final_summary(taxon_goals, epoch + 1, total_epochs, all_history)
                        return all_history
                
                # Avanzar al siguiente nivel
                self.advance_level()
                
                # Si ya completamos todos los niveles, hacer fine-tuning final
                if self.current_level >= len(self.taxon_order):
                    print(f"\n🎉 Todos los niveles completados. Continuando fine-tuning con pesos iguales...")
            
            # Imprimir progreso de todos los niveles
            print(f"\n📊 Progreso general:")
            for i, taxon in enumerate(self.taxon_order):
                acc = epoch_history['val_acc'][taxon][-1]
                best = taxon_goals[taxon]['best_acc']
                epochs = taxon_goals[taxon]['epochs_trained']
                
                # Determinar status
                if taxon_goals[taxon]['reached_threshold']:
                    status = "✅"
                elif i < self.current_level:
                    status = "✅" if taxon_goals[taxon]['completed_epochs'] else "⚠️"
                elif i == self.current_level:
                    status = "🔄"
                else:
                    status = "⏸️"
                
                print(f"   {status} {taxon:10s}: {acc*100:5.1f}% (best: {best*100:5.1f}%, épocas: {epochs})")
        
        # Si llegamos aquí, completamos todas las épocas
        print(f"\n{'='*70}")
        print(f"🎓 ENTRENAMIENTO COMPLETADO - MÁXIMO DE ÉPOCAS ALCANZADO")
        print(f"{'='*70}")
        self._print_final_summary(taxon_goals, total_epochs, total_epochs, all_history)
        
        return all_history


    def _print_final_summary(self, taxon_goals, epochs_trained, total_epochs, all_history):
        """
        Imprime resumen final del entrenamiento
        """
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   Épocas entrenadas: {epochs_trained}/{total_epochs}")
        
        print(f"\n🎯 Estado de objetivos por taxón:")
        all_reached = True
        all_completed = True
        
        for taxon in self.taxon_order:
            goals = taxon_goals[taxon]
            reached = "✅" if goals['reached_threshold'] else "❌"
            completed = "✅" if goals['completed_epochs'] else "❌"
            
            print(f"   {taxon:10s}:")
            print(f"      Threshold ({self.threshold*100:.1f}%): {reached} ({goals['best_acc']*100:.1f}%)")
            print(f"      Épocas ({self.max_epochs_per_level}):  {completed} ({goals['epochs_trained']} entrenadas)")
            
            if not goals['reached_threshold']:
                all_reached = False
            if not goals['completed_epochs']:
                all_completed = False
        
        print(f"\n📈 Resultados finales:")
        for taxon in self.taxon_order:
            final_acc = all_history['val_acc'][taxon][-1]
            best_acc = taxon_goals[taxon]['best_acc']
            status = "✅" if taxon_goals[taxon]['reached_threshold'] else "❌"
            print(f"   {status} {taxon:10s}: {final_acc*100:6.2f}% (mejor: {best_acc*100:6.2f}%)")
        
        # Razón de terminación
        print(f"\n🏁 Razón de terminación:")
        if all_reached:
            print(f"   ✅ Todos los taxones alcanzaron {self.threshold*100:.1f}% de accuracy")
        elif all_completed:
            print(f"   ✅ Todos los taxones completaron {self.max_epochs_per_level} épocas de entrenamiento")
        elif epochs_trained >= total_epochs:
            print(f"   ⏱️  Se alcanzó el límite de {total_epochs} épocas")
        
        print(f"\n📚 Historial del curriculum:")
        for entry in self.level_history:
            print(f"   {entry['taxon']:10s}: {entry['epochs']} épocas")
        if self.current_level < len(self.taxon_order):
            print(f"   {self.taxon_order[self.current_level]:10s}: {self.epochs_in_current_level} épocas (nivel final)")
        
        # Guardar historial
        self.save_history(all_history)
        print(f"\n💾 Historial guardado")
        print("="*70)

    
    
    def save_history(self, history, path='curriculum_history.json'):
        """Guarda el historial del entrenamiento"""
        # Convertir a formato serializable
        serializable_history = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_acc': {k: [float(v) for v in vals] for k, vals in history['val_acc'].items()},
            'weights_history': history['weights_history'],
            'level_changes': history['level_changes'],
            'config': {
                'target_accuracy': self.target_accuracy,
                'max_epochs_per_level': self.max_epochs_per_level,
                'taxon_order': self.taxon_order
            }
        }
        
        with open(path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"\n💾 Historial guardado en: {path}")
