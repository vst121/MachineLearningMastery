# Fine-Tuning and Transfer Learning

## Learning Objectives

- Understand how pretrained language models are adapted to downstream NLP tasks.
- Implement fine-tuning using Hugging Face Transformers and PyTorch.
- Separate training and validation data to monitor generalization.
- Apply early stopping and checkpointing to preserve the best-performing model.
- Compare full fine-tuning, partial layer freezing, and frozen-backbone strategies.
- Evaluate transfer-learning experiments using classification metrics, parameter counts, and training time.

## Core Concepts

### 1. Transfer Learning

Transfer learning reuses knowledge learned during pretraining to solve a downstream task. Instead of learning all language representations from scratch, a pretrained model is adapted using task-specific data.

For model parameters $$\theta$$, fine-tuning minimizes a task-specific loss:

$$
\theta^* = \underset{\theta}{\text{argmin}} \; \mathcal{L}_{\text{task}}(\theta)
$$

A pretrained model typically contains a Transformer backbone and a task-specific classification head.

### 2. Fine-Tuning and Gradient Updates

For a model with parameters $$\theta$$, gradient-based optimization updates parameters using the gradient of the training loss:

$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta}\mathcal{L}_{\text{train}}(\theta_t)
$$

Here, $$\eta$$ is the learning rate. A smaller learning rate can help preserve useful pretrained representations, while a larger rate may produce faster adaptation but can destabilize training.

### 3. Training and Validation

Training loss measures how well the model fits training examples. Validation loss helps assess generalization to examples excluded from gradient updates.

$$
\mathcal{L}_{\text{train}} = \frac{1}{N_{\text{train}}}
\sum_{i=1}^{N_{\text{train}}}
\ell(f_{\theta}(x_i), y_i)
$$

$$
\mathcal{L}_{\text{validation}} = \frac{1}{N_{\text{validation}}}
\sum_{i=1}^{N_{\text{validation}}}
\ell(f_{\theta}(x_i), y_i)
$$

A decreasing training loss alongside an increasing validation loss can indicate overfitting.

### 4. Early Stopping and Checkpointing

Early stopping terminates training when a monitored validation metric stops improving. Checkpointing saves model states so that the best observed checkpoint can be restored.

$$
e^* =
\underset{e \in \{1,\ldots,E\}}{\text{argmin}}
\mathcal{L}_{\text{validation}}(\theta_e)
$$

The selected model is the checkpoint with the lowest observed validation loss, rather than necessarily the parameters from the final epoch.

### 5. Parameter Freezing Strategies

Transfer learning can control which parameters are updated during fine-tuning.

- **Full fine-tuning:** all Transformer and classification-head parameters are trainable.
- **Frozen backbone:** only the task-specific head is trainable.
- **Partial fine-tuning:** selected upper Transformer layers and the classification head are trainable.

The percentage of trainable parameters is:

$$
P_{\text{trainable}} = 100 \cdot
\frac{N_{\text{trainable}}}{N_{\text{total}}}
$$

Freezing layers reduces gradient computation and optimizer state requirements, although the complete model still needs to perform forward inference.

### 6. Classification Evaluation

Accuracy measures the proportion of correct predictions:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Precision and recall measure positive-prediction quality and positive-class coverage:

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

$$
\text{Recall} = \frac{TP}{TP+FN}
$$

The F1 score is their harmonic mean:

$$
F_1 =
2 \cdot
\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

Using several metrics provides a more informative comparison than accuracy alone, particularly when class distributions are imbalanced.

## Implementation and Experiments

| File                                            | Experiment                                                                                                                                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_fine_tuning_fundamentals.py`                | Fine-tune pretrained DistilBERT for sentiment classification. Inspect the classification head, evaluate predictions before and after training, and measure parameter changes.            |
| `02_training_validation_and_checkpointing.py`   | Track training and validation metrics, apply early stopping, restore the best checkpoint, and save the selected model.                                                                   |
| `03_learning_rate_and_freezing_strategies.py`   | Compare full fine-tuning, a frozen backbone, partial layer freezing, and different learning rates. Measure trainable parameters and training time.                                       |
| `04_transfer_learning_experiment_comparison.py` | Compare a majority-class baseline with multiple pretrained-model adaptation strategies. Record classification metrics, validation loss, and resource measurements in CSV and JSON files. |

The experiments use Hugging Face Transformers, PyTorch, and a small illustrative sentiment-classification dataset. They are designed to demonstrate training workflows and trade-offs, not to establish reliable benchmark performance.

## Expected Outcome

After completing this section, you should be able to:

- Fine-tune a pretrained Transformer for a downstream NLP task.
- Implement reproducible training and validation workflows.
- Use early stopping and checkpoint restoration appropriately.
- Explain how learning rates and layer freezing influence adaptation.
- Compare model strategies using predictive quality and computational cost.
- Persist experimental results for subsequent analysis.

The key engineering lesson is that successful transfer learning requires more than selecting a pretrained model. It also requires careful optimization, validation, checkpoint selection, and controlled experimentation.
