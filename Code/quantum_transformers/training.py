from typing import Optional
import time

import numpy.typing as npt
import jax
import jax.numpy as jnp
import flax.linen
import flax.training.train_state
import optax
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import torch


dtype_mapping = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    # Add other necessary dtype mappings as needed
}

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


class TrainState(flax.training.train_state.TrainState):
    # See https://flax.readthedocs.io/en/latest/guides/dropout.html.
    key: jax.random.KeyArray  # type: ignore


@jax.jit
def train_step(state: TrainState, inputs: jax.Array, labels: jax.Array, key: jax.random.KeyArray) -> TrainState:
    """
    Performs a single training step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.
        key: The random key to use.

    Returns:
        The updated training state.
    """
    key, dropout_key = jax.random.split(key=key)
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            x=inputs,
            train=True,
            rngs={'dropout': dropout_train_key}
        )
        if logits.shape[1] <= 2:
            if logits.shape[1] == 2:
                logits = logits[:, 1]
            loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
        # return loss, logits
        return loss
    
    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss, logits), grads = grad_fn(state.params)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(state: TrainState, inputs: jax.Array, labels: jax.Array) -> tuple[float, jax.Array]:
    """
    Performs a single evaluation step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.

    Returns:
        loss: The loss on the given batch.
        logits: The logits on the given batch.
    """
    logits = state.apply_fn(
        {'params': state.params},
        x=inputs,
        train=False,
        rngs={'dropout': state.key}
    )
    if logits.shape[1] <= 2:
        if logits.shape[1] == 2:
            logits = logits[:, 1]
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits


def evaluate(state: TrainState, eval_dataloader, num_classes: int,
             tqdm_desc: Optional[str] = None, debug: bool = True) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logits_list, labels_list = [], []
    eval_loss = 0.0
    with tqdm(total=len(eval_dataloader), desc=tqdm_desc, unit="batch", bar_format=TQDM_BAR_FORMAT, disable=tqdm_desc is None) as progress_bar:
        for inputs_batch, labels_batch in eval_dataloader:
            inputs_batch_np = inputs_batch.cpu().detach().numpy().transpose((0, 2, 3, 1))
            labels_batch_np = labels_batch.cpu().detach().numpy()
            inputs_batch_jax = jnp.array(inputs_batch_np)
            labels_batch_jax = jnp.array(labels_batch_np)

            loss_batch, logits_batch = eval_step(state, inputs_batch_jax, labels_batch_jax)
            logits_list.append(logits_batch)
            labels_list.append(labels_batch_jax)
            eval_loss += loss_batch
            progress_bar.update(1)

        eval_loss /= len(eval_dataloader)
        logits = np.concatenate(logits_list)
        labels = np.concatenate(labels_list)

        if num_classes == 2:
            # Binary classification
            y_score = jax.nn.sigmoid(logits).ravel()
            eval_fpr, eval_tpr, _ = roc_curve(labels, y_score)
            eval_auc = 100.0 * auc(eval_fpr, eval_tpr)
        else:
            # Multi-class classification (assuming OvR approach for AUC)
            y_score = jax.nn.softmax(logits, axis=1)
            eval_fpr, eval_tpr, eval_auc = None, None, 100.0 * roc_auc_score(labels, y_score, multi_class="ovr")

        progress_bar.set_postfix_str(f"Loss = {eval_loss:.4f}, AUC = {eval_auc:.4f}%")


        # eval_loss /= len(eval_dataloader)
        # logits = jnp.concatenate(logits)  # type: ignore
        # y_true = jnp.concatenate(labels)  # type: ignore
        # if debug:
        #     print(f"logits = {logits}")
        # if num_classes == 2:
        #     y_pred = [jax.nn.sigmoid(l) for l in logits]
        # else:
        #     y_pred = [jax.nn.softmax(l) for l in logits]
        # if debug:
        #     print(f"y_pred = {y_pred}")
        #     print(f"y_true = {y_true}")
        # if num_classes == 2:
        #     eval_fpr, eval_tpr, _ = roc_curve(y_true, y_pred)
        #     eval_auc = 100.0 * auc(eval_fpr, eval_tpr)
        # else:
        #     eval_fpr, eval_tpr = [], []
        #     eval_auc = 100.0 * roc_auc_score(y_true, y_pred, multi_class='ovr')
        # progress_bar.set_postfix_str(f"Loss = {eval_loss:.4f}, AUC = {eval_auc:.4f}%")

    return eval_loss, eval_auc, eval_fpr, eval_tpr, logits, labels


# Initialize metrics lists and start timing
train_losses = []
val_losses = []
train_aucs = []
val_aucs = []
training_start_time = time.time()  # Start timing

def train_and_evaluate(model: flax.linen.Module, train_dataloader, val_dataloader, test_dataloader, num_classes: int,
                       num_epochs: int, lrs_peak_value: float = 1e-3, lrs_warmup_steps: int = 9_000, lrs_decay_steps: int = 70_000,
                       seed: int = 42, use_ray: bool = False, debug: bool = True) -> tuple[float, float, npt.ArrayLike, npt.ArrayLike]:
    """
    Trains the given model on the given dataloaders for the given hyperparameters.

    The progress and evaluation results are printed to stdout.

    Args:
        model: The model to train.
        train_dataloader: The dataloader for the training set.
        val_dataloader: The dataloader for the validation set.
        num_classes: The number of classes.
        num_epochs: The number of epochs to train for.
        learning_rate: The learning rate to use.
        seed: The seed to use for reproducibility.
        use_ray: Whether to use Ray for logging.
        debug: Whether to print extra information for debugging.

    Returns:
        None
    """

    if use_ray:
        from ray.air import session

    root_key = jax.random.PRNGKey(seed=seed)
    root_key, params_key, train_key = jax.random.split(key=root_key, num=3)

    dummy_batch = next(iter(train_dataloader))[0]
    input_shape = dummy_batch[0].shape
    input_dtype = dummy_batch[0].dtype
    np_dtype = dtype_mapping.get(input_dtype, None)
    batch_size = len(dummy_batch)
    root_key, input_key = jax.random.split(key=root_key)

    if np_dtype is None:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    if jnp.issubdtype(np_dtype, jnp.floating):
        dummy_batch = jax.random.uniform(key=input_key, shape=(batch_size,) + tuple(input_shape), dtype=np_dtype)
    elif jnp.issubdtype(np_dtype, jnp.integer):
        dummy_batch = jax.random.randint(key=input_key, shape=(batch_size,) + tuple(input_shape), minval=0, maxval=100, dtype=np_dtype)
    else:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    dummy_batch_cl = dummy_batch.transpose((0, 2, 3, 1))  # Convert to (batch_size, height, width, num_channels)
    #print("Converted dummy batch shape:", dummy_batch_cl.shape)  # Confirming converted dummy batch shape

    variables = model.init(params_key, dummy_batch_cl, train=False)

    if debug:
        print(jax.tree_map(lambda x: x.shape, variables))
    print(f"Number of parameters = {sum(x.size for x in jax.tree_util.tree_leaves(variables))}")

    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lrs_peak_value,
        warmup_steps=lrs_warmup_steps,
        decay_steps=lrs_decay_steps,
        end_value=0.0
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate_schedule),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        key=train_key,
        tx=optimizer
    )

    print(f"Starting training for {num_epochs} epochs...")  # Number of epochs

    best_val_auc, best_epoch, best_state = 0.0, 0, None
    start_time = time.time()
    for epoch in range(num_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1:3}/{num_epochs}", unit="batch", bar_format=TQDM_BAR_FORMAT) as progress_bar:
            for inputs_batch, labels_batch in train_dataloader:
                
                inputs_batch_np = inputs_batch.cpu().detach().numpy().transpose((0, 2, 3, 1))
                labels_batch_np = labels_batch.cpu().detach().numpy()

                #print(f"Batch shape after transpose: {inputs_batch_np.shape}")  # Inspect shape
                #print(f"Batch dtype: {inputs_batch_np.dtype}")  # Inspect dtype

                inputs_batch_jax = jax.numpy.array(inputs_batch_np)
                labels_batch_jax = jax.numpy.array(labels_batch_np)

                state = train_step(state, inputs_batch_jax, labels_batch_jax, train_key)

                progress_bar.update(1)

            val_loss, val_auc, _, _,_,_ = evaluate(state, val_dataloader, num_classes, tqdm_desc=None, debug=debug)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)

            progress_bar.set_postfix_str(f"Loss = {val_loss:.4f}, AUC = {val_auc:.4f}%")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                best_state = state

            print(f"Epoch {epoch+1} completed. Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}%")  # Epoch summary

            if use_ray:
                session.report({'val_loss': val_loss, 'val_auc': val_auc, 'best_val_auc': best_val_auc, 'best_epoch': best_epoch})

    print(f"Total training time = {time.time() - start_time:.2f}s, best validation AUC = {best_val_auc:.4f}% at epoch {best_epoch}")  # Final summary

    total_training_time = time.time() - training_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    print(f"Training Losses: {train_losses}")
    print(f"Training AUCs: {train_aucs}")
    print(f"Validation Losses: {val_losses}")
    print(f"Validation AUCs: {val_aucs}")
    print(f"Best Validation AUC: {best_val_auc:.4f}% at Epoch: {best_epoch}")

    # Evaluate on test set using the best model
    assert best_state is not None
    test_loss, test_auc, test_fpr, test_tpr, all_logits, all_labels = evaluate(best_state, test_dataloader, num_classes, tqdm_desc="Testing")
    if use_ray:
        session.report({'test_loss': test_loss, 'test_auc': test_auc})
    
    return test_loss, test_auc, test_fpr, test_tpr, all_logits, all_labels
