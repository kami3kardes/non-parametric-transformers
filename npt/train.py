"""Contains main training operations."""

import gc
import time
from multiprocessing import cpu_count

import numpy as np
import torch
from tqdm import tqdm

from npt.column_encoding_dataset import ColumnEncodingDataset, NPTDataset
from npt.loss import Loss
from npt.optim import LRScheduler
from npt.optim import TradeoffAnnealer
from npt.utils import debug
from npt.utils.batch_utils import collate_with_pre_batching
from npt.utils.encode_utils import torch_cast_to_dtype
from npt.utils.eval_checkpoint_utils import EarlyStopCounter, EarlyStopSignal
from npt.utils.logging_utils import Logger


class Trainer:
    def __init__(
            self, model, optimizer, scaler, c, wandb_run, cv_index,
            dataset: ColumnEncodingDataset = None,
            torch_dataset: NPTDataset = None,
            distributed_args=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = LRScheduler(
            c=c, name=c.exp_scheduler, optimizer=optimizer)
        self.c = c
        self.wandb_run = wandb_run
        self.is_distributed = False
        self.dataset = dataset
        if self.dataset is None:
            raise ValueError("Trainer requires a ColumnEncodingDataset (dataset) instance.")
        self.torch_dataset = torch_dataset
        self.cv_index = cv_index

        # Inject sampling-related data into dataset.data_dict
        # This must happen before NPTBatchDataset is created
        self._inject_cluster_assignments(cv_index)
        self._inject_static_prototypes(cv_index)

        self.max_epochs = self.get_max_epochs()

        # If precomputed per-CV cluster assignments exist, inject the correct
        # assignment array into the dataset's data_dict so downstream batching
        # (NPTBatchDataset) will pick it up.
        try:
            data_dict = getattr(self.dataset, 'data_dict', None)
            if data_dict is None:
                data_dict = getattr(self.dataset, 'dataset_dict', None)  # fallback name
            if data_dict is None and self.c.exp_batch_cluster_per_cv:
                # Use Logger if available else print
                try:
                    Logger.warn("Trainer could not find dataset.data_dict to inject per-CV clusters.")
                except Exception:
                    print("Warning: Trainer could not find dataset.data_dict to inject per-CV clusters.")
            if (data_dict is not None and
                    'cluster_assignments_per_cv' in data_dict and
                    getattr(self.c, 'exp_batch_cluster_per_cv', False)):
                data_dict['cluster_assignments'] = np.asarray(
                    data_dict['cluster_assignments_per_cv'][cv_index],
                    dtype=np.int32)
            elif data_dict is not None and 'cluster_assignments' in data_dict:
                # ensure dtype is integer and numpy
                data_dict['cluster_assignments'] = np.asarray(
                    data_dict['cluster_assignments'], dtype=np.int32)
        except Exception:
            # safe fallback: do nothing if mapping fails
            pass

        self.max_epochs = self.get_max_epochs()

        # Data Loading
        self.data_loader_nprocs = (
            cpu_count() if c.data_loader_nprocs == -1
            else c.data_loader_nprocs)

        if self.data_loader_nprocs > 0:
            print(
                f'Distributed data loading with {self.data_loader_nprocs} '
                f'processes.')

        # Only needs to be set in distributed setting; otherwise, submodules
        # such as Loss and EarlyStopCounter use c.exp_device for tensor ops.
        self.gpu = None

        if distributed_args is not None:
            print('Loaded in DistributedDataset.')
            self.is_distributed = True
            self.world_size = distributed_args['world_size']
            self.rank = distributed_args['rank']
            self.gpu = distributed_args['gpu']

        if c.exp_checkpoint_setting is None and c.exp_eval_test_at_end_only:
            raise Exception(
                'User is not checkpointing, but aims to evaluate the best '
                'performing model at the end of training. Please set '
                'exp_checkpoint_setting to "best_model" to do so.')

        # Prepare arguments for EarlyStopCounter and validate dataset metadata
        ds_metadata = getattr(self.dataset, 'metadata', None)
        if ds_metadata is None:
            raise ValueError("Trainer: dataset.metadata is required for checkpointing/early stopping.")
        ds_model_cache = getattr(self.dataset, 'model_cache_path', None)
        ds_n_cv_splits = getattr(self.dataset, 'n_cv_splits', 1)

        self.early_stop_counter = EarlyStopCounter(
            c=c, data_cache_prefix=ds_model_cache,
            metadata=ds_metadata, wandb_run=wandb_run, cv_index=cv_index,
            n_splits=min(ds_n_cv_splits, c.exp_n_runs),
            device=self.gpu)

        # Initialize from checkpoint, if available
        num_steps = 0

        if self.c.exp_load_from_checkpoint:
            checkpoint = self.early_stop_counter.get_most_recent_checkpoint()
            if checkpoint is not None:
                del self.model
                gc.collect()
                checkpoint_epoch, (
                    self.model, self.optimizer, self.scaler,
                    num_steps) = checkpoint

        # Initialize tradeoff annealer, fast forward to number of steps
        # recorded in checkpoint.
        if self.c.exp_tradeoff != -1:
            self.tradeoff_annealer = TradeoffAnnealer(
                c=c, num_steps=num_steps)
        else:
            self.tradeoff_annealer = None

        self.logger = Logger(
            self.c, self.optimizer, self.gpu, self.tradeoff_annealer)

        self.loss = Loss(
            self.c, dataset.metadata,
            device=self.gpu, tradeoff_annealer=self.tradeoff_annealer,
            is_minibatch_sgd=self.c.exp_minibatch_sgd)

        if self.c.exp_eval_every_epoch_or_steps == 'steps':
            self.last_eval = 0

    def _inject_cluster_assignments(self, cv_index):
        """Inject cluster assignments into data_dict for batch sampling.
        
        Supports:
        - Per-CV cluster assignments (cluster_assignments_per_cv)
        - Global cluster assignments (cluster_assignments)
        
        Args:
            cv_index: Current cross-validation fold index
        """
        try:
            data_dict = getattr(self.dataset, 'data_dict', None)
            if data_dict is None:
                data_dict = getattr(self.dataset, 'dataset_dict', None)
            
            if data_dict is None:
                if getattr(self.c, 'exp_batch_cluster_sampling', False) and self.c.verbose:
                    print("⚠ Warning: exp_batch_cluster_sampling enabled but no data_dict found")
                return
            
            if not getattr(self.c, 'exp_batch_cluster_sampling', False):
                return  # Cluster sampling not enabled
            
            # Case 1: Per-CV cluster assignments (computed separately for each fold)
            if ('cluster_assignments_per_cv' in data_dict and 
                    getattr(self.c, 'exp_batch_cluster_per_cv', False)):
                cluster_assigns = data_dict['cluster_assignments_per_cv'][cv_index]
                data_dict['cluster_assignments'] = np.asarray(cluster_assigns, dtype=np.int32)
                
                if self.c.verbose:
                    n_clusters = len(np.unique(cluster_assigns[cluster_assigns >= 0]))
                    print(f"✓ Loaded per-CV cluster assignments for fold {cv_index}: "
                          f"{n_clusters} clusters, {len(cluster_assigns)} samples")
            
            # Case 2: Global cluster assignments (same for all CV folds)
            elif 'cluster_assignments' in data_dict:
                data_dict['cluster_assignments'] = np.asarray(
                    data_dict['cluster_assignments'], dtype=np.int32)
                
                if self.c.verbose:
                    n_clusters = len(np.unique(data_dict['cluster_assignments']))
                    print(f"✓ Using global cluster assignments: "
                          f"{n_clusters} clusters, {len(data_dict['cluster_assignments'])} samples")
            
            # Case 3: Clustering enabled but no assignments found
            else:
                if self.c.verbose:
                    print("⚠ Warning: exp_batch_cluster_sampling enabled but no cluster_assignments found!")
                    print("   Run: python scripts/compute_clusters_for_npt.py --data_path <path>")
                    print("   Or disable exp_batch_cluster_sampling")

        except Exception as e:
            if self.c.verbose:
                print(f"Warning: Failed to load cluster assignments: {e}")

    def _inject_static_prototypes(self, cv_index):
        """Inject static prototype assignments into data_dict for batch sampling.
        
        Supports:
        - Per-CV prototypes (prototypes_per_cv)
        - Global prototypes (prototypes)
        
        Args:
            cv_index: Current cross-validation fold index
        """
        try:
            data_dict = getattr(self.dataset, 'data_dict', None)
            if data_dict is None:
                data_dict = getattr(self.dataset, 'dataset_dict', None)
            
            if data_dict is None:
                if getattr(self.c, 'exp_batch_prototype_sampling', False) and self.c.verbose:
                    print("⚠ Warning: exp_batch_prototype_sampling enabled but no data_dict found")
                return
            
            if not getattr(self.c, 'exp_batch_prototype_sampling', False):
                return  # Static prototype sampling not enabled
            
            # Case 1: Per-CV prototype assignments (computed separately for each fold)
            if ('prototypes_per_cv' in data_dict and 
                    getattr(self.c, 'exp_batch_prototype_per_cv', False)):
                proto_info = data_dict['prototypes_per_cv'][cv_index]
                data_dict['prototypes'] = proto_info
                
                if self.c.verbose:
                    n_prototypes = len(proto_info.get('prototype_indices', []))
                    k_neighbors = proto_info.get('k', 'unknown')
                    proto_by = proto_info.get('by', 'unknown')
                    print(f"✓ Loaded per-CV prototypes for fold {cv_index}: "
                          f"{n_prototypes} prototypes, k={k_neighbors}, by={proto_by}")
            
            # Case 2: Global prototypes (same for all CV folds)
            elif 'prototypes' in data_dict:
                if self.c.verbose:
                    proto_info = data_dict['prototypes']
                    n_prototypes = len(proto_info.get('prototype_indices', []))
                    k_neighbors = proto_info.get('k', 'unknown')
                    proto_by = proto_info.get('by', 'unknown')
                    print(f"✓ Using global prototypes: "
                          f"{n_prototypes} prototypes, k={k_neighbors}, by={proto_by}")
            
            # Case 3: Prototype sampling enabled but no prototypes found
            else:
                if self.c.verbose:
                    print("⚠ Warning: exp_batch_prototype_sampling enabled but no prototypes found!")
                    print("   Run: python scripts/compute_prototypes_for_npt.py --data_path <path>")
                    print("   Or disable exp_batch_prototype_sampling")

        except Exception as e:
            if self.c.verbose:
                print(f"Warning: Failed to load prototypes: {e}")

    def get_distributed_dataloader(self, epoch):
        if not self.is_distributed:
            raise Exception

        # Ensure the torch_dataset is available for distributed loading
        if self.torch_dataset is None:
            raise ValueError("Distributed dataloader requested but torch_dataset is None.")

        sampler = torch.utils.data.distributed.DistributedSampler(
            self.torch_dataset,
            num_replicas=self.world_size,
            rank=self.rank)

        dataloader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=1,  # The dataset is already batched.
            shuffle=False,
            num_workers=self.data_loader_nprocs,
            pin_memory=True,
            collate_fn=collate_with_pre_batching,
            sampler=sampler)

        dataloader.sampler.set_epoch(epoch=epoch)
        total_steps = len(dataloader)

        if self.c.verbose:
            print('Successfully loaded distributed batch dataloader.')

        return dataloader, total_steps

    def get_num_steps_per_epoch(self):
        if self.c.exp_batch_size == -1:
            return 1

        N = self.dataset.metadata['N']
        return int(np.ceil(N / self.c.exp_batch_size))

    def get_max_epochs(self):
        # When evaluating row interactions:
        # We assume a trained model loaded from checkpoint.
        # Run two epochs:
        #   - (1) evaluate train/val/test loss without row corruptions
        #   - (2) evaluate train/val/test loss with row corruptions
        if self.c.debug_eval_row_interactions:
            return 2

        num_steps_per_epoch = self.get_num_steps_per_epoch()
        return int(
            np.ceil(self.c.exp_num_total_steps / num_steps_per_epoch))

    def per_epoch_train_eval(self, epoch):
        early_stop = False
        if self.c.verbose:
            print(f'Epoch: {epoch}/{self.max_epochs}.')

        # need to increase step counter by one here (because step counter is)
        # still at last step
        end_experiment = (
                self.scheduler.num_steps + 1 >= self.c.exp_num_total_steps)

        # Immediately jump into end evaluation if we are debugging row interact
        end_experiment = end_experiment or (
            self.c.debug_eval_row_interactions and epoch == 2)

        eval_model = end_experiment or self.eval_check(epoch)

        if self.c.debug_eval_row_interactions:
            train_loss = None
        else:
            # The returned train loss is used for logging at eval time
            # It is None if minibatch_sgd is enabled, in which case we
            # perform an additional forward pass over all train entries
            train_loss = self.run_epoch(dataset_mode='train', epoch=epoch,
                                        eval_model=False)

        if eval_model:
            early_stop = self.eval_model(
                train_loss, epoch, end_experiment)
        if early_stop or end_experiment:
            early_stop = True
            return early_stop

        return early_stop

    def train_and_eval(self):
        """Main training and evaluation loop."""
        self.logger.start_counting()

        if self.is_distributed and self.c.mp_no_sync != -1:
            curr_epoch = 1

            while curr_epoch <= self.max_epochs:
                with self.model.no_sync():
                    print(f'No DDP synchronization for the next '
                          f'{self.c.mp_no_sync} epochs.')

                    for epoch in range(
                            curr_epoch, curr_epoch + self.c.mp_no_sync):
                        if self.per_epoch_train_eval(epoch=epoch):
                            return

                        if epoch >= self.max_epochs:
                            return

                curr_epoch += self.c.mp_no_sync

                if epoch >= self.max_epochs:
                    return

                print(f'Synchronizing DDP gradients in this epoch '
                      f'(epoch {curr_epoch}).')
                if self.per_epoch_train_eval(epoch=curr_epoch):
                    return

                curr_epoch += 1
        else:
            for epoch in range(1, self.max_epochs + 1):
                if self.per_epoch_train_eval(epoch=epoch):
                    break

    def eval_model(self, train_loss, epoch, end_experiment):
        """Obtain val and test losses."""
        kwargs = dict(epoch=epoch, eval_model=True)

        # Evaluate over val rows
        val_loss = self.run_epoch(dataset_mode='val', **kwargs)

        if not (self.c.debug_eval_row_interactions and epoch == 2):
            # Early stopping check -- TODO: consider loss other than label?
            counter, best_model_and_opt = self.early_stop_counter.update(
                val_loss=val_loss['label']['total_loss'],
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                end_experiment=end_experiment,
                tradeoff_annealer=self.tradeoff_annealer)
        else:
            counter = EarlyStopSignal.END

        if not self.c.debug_eval_row_interactions:
            if (counter == EarlyStopSignal.STOP) or end_experiment:
                if best_model_and_opt is not None:
                    print('Loaded best performing model for last evaluation.')
                    self.model, self.optimizer, self.scaler, num_steps = (
                        best_model_and_opt)

                    # Initialize tradeoff annealer, fast forward to number of steps
                    # recorded in checkpoint.
                    if self.tradeoff_annealer is not None:
                        self.tradeoff_annealer = TradeoffAnnealer(
                            c=self.c, num_steps=num_steps)

                        # Update the tradeoff annealer reference in the logger
                        self.logger.tradeoff_annealer = self.tradeoff_annealer

                # update val loss
                val_loss = self.run_epoch(dataset_mode='val', **kwargs)

        if train_loss is None and not self.c.debug_eval_row_interactions:
            # Train and compute loss over masked features in train rows
            train_loss = self.run_epoch(dataset_mode='train', **kwargs)
        elif self.c.debug_eval_row_interactions:
            train_loss = {}

        # Check if we need to eval test
        if ((counter == EarlyStopSignal.STOP)
            or (not self.c.exp_eval_test_at_end_only)
                or (self.c.exp_eval_test_at_end_only and end_experiment)):
            # Evaluate over test and val rows again
            test_loss = self.run_epoch(dataset_mode='test', **kwargs)
        else:
            test_loss = None

        loss_dict = self.logger.log(
            train_loss, val_loss, test_loss, self.scheduler.num_steps, epoch)

        # Update summary metrics
        new_min = (
            self.early_stop_counter.num_inc_valid_loss_epochs == 0)
        self.logger.summary_log(loss_dict, new_min)

        if counter == EarlyStopSignal.STOP:
            print(self.early_stop_counter.stop_signal_message)
            return True
        else:
            return False

    # fp = open('memory_profiler.log', 'w+')
    # @profile(stream=fp)
    # @profile
    def run_epoch(self, dataset_mode, epoch, eval_model=False):
        """Train or evaluate model for a full epoch.

        Args:
            dataset_mode (str) {'train', 'test', 'eval'}: Depending on value
                mask/input the relevant parts of the data.
            epoch (int): Only relevant for logging.
            eval_model (bool): If this is true, write some extra metrics into
                the loss_dict. Is always true for test and eval, but only
                sometimes true for train. (We do not log each train epoch).

        Returns:
            loss_dict: Results of model for logging purposes.

        If `self.c.exp_minibatch_sgd` is True, we backpropagate after every
        mini-batch. If it is False, we backpropagate once per epoch.
        """
        print_n = self.c.exp_print_every_nth_forward

        # Model prep
        # We also want to eval train loss
        if (dataset_mode == 'train') and not eval_model:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        # Dataset prep -- prepares dataset.batch_gen attribute
        # Relevant in 'production' setting: we want to only input train
        # at train, train/val at val and train/val/test at test.
        if not hasattr(self.dataset, 'set_mode'):
            raise ValueError("Dataset object does not implement set_mode(mode, epoch).")
        self.dataset.set_mode(mode=dataset_mode, epoch=epoch)

        # Initialize data loaders (serial / distributed, pinned memory)
        if self.is_distributed:
            # TODO: parallel DDP loading?
            if not hasattr(self.dataset, 'cv_dataset'):
                raise ValueError("Distributed mode requires dataset.cv_dataset to exist.")
            # Ensure torch_dataset can materialize from cv_dataset if supported
            if self.torch_dataset is not None and hasattr(self.torch_dataset, 'materialize'):
                self.torch_dataset.materialize(cv_dataset=self.dataset.cv_dataset)
            batch_iter, num_batches = self.get_distributed_dataloader(epoch)
        else:
            # TODO: can be beneficial to test > cpu_count() procs if our
            # loading is I/O bound (which it probably is)
            batch_dataset = self.dataset.cv_dataset
            extra_args = {}

            if not self.c.data_set_on_cuda:
                extra_args['pin_memory'] = True

            # Ensure cv_dataset exists
            if not hasattr(self.dataset, 'cv_dataset'):
                raise ValueError("Dataset object must expose 'cv_dataset' after set_mode().")

            batch_iter = torch.utils.data.DataLoader(
                dataset=batch_dataset,
                batch_size=1,  # The dataset is already batched.
                shuffle=False,  # Already shuffled
                num_workers=self.data_loader_nprocs,
                collate_fn=collate_with_pre_batching,
                **extra_args)

            batch_iter = tqdm(batch_iter, desc='Batch') if self.c.verbose else batch_iter

        if (eval_model and self.c.debug_eval_row_interactions
                and epoch == 2 and dataset_mode in {'test'}):
            if self.c.debug_eval_row_interactions_timer is not None:
                self.set_row_corruption_timer()

        for batch_index, batch_dict_ in enumerate(batch_iter):
            if self.c.debug_row_interactions:
                batch_dict_ = debug.modify_data(
                    self.c, batch_dict_, dataset_mode,
                    self.scheduler.num_steps)

            # Row-wise evaluation mode (debugging)
            if (eval_model and self.c.debug_eval_row_interactions
                    and epoch == 2 and dataset_mode in {'test'}):
                n_rows = batch_dict_['data_arrs'][0].shape[0]

                if (self.c.debug_row_interactions and
                    self.c.debug_row_interactions_mode == 'protein-duplicate'):
                    n_rows = n_rows // 2

                if batch_dict_['label_mask_matrix'] is not None:
                    label_matrix_key = 'label'
                else:
                    label_matrix_key = dataset_mode

                label_matrix = batch_dict_[f'{label_matrix_key}_mask_matrix']

                for row_index in range(n_rows):
                    if (self.c.debug_row_interactions and
                            self.c.debug_row_interactions_mode ==
                            'protein-duplicate'):
                        original_row_index = row_index + n_rows
                    else:
                        original_row_index = row_index

                    if label_matrix[original_row_index, :].sum() == 0:
                        continue

                    modified_batch_dict = debug.corrupt_rows(
                        self.c, batch_dict_, dataset_mode, row_index)

                    self.run_batch(modified_batch_dict, dataset_mode,
                                   eval_model, epoch, print_n, batch_index)

                if self.c.debug_eval_row_interactions_timer is not None:
                    if batch_index % 50 == 0:
                        loss_dict = self.loss.get_intermediate_epoch_losses()
                        loss_dicts = {
                            'train_loss': {},
                            'val_loss': {},
                            'test_loss': {}}
                        loss_dicts[f'{dataset_mode}_loss'] = loss_dict
                        self.logger.log(
                            steps=self.scheduler.num_steps, epoch=epoch,
                            **loss_dicts)

                    if self.check_row_corruption_timer():
                        break

            else:
                # Normal execution
                self.run_batch(
                    batch_dict_, dataset_mode, eval_model,
                    epoch, print_n, batch_index)

        # Perform batch GD?
        batch_GD = (dataset_mode == 'train') and (
            not self.c.exp_minibatch_sgd)

        if eval_model or batch_GD:
            # We want loss_dict either for logging purposes
            # or to backpropagate if we do full batch GD
            loss_dict = self.loss.finalize_epoch_losses(eval_model)

        # Full-batch GD path
        if (not eval_model) and batch_GD:
            train_loss = loss_dict['total_loss']
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.tradeoff_annealer is not None:
                self.tradeoff_annealer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

        # Reset batch and epoch losses
        self.loss.reset()

        if (not eval_model) and self.c.exp_minibatch_sgd:
            loss_dict = None

        return loss_dict

    def run_batch(self, batch_dict, dataset_mode, eval_model,
                  epoch, print_n, batch_index):
        # In stochastic label masking, we actually have a separate
        # label_mask_matrix. Else, it is just None.
        masked_tensors, label_mask_matrix, augmentation_mask_matrix = (
            batch_dict['masked_tensors'],
            batch_dict['label_mask_matrix'],
            batch_dict['augmentation_mask_matrix'])

        if self.c.debug_label_leakage:
            debug.leakage(
                self.c, batch_dict, masked_tensors, label_mask_matrix,
                dataset_mode)

        # Construct ground truth tensors
        ground_truth_tensors = batch_dict['data_arrs']

        if not self.c.data_set_on_cuda:
            if self.is_distributed:
                device = self.gpu
            else:
                device = self.c.exp_device

            # non_blocking flag is appropriate when we are pinning memory
            # and when we use Distributed Data Parallelism

            # If we are fitting the full dataset on GPU, the following
            # tensors are already on the remote device. Otherwise, we can
            # transfer them with the non-blocking flag, taking advantage
            # of pinned memory / asynchronous transfer.

            # Cast tensors to appropriate data type
            ground_truth_tensors = [
                torch_cast_to_dtype(obj=data, dtype_name=self.c.data_dtype)
                for data in ground_truth_tensors]
            ground_truth_tensors = [
                data.to(device=device, non_blocking=True)
                for data in ground_truth_tensors]
            masked_tensors = [
                data.to(device=device, non_blocking=True)
                for data in masked_tensors]

            # Send everything else used in loss compute to the device
            batch_dict[f'{dataset_mode}_mask_matrix'] = (
                batch_dict[f'{dataset_mode}_mask_matrix'].to(
                    device=device, non_blocking=True))

            if augmentation_mask_matrix is not None:
                augmentation_mask_matrix = augmentation_mask_matrix.to(
                    device=device, non_blocking=True)

            # Need label_mask_matrix for stochastic label masking
            if label_mask_matrix is not None:
                label_mask_matrix = label_mask_matrix.to(
                    device=device, non_blocking=True)

        forward_kwargs = dict(
            batch_dict=batch_dict,
            ground_truth_tensors=ground_truth_tensors,
            masked_tensors=masked_tensors, dataset_mode=dataset_mode,
            eval_model=eval_model, epoch=epoch,
            label_mask_matrix=label_mask_matrix,
            augmentation_mask_matrix=augmentation_mask_matrix)

        # This Automatic Mixed Precision autocast is a no-op
        # of c.model_amp = False
        with torch.cuda.amp.autocast(enabled=self.c.model_amp):
            self.forward_and_loss(**forward_kwargs)

        # (See docstring) Either perform mini-batch SGD (as here)
        # or full-batch GD (as further below)
        if (dataset_mode == 'train' and self.c.exp_minibatch_sgd
                and (not eval_model)):
            # Standardize and backpropagate on minibatch loss
            # if minibatch_sgd enabled
            loss_dict = self.loss.finalize_batch_losses()
            train_loss = loss_dict['total_loss']

            # ### Apply Automatic Mixed Precision ###
            # The scaler ops will be no-ops if we have specified
            # c.model_amp is False in the Trainer init

            # Scales loss.
            # Calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(train_loss).backward()

            # scaler.step() first unscales the gradients of the
            # optimizer's assigned params.
            # If these gradients do not contain infs or NaNs,
            # optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

            if self.tradeoff_annealer is not None:
                self.tradeoff_annealer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            if print_n and (self.scheduler.num_steps % print_n == 0):
                self.logger.intermediate_log(
                    loss_dict=loss_dict,
                    num_steps=self.scheduler.num_steps,
                    batch_index=batch_index, epoch=epoch)

        # Update the epoch loss info with detached minibatch losses
        self.loss.update_losses(eval_model=eval_model)

    def forward_and_loss(
            self, batch_dict, ground_truth_tensors, masked_tensors,
            dataset_mode, eval_model, epoch, label_mask_matrix,
            augmentation_mask_matrix,):
        """Run forward pass and evaluate model loss."""
        extra_args = {}

        if eval_model:
            with torch.no_grad():
                output = self.model(masked_tensors, **extra_args)
        else:
            output = self.model(masked_tensors, **extra_args)

        loss_kwargs = dict(
            output=output, ground_truth_data=ground_truth_tensors,
            label_mask_matrix=label_mask_matrix,
            augmentation_mask_matrix=augmentation_mask_matrix,
            data_dict=batch_dict, dataset_mode=dataset_mode,
            eval_model=eval_model)

        self.loss.compute(**loss_kwargs)

    def eval_check(self, epoch):
        """Check if it's time to evaluate val and test errors."""

        if self.c.exp_eval_every_epoch_or_steps == 'epochs':
            return epoch % self.c.exp_eval_every_n == 0
        elif self.c.exp_eval_every_epoch_or_steps == 'steps':
            # Cannot guarantee that we hit modulus directly.
            if (self.scheduler.num_steps - self.last_eval >=
                    self.c.exp_eval_every_n):
                self.last_eval = self.scheduler.num_steps
                return True
            else:
                return False
        else:
            raise ValueError

    def set_row_corruption_timer(self):
        assert self.c.debug_eval_row_interactions_timer is not None
        self.row_corruption_timer = time.time()
        self.n_row_corr_batches = 0

    def check_row_corruption_timer(self):
        break_loop = False
        self.n_row_corr_batches += 1
        n_examples = self.n_row_corr_batches * self.c.exp_batch_size
        print(f'Row Corruption: completed {self.n_row_corr_batches} batches, '
              f'{n_examples} examples.')

        if (time.time() - self.row_corruption_timer >
                (self.c.debug_eval_row_interactions_timer * 60 * 60)):
            print(f'Row Corruption: have reached time limit: '
                  f'{self.c.debug_eval_row_interactions_timer} hours.')
            print('Breaking loop.')
            break_loop = True

        return break_loop

    # (per-CV cluster assignment wiring is handled in Trainer.__init__)
