import copy

from mmengine.hooks.hook import Hook, DATA_BATCH
from mmengine.model import is_model_wrapper
import numpy as np
import torch
from tqdm import tqdm

from mmengine.logging import MMLogger
from mmpretrain.registry import HOOKS

from projects.DENTEX2.hooks.variational import (
      initialise_prior, VB_iteration, VB_posterior,
)


@HOOKS.register_module()
class BCCNetHook(Hook):

    priority = 'NORMAL'

    def __init__(
        self,
        classes: int,
        labellers: int,
        attributes: int=0,
        certain_classes: int=0,
        certain_labellers: int=0,
        alpha_diag_prior: float=0.1,
        use_confidence: bool=True,
    ):
        self.uncertain_classes = classes - certain_classes
        self.uncertain_labellers = labellers - certain_labellers
        self.use_confidence = use_confidence

        self.prior_class_cms = initialise_prior(
            self.uncertain_classes, self.uncertain_labellers, alpha_diag_prior,
        )
        self.variational_class_cms = copy.deepcopy(self.prior_class_cms)

        self.use_attributes = attributes > 0
        if self.use_attributes:
            self.prior_attributes_cms = [
                initialise_prior(2, labellers, alpha_diag_prior)
                for _ in range(attributes)
            ]
            self.variational_attributes_cms = copy.deepcopy(self.prior_attributes_cms)

    def before_run(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model

    def before_train_epoch(self, runner) -> None:
        """Check the begin_epoch/iter is smaller than max_epochs/iters.

        Args:
            runner (Runner): The runner of the training process.
        """
        # determine all annotators' labels and current model predictions
        ori_pipeline = runner.train_dataloader.dataset.pipeline
        runner.train_dataloader.dataset.pipeline = [runner.val_dataloader.dataset.pipeline]

        self.imgpath2idx = {}
        idx = 0
        class_scores, attribute_labels = [], []
        class_preds, attribute_preds = [], []
        for batch in tqdm(
            iterable=runner.train_dataloader,
            desc='Running forward pass for BCCNet',
        ):
            self.src_model.eval()
            with torch.no_grad():
                batch = self.src_model.data_preprocessor(batch)
                feats = self.src_model.extract_feat(batch['inputs'])
                embeddings = torch.stack([torch.tensor(ds.embedding) for ds in batch['data_samples']]).to(feats[-1])
                cls_score = self.src_model.head(feats, embeddings)
            self.src_model.train()

            if self.use_attributes:
                attribute_preds.append(cls_score[0])
                class_preds.append(cls_score[1])
            else:
                class_preds.append(cls_score)

            for result in batch['data_samples']:
                self.imgpath2idx[result.img_path] = idx
                idx += 1

                class_scores.append(result.gt_score)
                if self.use_attributes:
                    attribute_labels.append(torch.from_numpy(result.gt_attrs).to(result.gt_label))

        runner.train_dataloader.dataset.pipeline = ori_pipeline


        # determine posterior ground-truth class probabilities
        class_labels = torch.stack(class_scores)
        if not self.use_confidence:
            class_labels = class_labels.ceil().long()
        class_preds = torch.cat(class_preds)

        keep = torch.any(class_labels[:, :self.uncertain_labellers] > -1, dim=1)
        posterior, self.variational_class_cms = VB_iteration(
            class_labels[keep, :self.uncertain_labellers],
            class_preds[keep, :self.uncertain_classes],
            self.variational_class_cms, self.prior_class_cms,
        )

        self.class_posterior = torch.zeros_like(class_preds)
        self.class_posterior[keep, :self.uncertain_classes] = posterior
        if not torch.all(keep):
            self.class_posterior[~keep, class_labels[~keep, self.uncertain_labellers]] = 1

        for i, cm in enumerate(
            np.transpose(self.variational_class_cms, (2, 0, 1)),
        ):
            f1 = 2*cm[1, 1] / (2*cm[1, 1] + cm[0, 1] + cm[1, 0])
            MMLogger.get_current_instance().info(f'Labeller {i} F1={f1:.3f}')

        # determine posterior ground-truth attribute probabilities
        if not self.use_attributes:
            return
        
        attribute_labels = torch.stack(attribute_labels)
        attribute_preds = torch.cat(attribute_preds)
        posteriors = torch.zeros((attribute_labels.shape[0], 0)).to(self.class_posterior)
        for i, (prior_cms, variational_cms) in enumerate(zip(
            self.prior_attributes_cms, self.variational_attributes_cms,
        )):
            labels = attribute_labels[:, :, i]
            preds = torch.stack((
                -attribute_preds[:, i], attribute_preds[:, i],
            ), axis=1)

            posterior, self.variational_attributes_cms[i] = VB_iteration(
                labels, preds, variational_cms, prior_cms,
            )
            posteriors = torch.column_stack((posteriors, posterior[:, 1]))
            
        self.attribute_posteriors = posteriors

    def before_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None) -> None:
        for data_sample in data_batch['data_samples']:
            idx = self.imgpath2idx[data_sample.img_path]
            data_sample.set_field(self.class_posterior.shape[1], 'num_classes', field_type='metainfo')
            data_sample.set_gt_label(self.class_posterior[idx].argmax())         
            data_sample.set_gt_score(self.class_posterior[idx])            
            if self.use_attributes:
                data_sample.set_field(self.attribute_posteriors[idx], 'gt_attrs')

    def before_val_epoch(self, runner) -> None:
        """Check the begin_epoch/iter is smaller than max_epochs/iters.

        Args:
            runner (Runner): The runner of the training process.
        """
        # determine all annotators' labels and current model predictions
        self.imgpath2idx = {}
        idx = 0
        class_scores, attribute_labels = [], []
        for batch in tqdm(
            iterable=runner.val_dataloader,
            desc='Running forward pass for BCCNet',
        ):
            for result in batch['data_samples']:
                self.imgpath2idx[result.img_path] = idx
                idx += 1

                class_scores.append(result.gt_score)
                if self.use_attributes:
                    attribute_labels.append(torch.from_numpy(result.gt_attrs).to(result.gt_label))


        # determine posterior ground-truth class probabilities
        class_labels = torch.stack(class_scores)
        if not self.use_confidence:
            class_labels = class_labels.ceil().long()

        keep = torch.any(class_labels[:, :self.uncertain_labellers] > -1, dim=1)
        posterior = VB_posterior(
            class_labels[keep, :self.uncertain_labellers],
            self.variational_class_cms,
        )

        self.class_posterior = torch.zeros(class_labels.shape[0], self.variational_class_cms.shape[0]).to(posterior)
        self.class_posterior[keep, :self.uncertain_classes] = posterior
        if not torch.all(keep):
            self.class_posterior[~keep, class_labels[~keep, self.uncertain_labellers]] = 1

        # determine posterior ground-truth attribute probabilities
        if not self.use_attributes:
            return
        
        attribute_labels = torch.stack(attribute_labels)
        posteriors = torch.zeros((attribute_labels.shape[0], 0)).to(self.class_posterior)
        for i, variational_cms in enumerate(self.variational_attributes_cms):
            labels = attribute_labels[:, :, i]

            posterior = VB_posterior(
                labels, variational_cms,
            )
            posteriors = torch.column_stack((posteriors, posterior[:, 1]))
            
        self.attribute_posteriors = posteriors

    def before_val_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None) -> None:
        for data_sample in data_batch['data_samples']:
            idx = self.imgpath2idx[data_sample.img_path]
            data_sample.set_field(self.class_posterior.shape[1], 'num_classes', field_type='metainfo')
            data_sample.set_gt_label(self.class_posterior[idx].argmax())
            data_sample.set_gt_score(self.class_posterior[idx])
            if self.use_attributes:
                data_sample.set_field(self.attribute_posteriors[idx], 'gt_attrs')

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        checkpoint['class_cms'] = self.variational_class_cms
        if self.use_attributes:
            checkpoint['attributes_cms'] = self.variational_attributes_cms
