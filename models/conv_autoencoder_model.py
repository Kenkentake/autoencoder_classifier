import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from pytorch_lightning import LightningModule

from optim import get_optimizer, get_scheduler

from utils import save_imgs, save_confusion_matrix


class ConvAutoEncoderCLFModel(LightningModule):
    def __init__(
        self,
        args,
        device: str,
        hparams: dict,
        in_channel: int,
        out_channel: int,
        trial,
        run_id: str,
        tmp_results_dir: str
    ) -> None:
        super(ConvAutoEncoderCLFModel, self).__init__()
        self.args = args
        self._device = device
        self.hparams = hparams
        self.trial = trial
        self.run_id = run_id
        self.tmp_results_dir = tmp_results_dir

        self.weight = self.args.TRAIN.LOSS_WEIGHT
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # encoder
        self.encoder = nn.Sequential(
                            nn.Conv2d(in_channel, 16, 3, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(16, 4, 3, padding=1),
                            nn.BatchNorm2d(4),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2)
                        )
        # decoder
        self.decoder = nn.Sequential(
                            nn.ConvTranspose2d(4, 16, 2, stride=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(16, 3, 2, stride=2),
                            nn.Sigmoid()
                        )
        # classifier
        self.classifier = nn.Sequential(
                            nn.Linear(4 * 8 * 8, out_channel),
                            nn.Dropout(0.2)
                        )

    def configure_optimizers(self) -> list:
        if self.trial is not None:
            lr = self.trial.suggest_loguniform('optimizer_lr', 1e-5, 1e-1)
            self.logger.log_hyperparams({'optimizer_lr': lr})
        else:
            lr = self.args.TRAIN.LR
        optimizer = get_optimizer(self.args, self, lr=lr)
        scheduler = get_scheduler(self.args, optimizer)

        return [optimizer], [scheduler]

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        clf_in = encoded.view(-1, 4 * 8 * 8)
        clf_out = self.classifier(clf_in)

        return clf_out, decoded

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, decoded = self(inputs)
        ae_loss = self.mse_loss(decoded, inputs)
        clf_loss = self.cross_entropy_loss(outputs, labels)
        loss = self.weight[0] * ae_loss + self.weight[1] * clf_loss
        accuracy = (outputs.argmax(1) == labels).sum().item()

        return OrderedDict({
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss,
            'ae_loss': ae_loss,
            'clf_loss': clf_loss
        })

    def training_epoch_end(self, outputs) -> None:
        accuracy = loss = ae_loss = clf_loss = 0.0
        count = 0
        for output in outputs:
            accuracy += output['accuracy']
            loss += output['loss'].data.item()
            ae_loss += output['ae_loss'].data.item()
            clf_loss += output['clf_loss'].data.item()
            count += output['count']

        results = {
            'training_accuracy': accuracy / count,
            'training_loss': loss / count,
            'training_ae_loss': ae_loss / count,
            'training_clf_loss': clf_loss / count
        }

        self.logger.log_metrics(results, step=self.current_epoch)

        return None

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, decoded = self(inputs)

        ae_loss = self.mse_loss(decoded, inputs)
        clf_loss = self.cross_entropy_loss(outputs, labels)
        loss = self.weight[0] * ae_loss + self.weight[1] * clf_loss

        accuracy = (outputs.argmax(1) == labels).sum().item()

        return OrderedDict({
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss,
            'ae_loss': ae_loss,
            'clf_loss': clf_loss
        })

    def validation_epoch_end(self, outputs) -> dict:
        accuracy = loss = ae_loss = clf_loss = 0.0
        count = 0
        for output in outputs:
            accuracy += output['accuracy']
            loss += output['loss'].data.item()
            ae_loss += output['ae_loss'].data.item()
            clf_loss += output['clf_loss'].data.item()
            count += output['count']

        results = {
            'validation_accuracy': accuracy / count,
            'validation_loss': loss / count,
            'validation_ae_loss': ae_loss / count,
            'validation_clf_loss': clf_loss / count
        }

        self.logger.log_metrics(results, step=self.current_epoch)

        return results

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, decoded = self(inputs)

        # save input and decoded img
        if batch_idx < 5 :
            save_imgs(batch_idx, inputs[:5], decoded[:5], self.run_id, self.tmp_results_dir)

        ae_loss = self.mse_loss(decoded, inputs)
        clf_loss = self.cross_entropy_loss(outputs, labels)
        loss = self.weight[0] * ae_loss + self.weight[1] * clf_loss

        accuracy = (outputs.argmax(1) == labels).sum().item()

        # calc indivisual class accuracy
        class_correct = list(0 for i in range(10))
        class_counter = list(0 for i in range(10))
        is_correct = (outputs.argmax(1) == labels).squeeze()
        for i, pred, label in zip(list(range(len(outputs))), outputs, labels):
            class_correct[label] += is_correct[i].item()
            class_counter[label] += 1

        return OrderedDict({
            'preds': outputs.argmax(1),
            'labels': labels,
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss,
            'ae_loss': ae_loss,
            'clf_loss': clf_loss,
            'class_correct': class_correct,
            'class_counter': class_counter
        })
            
    def test_epoch_end(self, outputs) -> dict:
        accuracy = loss = ae_loss = clf_loss = 0.0
        count = 0
        all_class_correct = list(0 for i in range(10))
        all_class_counter = list(0 for i in range(10))
        preds_all = []
        labels_all = [] 

        for output in outputs:
            preds_all.extend(output['preds'].tolist())
            labels_all.extend(output['labels'].tolist())
            accuracy += output['accuracy']
            loss += output['loss'].data.item()
            ae_loss += output['ae_loss'].data.item()
            clf_loss += output['clf_loss'].data.item()
            count += output['count']
            for i in range(10):
                class_correct = output['class_correct']
                class_counter = output['class_counter']
                all_class_correct = [x + y for x, y in zip(all_class_correct, class_correct)]
                all_class_counter = [x + y for x, y in zip(all_class_counter, class_counter)]
        save_confusion_matrix(labels_all, preds_all, self.args.DATA.CLASSES, self.run_id, self.tmp_results_dir)
        class_accuracy = {'test_accuracy_{}'.format(i): x/y for i, x, y in zip(list(range(10)), all_class_correct, all_class_counter)} 
        results = {
            'test_mean_accuracy': accuracy / count,
            'test_loss': loss / count,
            'test_ae_loss': ae_loss / count,
            'test_clf_loss': clf_loss / count
        }
        results.update(class_accuracy)
        self.logger.log_metrics(results, step=self.current_epoch)

        return results

    
