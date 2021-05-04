import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from pytorch_lightning import LightningModule

from optim import get_optimizer, get_scheduler

from utils import save_imgs, save_confusion_matrix


class ConvAEWithCNNCLFModel(LightningModule):
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
        super(ConvAEWithCNNCLFModel, self).__init__()
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
                            ConvBatchNormRelu(in_channel, 3, 32, 1),
                            ConvBatchNormRelu(32, 3, 32, 1),
                            nn.MaxPool2d(2, 2),
                            ConvBatchNormRelu(32, 3, 64, 1),
                            ConvBatchNormRelu(64, 3, 64, 1),
                            nn.MaxPool2d(2, 2),
                            ConvBatchNormRelu(64, 3, 128, 1)
                        )

        # decoder
        self.decoder = nn.Sequential(
                            nn.ConvTranspose2d(128, 128, 2, stride=2),
                            ConvBatchNormRelu(128, 3, 64, 1),
                            ConvBatchNormRelu(64, 3, 64, 1),
                            nn.ConvTranspose2d(64, 64, 2, stride=2),
                            nn.Conv2d(64, 3, 1),
                            nn.Sigmoid()
                        )
        # classifier
        self.classifier_cnn = nn.Sequential(
                            ConvBatchNormRelu(128, 3, 1028, 1),
                            nn.MaxPool2d(2, 2),
                            nn.Dropout(0.5),
                            ConvBatchNormRelu(1028, 3, 128, 1),
                            nn.MaxPool2d(2, 2),
                            nn.Dropout(0.5)
                        )
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 10)


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
        clf_cnn_out = self.classifier_cnn(encoded)
        fc_in = clf_cnn_out.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(fc_in))
        x = nn.Dropout(0.35)(x)
        x = F.relu(self.fc2(x))
        x = nn.Dropout(0.69)(x)
        clf_out = self.fc3(x)

        return clf_out, decoded

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, decoded = self(inputs)
        ae_loss = self.mse_loss(decoded, inputs)
        if len(self.args.TRAIN.CE_CLASS_WEIGHT) != 0:
            weight = torch.tensor(self.args.TRAIN.CE_CLASS_WEIGHT).to(self._device)
            cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)
            clf_loss = cross_entropy_loss(outputs, labels)
        else:
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

class ConvBatchNormRelu(LightningModule):
    def __init__(self, input_channel, kernel_size, output_channel, padding):
        super(ConvBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
