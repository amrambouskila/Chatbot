from train_chatbot import get_or_build_tokenizer
from generic_transformer import build_transformer, Transformer
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch


class TransformerGAN(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config['learning_rate']
        self.config = config
        self.tokenizer = get_or_build_tokenizer(config=config)
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.generator = build_transformer(self.vocab_size, config['seq_len'], config['d_model'], config['N'],
                                           config['h'], config['dropout'], config['d_ff'])
        self.discriminator = self.build_discriminator()
        self.automatic_optimization = False

    def build_discriminator(self):
        # Discriminator is another Transformer model or a separate model based on transformer
        return Transformer(
            encoder=self.generator.encoder,
            decoder=self.generator.decoder,
            src_embed=self.generator.src_embed,
            tgt_embed=self.generator.tgt_embed,
            src_pos=self.generator.src_pos,
            tgt_pos=self.generator.tgt_pos,
            projection_layer=nn.Linear(self.generator.projection_layer.in_features, 1)  # Single output for real/fake
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Use generator to produce output
        return self.generator(src, tgt, src_mask, tgt_mask)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx):
        # Extract data from data loader
        src, tgt = batch['encoder_input'], batch['decoder_input']
        src_mask, tgt_mask = batch['encoder_mask'], batch['decoder_mask']

        # Access optimizers
        opt_g, opt_d = self.optimizers()

        # Sample noise
        z = torch.randn(src.size(0), self.config['seq_len'])
        z = z.type_as(src)

        # Zero out the generator gradient
        opt_g.zero_grad()

        # Generate fake responses
        generated_responses = self(z, tgt, src_mask, tgt_mask)

        # Train generator
        y_hat = self.discriminator(generated_responses, tgt, src_mask, tgt_mask)
        y = torch.ones(y_hat.size(0), 1)
        y = y.type_as(src)
        g_loss = self.adversarial_loss(y_hat, y)
        self.manual_backward(g_loss)
        opt_g.step()

        # Train discriminator
        opt_d.zero_grad()
        y_hat_real = self.discriminator(src, tgt, src_mask, tgt_mask)
        y_real = torch.ones(src.size(0), 1)
        y_real = y_real.type_as(src)
        real_loss = self.adversarial_loss(y_hat_real, y_real)

        y_hat_fake = self.discriminator(generated_responses.detach())
        y_fake = torch.zeros(src.size(0), 1)
        y_fake = y_fake.type_as(src)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()

        log_dict = {"g_loss": g_loss, "d_loss": d_loss}
        self.log_dict(log_dict)

    def validation_step(self, batch, batch_idx):
        # Extract data from data loader
        src, tgt = batch['encoder_input'], batch['decoder_input']
        src_mask, tgt_mask = batch['encoder_mask'], batch['decoder_mask']

        # Sample noise
        z = torch.randn(src.size(0), self.config['seq_len'])
        z = z.type_as(src)

        # Generate fake responses
        generated_responses = self(z, tgt, src_mask, tgt_mask)

        # Compute discriminator loss on real data
        y_hat_real = self.discriminator(src, tgt, src_mask, tgt_mask)
        y_real = torch.ones(y_hat_real.size(0), 1)
        y_real = y_real.type_as(src)
        real_loss = self.adversarial_loss(y_hat_real, y_real)

        # Compute discriminator loss on fake data
        y_hat_fake = self.discriminator(generated_responses.detach(), tgt, src_mask, tgt_mask)
        y_fake = torch.zeros(y_hat_fake.size(0), 1)
        y_fake = y_fake.type_as(src)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        # Compute generator loss
        y = torch.ones(generated_responses.size(0), 1)
        y = y.type_as(src)
        g_loss = self.adversarial_loss(generated_responses, y)

        self.log('val_g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'val_g_loss': g_loss, 'val_d_loss': d_loss}

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # Define the learning rate scheduler for generator
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=100)

        # Define the learning rate scheduler for discriminator
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=100)

        return [opt_g, opt_d], [scheduler_g, scheduler_d]
