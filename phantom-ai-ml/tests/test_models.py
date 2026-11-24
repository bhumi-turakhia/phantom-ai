import torch
from models.nlp_model import BiLSTMAttributeLabeler
from models.uncertainty_model import EvidentialUncertaintyClassifier
from models.cnn_model import ForensicFeatureCNN
from models.dcgan_model import ConditionalDCGANGenerator, ConditionalDCGANDiscriminator
from utils.device import resolve_device


def test_nlp_bilstm_forward_pass():
    device = resolve_device()
    vocab_size = 50
    num_tags = 15
    seq_len = 8
    batch_size = 2

    model = BiLSTMAttributeLabeler(vocab_size=vocab_size, num_tags=num_tags, hidden_dim=64)
    model.to(device)

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    lengths = torch.tensor([8, 6])

    emissions = model(dummy_input, lengths=lengths)
    assert emissions.shape == (batch_size, seq_len, num_tags)
    assert model.count_parameters() > 0


def test_uncertainty_classifier_forward_pass():
    device = resolve_device()
    batch_size = 3
    input_dim = 128
    num_categories = 7

    model = EvidentialUncertaintyClassifier(input_dim=input_dim, hidden_dim=64, num_categories=num_categories)
    model.to(device)

    dummy_feats = torch.randn(batch_size, input_dim, device=device)
    probs, uncertainty, alphas = model(dummy_feats)

    assert probs.shape == (batch_size, num_categories, 4)
    assert uncertainty.shape == (batch_size, num_categories)
    assert alphas.shape == (batch_size, num_categories, 4)

    # Probabilities along states should sum to ~1.0
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)

    # Uncertainty in (0, 1]
    assert (uncertainty > 0.0).all()
    assert (uncertainty <= 1.0).all()


def test_cnn_feature_extractor_forward_pass():
    device = resolve_device()
    batch_size = 2
    model = ForensicFeatureCNN(in_channels=3, feature_dim=128, image_size=64)
    model.to(device)

    dummy_img = torch.randn(batch_size, 3, 64, 64, device=device)
    embeddings = model(dummy_img)

    assert embeddings.shape == (batch_size, 128)
    assert model.count_parameters() > 0


def test_dcgan_generator_and_discriminator_forward_pass():
    device = resolve_device()
    batch_size = 2
    latent_dim = 100
    cond_dim = 32

    netG = ConditionalDCGANGenerator(latent_dim=latent_dim, condition_dim=cond_dim, image_size=64).to(device)
    netD = ConditionalDCGANDiscriminator(in_channels=3, condition_dim=cond_dim, image_size=64).to(device)

    noise = torch.randn(batch_size, latent_dim, device=device)
    condition = torch.randn(batch_size, cond_dim, device=device)

    fake_images = netG(noise, condition)
    assert fake_images.shape == (batch_size, 3, 64, 64)

    # Values in [-1, 1] from Tanh
    assert fake_images.min() >= -1.0
    assert fake_images.max() <= 1.0

    validity = netD(fake_images, condition)
    assert validity.shape == (batch_size, 1)
    # Probabilities in [0, 1] from Sigmoid
    assert validity.min() >= 0.0
    assert validity.max() <= 1.0

