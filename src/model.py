
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_encoder(model_name):
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    is_t5 = getattr(config, "model_type", "") == "t5" or "t5" in str(model_name).lower()
    if is_t5:
        from transformers import T5EncoderModel
        return T5EncoderModel.from_pretrained(model_name)
    return AutoModel.from_pretrained(model_name)


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class FewShotEncoder(nn.Module):

    def __init__(self, model_name, center=False, center_momentum=0.9,
                 whiten=False, whiten_eps=0.1):
        super().__init__()
        self.encoder = build_encoder(model_name)
        self.center = center
        self.center_momentum = center_momentum
        self.whiten = whiten
        self.whiten_eps = whiten_eps
        if center or whiten:
            hidden = self.encoder.config.hidden_size
            self.register_buffer("running_mean", torch.zeros(hidden))
            self.register_buffer("center_initialized", torch.tensor(False))

    def _whiten(self, emb):
        orig_dtype = emb.dtype
        x32 = emb.float()
        mu = x32.mean(dim=0, keepdim=True)
        x = x32 - mu
        n = x.shape[0]
        cov = (x.t() @ x) / max(n - 1, 1)
        cov = cov + self.whiten_eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals.clamp(min=1e-6))) @ eigvecs.t()
            whitened = x @ inv_sqrt
        except Exception:
            whitened = x
        return whitened.to(orig_dtype)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = mean_pool(out.last_hidden_state, attention_mask)
        if self.whiten:
            emb = self._whiten(emb)
        elif self.center:
            emb = emb - emb.mean(dim=0, keepdim=True)
        return emb


def compute_prototypes(support_emb, support_label, n_way):
    d = support_emb.size(1)
    protos = support_emb.new_zeros(n_way, d)
    for c in range(n_way):
        mask = (support_label == c)
        if mask.any():
            protos[c] = support_emb[mask].mean(dim=0)
    return protos


def prototype_ce_loss(query_emb, query_label, protos):
    dist2 = ((query_emb.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=2)
    logits = -dist2
    loss = F.cross_entropy(logits, query_label)
    preds = logits.argmax(dim=1)
    return loss, logits, preds


def contrastive_loss(embeddings, labels, temperature=0.5, numerator="P"):
    device = embeddings.device
    feats = F.normalize(embeddings, dim=1)
    sim = feats @ feats.T / temperature                 # [M, M]
    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - logits_max.detach()

    m = labels.size(0)
    self_mask = torch.eye(m, dtype=torch.bool, device=device)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)  # P(i)
    diff = (labels.unsqueeze(0) != labels.unsqueeze(1))                 # N(i)

    exp_sim = torch.exp(sim) * (~self_mask)
    denom = exp_sim.sum(dim=1) + 1e-12

    num_mask = same if numerator == "P" else diff
    numer = (torch.exp(sim) * num_mask.float()).sum(dim=1) + 1e-12

    valid = num_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return embeddings.new_tensor(0.0)
    loss = -torch.log(numer / denom)[valid]
    return loss.mean()


def supcon_loss(embeddings, labels, temperature=0.5):
    return contrastive_loss(embeddings, labels, temperature, numerator="P")


def episode_losses(model, support_ids, support_mask, support_label,
                   query_ids, query_mask, query_label,
                   n_way, cl_weight=1.0, temperature=0.5, cl_numerator="P"):
    support_emb = model(support_ids, support_mask)
    query_emb = model(query_ids, query_mask)

    protos = compute_prototypes(support_emb, support_label, n_way)
    loss_ce, _, preds = prototype_ce_loss(query_emb, query_label, protos)

    if cl_weight and cl_weight > 0:
        all_emb = torch.cat([support_emb, query_emb], dim=0)
        all_lab = torch.cat([support_label, query_label], dim=0)
        loss_cl = contrastive_loss(all_emb, all_lab, temperature, numerator=cl_numerator)
    else:
        loss_cl = query_emb.new_tensor(0.0)

    loss = loss_ce + cl_weight * loss_cl
    return {
        "loss": loss,
        "loss_ce": loss_ce.detach(),
        "loss_cl": loss_cl.detach(),
        "preds": preds.detach(),
        "query_label": query_label,
    }
