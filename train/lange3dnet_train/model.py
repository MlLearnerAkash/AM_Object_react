"""
LangGeoNetV2: Dual-backbone language-guided geodesic cost predictor.

Architecture:
  1. Dual visual backbone: CLIP ViT-B/16 (semantic) + DINOv2-small (spatial/depth).
     Last 4 CLIP blocks and last 2 DINOv2 blocks are trainable.
  2. Background-subtracted masked pooling on both backbones.
  3. Cross-modal transformer: objects cross-attend to language.
  4. Cost head: predicts per-object normalised cost in [0, 1].

forward(images, masks_list, input_ids, attention_mask) ->
    (predictions: list[B] of [K_b] tensors,
     attn_weights_all: list of cross-attn weight tensors)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, Dinov2Model


# ── Normalisation constants ────────────────────────────────────────────────────
# Convert CLIP-preprocessed images → DINOv2-preprocessed images inline.
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_DINO_MEAN = [0.485,      0.456,      0.406      ]
_DINO_STD  = [0.229,      0.224,      0.225      ]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: background-subtracted masked average pooling
# ─────────────────────────────────────────────────────────────────────────────

def _bg_sub_masked_pool(feature_map: torch.Tensor,
                        masks_list: list) -> list:
    """
    For each batch item, pool per-object features then subtract the scene mean.

    Args:
        feature_map : [B, N_patches, D]  (square patch grid: N = H_f * W_f)
        masks_list  : list[B] of [K_b, H, W] bool tensors

    Returns:
        list[B] of [K_b, D] background-subtracted object features.
    """
    B = feature_map.shape[0]
    results = []
    for b in range(B):
        mask_b = masks_list[b]          # [K, H, W]
        K = mask_b.shape[0]
        if K == 0:
            results.append(torch.zeros(0, feature_map.shape[-1],
                                       device=feature_map.device))
            continue
        feat = feature_map[b]           # [N, D]
        N, D = feat.shape
        H_f = W_f = int(N ** 0.5)
        feat_flat = feat.reshape(-1, D)                              # [H_f*W_f, D]
        masks_r = F.interpolate(
            mask_b.unsqueeze(1).float(),
            size=(H_f, W_f),
            mode='nearest',
        ).squeeze(1).reshape(K, -1)                                  # [K, H_f*W_f]
        mask_sum = masks_r.sum(1, keepdim=True).clamp(min=1e-6)
        pooled   = torch.mm(masks_r, feat_flat) / mask_sum           # [K, D]
        scene_mn = feat_flat.mean(0, keepdim=True)                   # [1, D]
        results.append(pooled - scene_mn)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Module 1: FiLM Layer
# ─────────────────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Given an instruction vector, produce per-channel scale (gamma) and shift
    (beta) to modulate object features before the transformer.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj  = nn.Linear(d_model, d_model)

    def forward(self, instr_vec: torch.Tensor):
        """instr_vec : [B, d]  ->  gamma [B, d], beta [B, d]"""
        return self.gamma_proj(instr_vec), self.beta_proj(instr_vec)


# ─────────────────────────────────────────────────────────────────────────────
# Module 2: Cross-Modal Transformer Layer
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalTransformerLayer(nn.Module):
    """
    Objects cross-attend to language tokens and exchange information
    via a feed-forward layer.
    """

    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query, context, query_mask=None, context_mask=None):
        c_pad = (~context_mask) if context_mask is not None else None
        x_cross, attn_w = self.cross_attn(query, context, context,
                                           key_padding_mask=c_pad)
        x = self.norm2(query + self.drop2(x_cross))
        x = self.norm3(x + self.ffn(x))
        return x, attn_w


# ─────────────────────────────────────────────────────────────────────────────
# Module 3: Listwise Rank Refinement
# ─────────────────────────────────────────────────────────────────────────────

class RankRefinementLayer(nn.Module):
    """
    Listwise score refinement: objects re-score themselves relative to each
    other, conditioned on their initial pointwise scores and the global
    instruction.

    Inputs:
        x           : [B, K, d]  — post-transformer object features
        init_scores : [B, K, 1]  — initial sigmoid scores from cost_head
        instr_vec   : [B, d]     — pooled instruction
        obj_mask    : [B, K]     — True = valid object
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.score_embed = nn.Linear(1, d_model)
        self.instr_proj  = nn.Linear(d_model, d_model)
        self.self_attn   = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:           torch.Tensor,
        init_scores: torch.Tensor,
        instr_vec:   torch.Tensor,
        obj_mask=None,
    ) -> torch.Tensor:
        score_ctx = self.score_embed(init_scores)           # [B, K, d]
        instr_ctx = self.instr_proj(instr_vec).unsqueeze(1) # [B, 1, d]
        x_aug     = x + score_ctx + instr_ctx               # [B, K, d]
        pad_mask  = (~obj_mask) if obj_mask is not None else None
        x_sa, _   = self.self_attn(x_aug, x_aug, x_aug, key_padding_mask=pad_mask)
        return self.norm(x_aug + self.drop(x_sa))


# ─────────────────────────────────────────────────────────────────────────────
# Module 4: LangGeoNetV2
# ─────────────────────────────────────────────────────────────────────────────

class LangGeoNetV2(nn.Module):
    """
    Dual-backbone (CLIP ViT-B/16 + DINOv2-small) language-guided geodesic
    cost predictor.

    Peak VRAM: ~10-15 GB at batch=32 on a single GPU.  Fits within 32 GB.

    Trainable parameter groups:
        CLIP visual : last 4 transformer blocks + post_layernorm
        CLIP text   : last 4 transformer blocks + final_layer_norm
        DINOv2      : last 2 encoder blocks + layernorm
        All new heads (obj_proj, transformer_layers, cost_head)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        dino_model_name: str = "facebook/dinov2-small",
        freeze_clip: bool = True,
        freeze_dino: bool = True,
        max_objects: int = 50,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_objects = max_objects

        # ── Backbone 1: CLIP (visual + text) ─────────────────────────────────
        self.clip        = CLIPModel.from_pretrained(clip_model_name)
        clip_vis_dim     = self.clip.config.vision_config.hidden_size   # 768
        clip_txt_dim     = self.clip.config.text_config.hidden_size     # 512

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            for layer in self.clip.vision_model.encoder.layers[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            for layer in self.clip.text_model.encoder.layers[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.text_model.final_layer_norm.parameters():
                p.requires_grad = True

        # ── Backbone 2: DINOv2-small (spatial / depth features) ──────────────
        self.dino    = Dinov2Model.from_pretrained(dino_model_name)
        dino_dim     = self.dino.config.hidden_size                     # 384

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False
            for layer in self.dino.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.dino.layernorm.parameters():
                p.requires_grad = True

        # Normalisation buffers survive .to(device) / state_dict saves
        self.register_buffer('_clip_mean',
                             torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_clip_std',
                             torch.tensor(_CLIP_STD ).view(1, 3, 1, 1))
        self.register_buffer('_dino_mean',
                             torch.tensor(_DINO_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_dino_std',
                             torch.tensor(_DINO_STD ).view(1, 3, 1, 1))

        # ── Projections ───────────────────────────────────────────────────────
        self.clip_vis_proj = nn.Linear(clip_vis_dim, d_model)
        self.dino_vis_proj = nn.Linear(dino_dim,      d_model)
        self.lang_proj     = nn.Linear(clip_txt_dim,  d_model)

        # Combined object projection: clip_obj(d) + dino_obj(d) -> d
        self.obj_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # ── Type + positional embeddings ──────────────────────────────────────
        self.obj_type_embed  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.obj_pos_embed   = nn.Parameter(
            torch.randn(1, max_objects, d_model) * 0.02
        )

        # ── FiLM conditioning: per-object attended instruction ─────────────────
        # Each object attends to all language tokens → per-object gamma/beta.
        self.film_layer        = FiLMLayer(d_model)
        self.instr_to_obj_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.instr_to_obj_norm = nn.LayerNorm(d_model)

        # ── Cross-modal transformer ───────────────────────────────────────────
        self.transformer_layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Instruction pooling ───────────────────────────────────────────────
        self.instruction_pool_attn = nn.Linear(d_model, 1)

        # ── Per-object language context (post-transformer) ────────────────────
        # Each object queries language tokens after the transformer to get
        # its own relevant language context for the cost head.
        self.obj_to_lang_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.obj_to_lang_norm = nn.LayerNorm(d_model)

        # ── Cost head ─────────────────────────────────────────────────────────
        # input: obj [d] | per-obj lang context [d] | interaction [d]  -> 3d
        # Outputs raw logits — sigmoid applied outside (in losses / inference).
        self.cost_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # ── Listwise ranking refinement ───────────────────────────────────────
        # Outputs raw logits — sigmoid applied outside.
        self.rank_refine = RankRefinementLayer(d_model, n_heads, dropout)
        self.rank_head   = nn.Sequential(nn.Linear(d_model, 1))


    # ── Internal helpers ───────────────────────────────────────────────────────

    def _renorm_for_dino(self, images: torch.Tensor) -> torch.Tensor:
        """Convert CLIP-normalised images to DINOv2-normalised images."""
        rgb01 = images * self._clip_std + self._clip_mean   # -> [0, 1]
        return (rgb01 - self._dino_mean) / self._dino_std

    def encode_visual_clip(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP ViT-B/16 -> [B, 196, d_model] patch features."""
        grad_on = any(p.requires_grad
                      for p in self.clip.vision_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.clip.vision_model(pixel_values=images)
            patches = out.last_hidden_state[:, 1:, :]   # drop CLS  [B, 196, 768]
        return self.clip_vis_proj(patches)

    def encode_visual_dino(self, images: torch.Tensor) -> torch.Tensor:
        """DINOv2-small -> [B, 256, d_model] patch features (224x224 input)."""
        images_dino = self._renorm_for_dino(images)
        grad_on = any(p.requires_grad for p in self.dino.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.dino(pixel_values=images_dino,
                            interpolate_pos_encoding=True)
            patches = out.last_hidden_state[:, 1:, :]   # drop CLS  [B, 256, 384]
        return self.dino_vis_proj(patches)

    def encode_language(self, input_ids, attention_mask):
        """CLIP text -> [B, L, d_model] + bool mask."""
        grad_on = any(p.requires_grad
                      for p in self.clip.text_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.clip.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask)
        return self.lang_proj(out.last_hidden_state), attention_mask.bool()

    def pool_instruction(self, lang_feats, lang_mask):
        """Attention-weighted pooling -> [B, d_model]."""
        scores = self.instruction_pool_attn(lang_feats).squeeze(-1)
        scores = scores.masked_fill(~lang_mask, -1e9)
        return torch.bmm(F.softmax(scores, -1).unsqueeze(1),
                         lang_feats).squeeze(1)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, images, masks_list, input_ids, attention_mask,
                return_geo: bool = False):
        """
        Args:
            images         : [B, 3, 224, 224]  CLIP-preprocessed
            masks_list     : list[B] of [K_b, H, W] bool tensors
            input_ids      : [B, L]
            attention_mask : [B, L]
            return_geo     : if True, also return pre-refinement geo_preds

        Returns:
            predictions      : list[B] of [K_b] raw logit tensors
            (geo_preds)      : list[B] of [K_b] raw logits from cost_head
                               (only when return_geo=True)
            attn_weights_all : list of [B, K_max, L] cross-attn weight tensors
        """
        B      = images.shape[0]
        device = images.device

        # ── Visual encoding ────────────────────────────────────────────────────
        clip_feats = self.encode_visual_clip(images)    # [B, 196, d]
        dino_feats = self.encode_visual_dino(images)    # [B, 256, d]

        # ── Language encoding ──────────────────────────────────────────────────
        lang_feats, lang_mask = self.encode_language(input_ids, attention_mask)
        instr_vec = self.pool_instruction(lang_feats, lang_mask)  # [B, d]

        # ── Per-object pooling (background-subtracted, both backbones) ─────────
        clip_obj_list = _bg_sub_masked_pool(clip_feats, masks_list)
        dino_obj_list = _bg_sub_masked_pool(dino_feats, masks_list)

        # ── Per-object feature projection ──────────────────────────────────────
        K_counts = [masks_list[b].shape[0] for b in range(B)]
        K_max    = max(K_counts) if K_counts else 1

        obj_feats_list = []
        for b in range(B):
            K_b = K_counts[b]
            if K_b == 0:
                obj_feats_list.append(
                    torch.zeros(0, self.d_model, device=device))
                continue
            combined = torch.cat(
                [clip_obj_list[b], dino_obj_list[b]], dim=-1
            )                                           # [K_b, 2d]
            obj_feats_list.append(self.obj_proj(combined))    # [K_b, d]

        # ── Pad to [B, K_max, d] ──────────────────────────────────────────────
        obj_padded = torch.zeros(B, K_max, self.d_model, device=device)
        obj_mask   = torch.zeros(B, K_max, dtype=torch.bool, device=device)

        for b in range(B):
            K_c = min(K_counts[b], K_max)
            if K_c > 0:
                obj_padded[b, :K_c] = obj_feats_list[b][:K_c]
                obj_mask[b,   :K_c] = True

        # ── Type + positional embeddings ───────────────────────────────────────
        if K_max <= self.max_objects:
            pos_emb = self.obj_pos_embed[:, :K_max]
        else:
            pos_emb = F.interpolate(
                self.obj_pos_embed.permute(0, 2, 1),
                size=K_max, mode='linear', align_corners=False,
            ).permute(0, 2, 1)

        x           = obj_padded + self.obj_type_embed + pos_emb
        lang_tokens = lang_feats + self.lang_type_embed
        lang_pad    = ~lang_mask                                 # [B, L] True=padding

        # ── FiLM: per-object attended instruction modulates features ───────────
        # Each object attends to ALL language tokens → unique gamma/beta per obj.
        film_ctx, _ = self.instr_to_obj_attn(
            x, lang_feats, lang_feats,
            key_padding_mask=lang_pad,
        )                                                        # [B, K_max, d]
        film_ctx    = self.instr_to_obj_norm(film_ctx)          # pure lang — no x residual
        gamma, beta = self.film_layer(film_ctx)                  # [B, K_max, d]
        x = x * (1 + gamma) + beta

        # ── Cross-modal transformer (objects <-> language) ─────────────────────
        attn_weights_all = []
        for layer in self.transformer_layers:
            x, attn_w = layer(x, lang_tokens, obj_mask, lang_mask)
            attn_weights_all.append(attn_w)

        per_obj_lang, _ = self.obj_to_lang_attn(
            x, lang_feats, lang_feats,
            key_padding_mask=lang_pad,
        )                                                        # [B, K_max, d]
        # Keep x residual so each object's language context is grounded in its
        # own visual feature — prevents all objects from collapsing to the same
        # language context when backbone features are similar.
        per_obj_lang = self.obj_to_lang_norm(x + per_obj_lang)  # [B, K_max, d]

        obj_lang_product = x * per_obj_lang                      # [B, K_max, d]
        head_in  = torch.cat([x, per_obj_lang, obj_lang_product], dim=-1)  # [B, K_max, 3d]
        geo_pred = self.cost_head(head_in).squeeze(-1)           # [B, K_max]


        rank_ctx   = self.rank_refine(x, geo_pred.unsqueeze(-1), instr_vec, obj_mask)
        final_pred = self.rank_head(rank_ctx).squeeze(-1)        # [B, K_max]

        predictions = [
            final_pred[b, :min(K_counts[b], K_max)] for b in range(B)
        ]
        geo_preds = [
            geo_pred[b, :min(K_counts[b], K_max)] for b in range(B)
        ]

        if return_geo:
            return predictions, geo_preds, attn_weights_all
        return predictions, attn_weights_all
