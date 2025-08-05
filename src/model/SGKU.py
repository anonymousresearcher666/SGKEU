import torch.nn as nn
from src.utilities.utilities import *


class SGKU(nn.Module):
    """Schema-guided Knowledge Unlearning for knowledge graphs."""

    def __init__(self, args, kg, kge_model_class, schema_store=None):
        """
        Initialize SGKU model.

        Args:
            args: Arguments with model configuration
            kg: Knowledge graph instance
            schema_store: Optional schema store for pattern retrieval
        """
        # Initialize
        super(SGKU, self).__init__()

        # Initialize the base KGE model
        self.kge_model_class = kge_model_class  # Assign this BEFORE using it
        self.kge_model = self.kge_model_class(args, kg)

        self.args = args
        self.kg = kg
        self.schema_store = schema_store
        self.huber_loss = nn.HuberLoss(reduction="sum")
        self.entity_distill_mask = None

        # Tracking variables
        self._last_warning_time = 0
        self._loss_counter = 0
        self.kge_model_class = kge_model_class

        # Configure hyperparameters
        self._set_hyperparameters()

    def _set_hyperparameters(self):
        """Set GRPO hyperparameters with defaults."""
        # Main GRPO hyperparameters
        self.args.epsilon_grpo = getattr(self.args, 'epsilon_grpo', 0.2)  # Clipping parameter
        self.args.beta_grpo = getattr(self.args, 'beta_grpo', 0.001)  # KL divergence coefficient
        self.args.group_size_grpo = getattr(self.args, 'group_size_grpo', 128)  # Group size
        self.args.grpo_lambda = getattr(self.args, 'grpo_lambda', 0.5)  # GRPO loss weight
        self.args.preservation_lambda = getattr(self.args, 'preservation_lambda', 0.5)  # Distillation loss weight

        # Grouping strategy for triple organization
        self.args.grouping_strategy = getattr(self.args, 'grouping_strategy', 'relation')

    def save_embeddings(self):
        """Save current embeddings as reference for GRPO and distillation."""
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def embedding(self):
        """Get current embeddings.

        Returns:
            Tuple of (entity_embeddings, relation_embeddings)
        """
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def old_embeddings(self):
        """Get old embeddings for GRPO.

        Returns:
            Tuple of (old_entity_embeddings, old_relation_embeddings)
        """
        old_data_ent_embeddings_weight = None
        old_data_rel_embeddings_weight = None

        for name, value in self.named_buffers():
            if name == "old_data_ent_embeddings_weight":
                old_data_ent_embeddings_weight = value
            elif name == "old_data_rel_embeddings_weight":
                old_data_rel_embeddings_weight = value

        # Create if they don't exist
        if old_data_ent_embeddings_weight is None:
            old_data_ent_embeddings_weight = self.kge_model.ent_embeddings.weight.clone().detach()
        if old_data_rel_embeddings_weight is None:
            old_data_rel_embeddings_weight = self.kge_model.rel_embeddings.weight.clone().detach()

        return old_data_ent_embeddings_weight, old_data_rel_embeddings_weight

    def preservation_loss(self):
        """Calculate knowledge preservation loss between current and old embeddings.

        Returns:
            Tensor: preservation loss value
        """
        # Get current and old embeddings
        ent_embeddings, rel_embeddings = self.embedding()
        old_data_ent_embeddings, old_data_rel_embeddings = self.old_embeddings()

        # Apply entity distillation mask if available
        if self.entity_distill_mask is not None:
            self.entity_distill_mask = self.entity_distill_mask.to(self.args.device)
            ent_embeddings = ent_embeddings * self.entity_distill_mask.unsqueeze(1)
            old_data_ent_embeddings = old_data_ent_embeddings * self.entity_distill_mask.unsqueeze(1)

        # Calculate Huber loss between current and old embeddings
        ent_distill_loss = self.huber_loss(ent_embeddings, old_data_ent_embeddings)
        rel_distill_loss = self.huber_loss(rel_embeddings, old_data_rel_embeddings)

        # Total distillation loss
        distill_loss = ent_distill_loss + rel_distill_loss

        return distill_loss

    def form_triple_groups(self, triples, weights=None):
        """Form groups of triples based on specified strategy.

        Args:
            triples: Tensor of triples to group
            weights: Optional tensor of weights for each triple

        Returns:
            List of (group_triples, group_weights) tuples
        """
        device = triples.device

        # Ensure weights are on same device
        if weights is not None:
            weights = weights.to(device)

        # Choose grouping strategy
        strategy_map = {
            'relation': self._relation_based_grouping,
            'entity': self._entity_neighborhood_grouping,
            'schema': self._schema_coherent_grouping,
            'batch': self._batch_grouping
        }

        grouping_func = strategy_map.get(self.args.grouping_strategy, self._batch_grouping)
        return grouping_func(triples, weights)

    def _relation_based_grouping(self, triples, weights=None):
        """Group triples that share the same relation type.

        Args:
            triples: Tensor of triples
            weights: Optional tensor of weights

        Returns:
            List of (group_triples, group_weights) tuples
        """
        groups = []
        relations = triples[:, 1]
        device = triples.device

        # Ensure weights are on same device
        if weights is not None:
            weights = weights.to(device)

        # Find unique relations
        unique_relations = torch.unique(relations)

        for rel in unique_relations:
            # Use boolean indexing to find all triples with this relation
            mask = (relations == rel)
            group_triples = triples[mask]

            # Skip small groups
            if len(group_triples) < 2:
                continue

            # Apply weights if provided
            group_weights = weights[mask] if weights is not None else None
            groups.append((group_triples, group_weights))

        return groups

    def _entity_neighborhood_grouping(self, triples, weights=None):
        """Group triples connected to the same entity.

        Args:
            triples: Tensor of triples
            weights: Optional tensor of weights

        Returns:
            List of (group_triples, group_weights) tuples
        """
        groups = []
        device = triples.device

        # Ensure weights are on same device
        if weights is not None:
            weights = weights.to(device)

        # Extract head and tail entities
        heads = triples[:, 0]
        tails = triples[:, 2]

        # Find unique entities
        all_entities = torch.cat([heads, tails])
        unique_entities = torch.unique(all_entities)

        for entity in unique_entities:
            # Find all triples where this entity appears as head or tail
            head_mask = (heads == entity)
            tail_mask = (tails == entity)
            mask = head_mask | tail_mask

            group_triples = triples[mask]

            # Skip small groups
            if len(group_triples) < 2:
                continue

            # Apply weights if provided
            group_weights = weights[mask] if weights is not None else None
            groups.append((group_triples, group_weights))

        return groups

    def _schema_coherent_grouping(self, triples, weights=None):
        """Group triples based on schema weight similarity.

        Args:
            triples: Tensor of triples
            weights: Optional tensor of weights

        Returns:
            List of (group_triples, group_weights) tuples
        """
        groups = []
        device = triples.device

        # Fall back to relation grouping if no weights
        if weights is None:
            return self._relation_based_grouping(triples, weights)

        # Ensure weights are on same device
        weights = weights.to(device)

        # Define weight ranges for grouping
        w_min, w_max = weights.min().item(), weights.max().item()
        num_buckets = 4
        bucket_size = (w_max - w_min) / max(1, num_buckets)

        for i in range(num_buckets):
            bucket_min = w_min + i * bucket_size
            bucket_max = bucket_min + bucket_size

            # Find all triples with weights in this range
            mask = (weights >= bucket_min) & (weights < bucket_max)
            group_triples = triples[mask]

            # Skip small groups
            if len(group_triples) < 2:
                continue

            group_weights = weights[mask]
            groups.append((group_triples, group_weights))

        return groups

    def _batch_grouping(self, triples, weights=None):
        """Simple batching into groups of fixed size.

        Args:
            triples: Tensor of triples
            weights: Optional tensor of weights

        Returns:
            List of (group_triples, group_weights) tuples
        """
        groups = []
        device = triples.device

        # Calculate group size with safety checks
        group_size = min(getattr(self.args, 'group_size_grpo', 128), triples.size(0))
        num_groups = max(1, triples.size(0) // max(1, group_size))

        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min(start_idx + group_size, triples.size(0))

            group_triples = triples[start_idx:end_idx]

            if weights is not None:
                group_weights = weights[start_idx:end_idx]
            else:
                group_weights = None

            groups.append((group_triples, group_weights))

        return groups

    def compute_gradient_projection(self, forget_grad, retain_grad, schema_weight, beta=0.5):
        """Implement the gradient projection to protect retained knowledge.

        Args:
            forget_grad: Gradient to forget a specific triple
            retain_grad: Gradient to retain related triples
            schema_weight: Schema importance weight
            beta: Projection strength parameter

        Returns:
            Projected gradient
        """
        # Calculate projection of forget_grad onto retain_grad
        dot_product = torch.sum(forget_grad * retain_grad)
        retain_norm_squared = torch.sum(retain_grad * retain_grad)

        # Avoid division by zero
        if retain_norm_squared < 1e-8:
            return forget_grad

        projection_magnitude = dot_product / retain_norm_squared
        projected_component = projection_magnitude * retain_grad

        # Apply schema-weighted projection
        projected_grad = forget_grad - beta * schema_weight * projected_component

        return projected_grad

    def compute_policy_ratio(self, current_scores, old_scores):
        """Compute ratio between current and old policies.

        Args:
            current_scores: Current scores tensor
            old_scores: Old scores tensor

        Returns:
            Policy ratio tensor
        """
        # In TransE, lower scores are better, so we negate before sigmoid
        current_prob = torch.sigmoid(-current_scores)
        old_prob = torch.sigmoid(-old_scores)

        # Calculate policy ratio (with epsilon to prevent division by zero)
        ratio = current_prob / (old_prob + 1e-8)

        return ratio

    def compute_advantage(self, rewards):
        """Compute advantage using the GRPO formula.

        Args:
            rewards: Rewards tensor

        Returns:
            Advantages tensor
        """
        # Normalize rewards within group
        mean_reward = torch.mean(rewards)
        std_reward = torch.std(rewards) + 1e-8
        advantages = (rewards - mean_reward) / std_reward

        return advantages

    def compute_kl_divergence(self, current_prob, old_prob):
        """Compute KL divergence between current and old policies.

        Args:
            current_prob: Current probability tensor
            old_prob: Old probability tensor

        Returns:
            KL divergence value
        """
        # KL(current || old)
        kl_div = (current_prob * torch.log((current_prob + 1e-8) / (old_prob + 1e-8))).mean()
        return kl_div

    @torch.no_grad()
    def schema_grpo_loss(self, pos_triples, pos_weights, neg_triples):
        """Optimized schema-aware GRPO loss calculation.

        Args:
            pos_triples: Positive triples tensor
            pos_weights: Schema weights tensor
            neg_triples: Negative triples tensor

        Returns:
            Tensor: GRPO loss value
        """
        # Check for valid inputs
        if pos_triples is None or neg_triples is None or pos_triples.size(0) == 0:
            print("ERROR! no pos or neg triples GRPO loss")
            exit(1)
            return torch.tensor(0.0, device=self.args.device)



        # Get embeddings
        ent_embeddings, rel_embeddings = self.kge_model.embedding()
        old_ent_embeddings, old_rel_embeddings = self.old_embeddings()

        # Add noise if embeddings are too similar
        if torch.norm(ent_embeddings[:10] - old_ent_embeddings[:10]) < 1e-3:
            old_ent_embeddings = old_ent_embeddings + 0.01 * torch.rand_like(old_ent_embeddings)
            old_rel_embeddings = old_rel_embeddings + 0.01 * torch.rand_like(old_rel_embeddings)

        # Group triples according to selected strategy
        matched_groups = self._create_matched_groups(
            pos_triples, neg_triples, pos_weights
        )

        # Calculate losses for each group
        losses = []
        for group_pos, group_neg, group_weights in matched_groups:
            # Skip empty groups
            if len(group_pos) == 0 or len(group_neg) == 0:
                continue

            # Get embeddings for current group
            pos_h, pos_r, pos_t = self._get_embeddings_batch(
                ent_embeddings, rel_embeddings, group_pos
            )
            old_pos_h, old_pos_r, old_pos_t = self._get_embeddings_batch(
                old_ent_embeddings, old_rel_embeddings, group_pos
            )
            neg_h, neg_r, neg_t = self._get_embeddings_batch(
                ent_embeddings, rel_embeddings, group_neg
            )
            old_neg_h, old_neg_r, old_neg_t = self._get_embeddings_batch(
                old_ent_embeddings, old_rel_embeddings, group_neg
            )

            # Compute scores
            pos_scores = self.kge_model.score_fun(pos_h, pos_r, pos_t)
            old_pos_scores = self.kge_model.score_fun(old_pos_h, old_pos_r, old_pos_t)
            neg_scores = self.kge_model.score_fun(neg_h, neg_r, neg_t)
            old_neg_scores = self.kge_model.score_fun(old_neg_h, old_neg_r, old_neg_t)

            # Add noise if scores are too similar
            if torch.norm(pos_scores - old_pos_scores) < 1e-4:
                pos_scores = pos_scores + 0.01 * torch.rand_like(pos_scores)
            if torch.norm(neg_scores - old_neg_scores) < 1e-4:
                neg_scores = neg_scores + 0.01 * torch.rand_like(neg_scores)

            # Compute policy probabilities
            pos_prob = torch.sigmoid(-pos_scores)
            old_pos_prob = torch.sigmoid(-old_pos_scores)
            neg_prob = torch.sigmoid(-neg_scores)
            old_neg_prob = torch.sigmoid(-old_neg_scores)

            # Compute rewards with schema weighting
            rewards = (pos_prob - neg_prob)
            if group_weights is not None:
                rewards = rewards * group_weights

            # Ensure rewards have variation
            if rewards.std() < 1e-4:
                rewards = rewards + 0.01 * torch.rand_like(rewards)

            # Compute policy ratios
            pos_ratios = self.compute_policy_ratio(pos_scores, old_pos_scores)
            neg_ratios = self.compute_policy_ratio(neg_scores, old_neg_scores)

            # Normalize advantages
            advantages = self.compute_advantage(rewards)

            # Compute clipped surrogate objectives
            epsilon = self.args.epsilon_grpo
            clipped_pos_ratios = torch.clamp(pos_ratios, 1.0 - epsilon, 1.0 + epsilon)
            clipped_neg_ratios = torch.clamp(neg_ratios, 1.0 - epsilon, 1.0 + epsilon)

            pos_surrogate = torch.min(pos_ratios * advantages, clipped_pos_ratios * advantages)
            neg_surrogate = torch.min(neg_ratios * -advantages, clipped_neg_ratios * -advantages)

            # Compute KL divergence
            kl_pos = self.compute_kl_divergence(pos_prob, old_pos_prob)
            kl_neg = self.compute_kl_divergence(neg_prob, old_neg_prob)

            # Compute loss
            group_loss = -pos_surrogate.mean() - neg_surrogate.mean() + self.args.beta_grpo * (kl_pos + kl_neg)
            losses.append(group_loss)

        # Compute average loss
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=pos_triples.device)

    def set_boundary_preservation_entities(self, pos_triples):
        # Extract all unique relation IDs from pos_triples
        relations = pos_triples[:, 1].unique()

        # For each relation, identify the entities connected by this relation
        boundary_entities = set()

        for relation in relations:
            # Find all triples with this relation
            rel_mask = (pos_triples[:, 1] == relation)
            rel_triples = pos_triples[rel_mask]

            # Extract head and tail entities
            heads = rel_triples[:, 0].tolist()
            tails = rel_triples[:, 2].tolist()

            # Add these entities to boundary entities
            boundary_entities.update(heads + tails)

        # Convert to tensor
        return torch.tensor(list(boundary_entities),
                            dtype=torch.long, device=self.args.device)

    def boundary_preservation_loss(self, preservation_entities):
        """Calculate boundary preservation loss to enhance recall of knowledge in the forgetting boundary.

        Args:
            preservation_entities: Optional tensor of entity IDs in the distillation boundary.
                             If None, uses self.boundary_distill_entities.

        Returns:
            Tensor: Boundary distillation loss value
        """
        # Use stored boundary entities if none provided

        # Get current and reference embeddings
        current_embeddings = self.kge_model.ent_embeddings.weight

        # Check if reference model embeddings are available
        if not hasattr(self, 'reference_model') or self.reference_model is None:
            # Fall back to using old embeddings as reference
            reference_embeddings, _ = self.old_embeddings()
        else:
            # Use dedicated reference model
            reference_embeddings = self.kge_model.ent_embeddings.weight

        # Extract embeddings for distillation entities
        preservation_entities = preservation_entities.to(self.args.device)
        e = torch.index_select(current_embeddings, 0, preservation_entities)
        e_ref = torch.index_select(reference_embeddings, 0, preservation_entities)

        # Calculate element-wise differences
        diffs = e - e_ref
        diff_norms = torch.norm(diffs, dim=1)

        # Apply Huber-like loss as per Equation 15
        mask_small = (diff_norms <= 1.0)
        mask_large = (diff_norms > 1.0)

        # Compute loss for each case
        loss_small = 0.5 * torch.sum(diffs[mask_small] ** 2, dim=1)
        loss_large = diff_norms[mask_large] - 0.5

        # Combine losses
        total_loss = torch.zeros_like(diff_norms)
        total_loss[mask_small] = loss_small
        total_loss[mask_large] = loss_large

        # Average loss over all boundary entities
        return torch.mean(total_loss)

    def _create_matched_groups(self, pos_triples, neg_triples, pos_weights):
        """Create groups of positive and negative triples.

        Args:
            pos_triples: Positive triples tensor
            neg_triples: Negative triples tensor
            pos_weights: Schema weights tensor

        Returns:
            List of (group_pos, group_neg, group_weights) tuples
        """
        if self.args.grouping_strategy != 'batch':
            # Group by selected strategy
            triple_groups = self.form_triple_groups(pos_triples, pos_weights)

            matched_groups = []
            for group_pos, group_weights in triple_groups:
                # Find matching negative triples
                group_indices = torch.zeros(pos_triples.size(0), dtype=torch.bool, device=pos_triples.device)

                for pos_triple in group_pos:
                    # Find index of this triple in the original pos_triples
                    matches = ((pos_triples == pos_triple).all(dim=1))
                    group_indices = group_indices | matches

                group_neg = neg_triples[group_indices]
                matched_groups.append((group_pos, group_neg, group_weights))
        else:
            # Process in larger groups (simple batching)
            group_size = min(self.args.group_size_grpo, pos_triples.size(0))
            num_groups = max(1, pos_triples.size(0) // group_size)

            matched_groups = []
            for g in range(num_groups):
                start_idx = g * group_size
                end_idx = min(start_idx + group_size, pos_triples.size(0))

                group_pos = pos_triples[start_idx:end_idx]
                group_neg = neg_triples[start_idx:end_idx]
                group_weights = pos_weights[start_idx:end_idx] if pos_weights is not None else None

                matched_groups.append((group_pos, group_neg, group_weights))

        return matched_groups

    def _get_embeddings_batch(self, ent_emb, rel_emb, triples):
        """Efficient batched embedding lookup.

        Args:
            ent_emb: Entity embeddings
            rel_emb: Relation embeddings
            triples: Triples tensor

        Returns:
            Tuple of (h, r, t) embeddings
        """
        h = torch.index_select(ent_emb, 0, triples[:, 0])
        r = torch.index_select(rel_emb, 0, triples[:, 1])
        t = torch.index_select(ent_emb, 0, triples[:, 2])
        return h, r, t

    def combined_loss(self, head, relation, tail, label, pos_triples, neg_triples, pos_weights=None):
        """Combined loss function with schema guidance.

        Args:
            head: Head entity embeddings
            relation: Relation embeddings
            tail: Tail entity embeddings
            label: Triple labels
            pos_triples: Positive triples tensor
            neg_triples: Negative triples tensor
            pos_weights: Schema weights tensor

        Returns:
            Tensor: Combined loss value
        """
        # Base model loss
        base_loss = self.kge_model.loss(head, relation, tail, label)

        # Schema-guided GRPO loss
        grpo_loss = self.schema_grpo_loss(pos_triples, pos_weights, neg_triples)

        # Apply lambda weighting
        weighted_grpo_loss = self.args.grpo_lambda * grpo_loss

        preservation_loss = -1  # it means not used

        # Add distillation loss
        if self.args.use_distill_loss:
            preservation_loss = self.boundary_preservation_loss(
                self.set_boundary_preservation_entities(pos_triples))  # self.preservation_loss()
            weighted_distill_loss = self.args.distill_lambda * preservation_loss
            total_loss = base_loss + weighted_grpo_loss + weighted_distill_loss
        else:
            # Combined loss
            total_loss = base_loss + weighted_grpo_loss  # + weighted_distill_loss

        # Periodic logging
        # self._log_loss_values(base_loss, grpo_loss, preservation_loss, total_loss)

        return total_loss

    def _log_loss_values(self, base_loss, grpo_loss, distill_loss, total_loss):
        """Log loss values periodically.

        Args:
            base_loss: Base loss value
            grpo_loss: GRPO loss value
            distill_loss: Distillation loss value
            total_loss: Total combined loss value
        """
        self._loss_counter += 1
        if self._loss_counter % 10 == 0:
            print(
                f" Loss: Base={base_loss.item():.4f}, GRPO={grpo_loss.item():.4f}, " +
                f" Distill={distill_loss.item():.4f}, Total={total_loss.item():.4f}")

    def gradient_guided_optimization_step(self, forget_triples, retain_triples, schema_weights=None):
        """Implement explicit gradient projection for selective forgetting.

        Args:
            forget_triples: Triples to forget
            retain_triples: Triples to retain
            schema_weights: Schema weights tensor

        Returns:
            dict: Result with forget_loss
        """
        # Make sure we're in training mode
        self.train()

        # Enable gradient computation for all parameters
        for param in self.parameters():
            param.requires_grad = True

        # Convert triples to tensors if they aren't already
        if not torch.is_tensor(forget_triples):
            forget_triples = torch.tensor(forget_triples, device=self.args.device)
        if not torch.is_tensor(retain_triples):
            retain_triples = torch.tensor(retain_triples, device=self.args.device)

        # 1. Compute forget gradients
        try:
            # Ensure we're working with tensors that require gradients
            forget_h, forget_r, forget_t = self._get_embeddings_batch(
                self.kge_model.ent_embeddings.weight, self.kge_model.rel_embeddings.weight, forget_triples
            )

            # Make sure tensors require gradients
            if not forget_h.requires_grad:
                forget_h = forget_h.detach().requires_grad_(True)
            if not forget_r.requires_grad:
                forget_r = forget_r.detach().requires_grad_(True)
            if not forget_t.requires_grad:
                forget_t = forget_t.detach().requires_grad_(True)

            forget_scores = self.kge_model.score_fun(forget_h, forget_r, forget_t)
            forget_loss = torch.mean(forget_scores)

            self.zero_grad()
            forget_loss.backward(retain_graph=True)

            # Store forget gradients
            forget_grads = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                            for name, param in self.named_parameters()}

            # 2. Compute retain gradients
            self.zero_grad()
            retain_h, retain_r, retain_t = self._get_embeddings_batch(
                self.kge_model.ent_embeddings.weight, self.kge_model.rel_embeddings.weight, retain_triples
            )

            # Make sure tensors require gradients
            if not retain_h.requires_grad:
                retain_h = retain_h.detach().requires_grad_(True)
            if not retain_r.requires_grad:
                retain_r = retain_r.detach().requires_grad_(True)
            if not retain_t.requires_grad:
                retain_t = retain_t.detach().requires_grad_(True)

            retain_scores = self.kge_model.score_fun(retain_h, retain_r, retain_t)
            retain_loss = torch.mean(retain_scores)

            self.zero_grad()
            retain_loss.backward()

            # Store retain gradients
            retain_grads = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                            for name, param in self.named_parameters()}

            # 3. Apply schema-weighted gradient projection
            beta = self.args.gradient_projection_weight  # Projection strength
            projected_grads = {}

            # Handle schema weights properly
            if schema_weights is not None:
                if isinstance(schema_weights, float):
                    avg_schema_weight = torch.tensor(schema_weights, device=self.args.device)
                elif torch.is_tensor(schema_weights):
                    avg_schema_weight = torch.mean(schema_weights)
                else:
                    # Default if schema_weights is neither tensor nor float
                    avg_schema_weight = torch.tensor(1.0, device=self.args.device)
            else:
                avg_schema_weight = torch.tensor(1.0, device=self.args.device)

            for name in forget_grads:
                if name in retain_grads:
                    # Apply gradient projection
                    projected_grads[name] = self.compute_gradient_projection(
                        forget_grads[name],
                        retain_grads[name],
                        avg_schema_weight,
                        beta
                    )

            # 4. Apply projected gradients
            self.zero_grad()
            for name, param in self.named_parameters():
                if name in projected_grads:
                    if param.grad is None:
                        param.grad = projected_grads[name].to(param.device)
                    else:
                        param.grad.copy_(projected_grads[name])

            # 5. Update parameters
            self.optimizer.step()

            return {'forget_loss': forget_loss.item(), 'projection_applied': True}

        except Exception as e:
            print(f"ERROR in gradient-guided optimization: {e}")
            # Return a default result with error information
            return {'forget_loss': 0.0, 'error': str(e), 'projection_applied': False}
