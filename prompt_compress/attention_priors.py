"""
Per-prompt attention-informed priors for the Bayesian optimiser.

For each component in the parsed prompt, compute the mean cosine similarity
("attention") to every other component using sentence-transformers embeddings.
High mean attention (>0.7) means the component is tightly coupled to the rest
of the prompt — keep it. Low mean attention (<0.4) means it is independent —
safe to drop.

The output is shaped to match `InformedBayesianOptimiser._load_prior()` so the
existing prior pipeline can consume it without changes.
"""

import numpy as np
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer

COMPONENT_NAMES = ['instruction', 'examples', 'constraints', 'style', 'context']


class AttentionPriorGenerator:
    """
    Generate a prompt-specific prior from component-to-component attention.
    """

    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

    HIGH_THRESHOLD = 0.7
    LOW_THRESHOLD = 0.4

    # Inclusion probabilities mapped from coupling strength
    HIGH_INCLUSION = 0.85
    MID_INCLUSION = 0.5
    LOW_INCLUSION = 0.25

    _shared_model: Optional[SentenceTransformer] = None

    def __init__(self):
        if AttentionPriorGenerator._shared_model is None:
            AttentionPriorGenerator._shared_model = SentenceTransformer(self.EMBEDDING_MODEL)
        self.model = AttentionPriorGenerator._shared_model

    def generate(self, components: Dict) -> Dict:
        """
        Build a prior dict compatible with InformedBayesianOptimiser.

        Args:
            components: Dict[str, list[str]] from PromptParser.parse(). May
                        include a '__protected__' key, which is ignored.

        Returns:
            Dict with keys 'mean', 'variance', 'source', plus an extra
            'attention_matrix' (5x5) and 'mean_attention' (per-component)
            for debugging/inspection.
        """
        present, embeddings = self._embed_components(components)

        # Default inclusion probabilities (used when a component is empty)
        prior_mean = {
            'has_instruction': self.HIGH_INCLUSION,  # always favour instruction
            'has_examples': self.MID_INCLUSION,
            'has_constraints': self.MID_INCLUSION,
            'has_style': self.LOW_INCLUSION,
            'has_context': self.LOW_INCLUSION,
            'instruction_length': 0.7,
            'num_examples': 0.4,
        }

        attention_matrix = np.zeros((5, 5))
        mean_attention = {name: 0.0 for name in COMPONENT_NAMES}

        if len(present) >= 2:
            sim = self._cosine_matrix(embeddings)
            for i, name_i in enumerate(present):
                col = COMPONENT_NAMES.index(name_i)
                for j, name_j in enumerate(present):
                    row = COMPONENT_NAMES.index(name_j)
                    attention_matrix[col, row] = sim[i, j]
                # Mean attention excludes the diagonal (self-similarity = 1.0)
                others = [sim[i, j] for j in range(len(present)) if j != i]
                mean_attention[name_i] = float(np.mean(others)) if others else 0.0

            for name in present:
                ma = mean_attention[name]
                if ma >= self.HIGH_THRESHOLD:
                    prior_mean[f'has_{name}'] = self.HIGH_INCLUSION
                elif ma < self.LOW_THRESHOLD:
                    prior_mean[f'has_{name}'] = self.LOW_INCLUSION
                else:
                    prior_mean[f'has_{name}'] = self.MID_INCLUSION

        # Variance: shrink when we have strong signal (many present components),
        # widen when we have little to learn from.
        prior_variance = 0.15 if len(present) >= 3 else 0.25

        return {
            'mean': prior_mean,
            'variance': prior_variance,
            'source': 'per-prompt attention',
            'attention_matrix': attention_matrix.tolist(),
            'mean_attention': mean_attention,
        }

    def _embed_components(self, components: Dict):
        """
        Embed each present component (concatenated sentences) and return the
        ordered list of names and their embedding matrix.
        """
        present_names: list[str] = []
        texts: list[str] = []
        for name in COMPONENT_NAMES:
            sentences = components.get(name, [])
            if not sentences:
                continue
            joined = ' '.join(sentences) if isinstance(sentences, list) else str(sentences)
            if not joined.strip():
                continue
            present_names.append(name)
            texts.append(joined)
        if not texts:
            return [], np.zeros((0, 384))
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return present_names, embeddings

    @staticmethod
    def _cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalised = embeddings / norms
        return normalised @ normalised.T


if __name__ == "__main__":
    sample_components = {
        'instruction': ['You are a SQL tutor. Explain query plans.'],
        'examples': ['Example: SELECT * FROM users WHERE id = 1.'],
        'constraints': ['Always include the cost estimate.', 'Never invent table names.'],
        'style': ['Use a formal tone.'],
        'context': [],
    }
    gen = AttentionPriorGenerator()
    prior = gen.generate(sample_components)
    print('mean inclusion probabilities:')
    for k, v in prior['mean'].items():
        print(f'  {k}: {v:.3f}')
    print()
    print('mean attention per component:')
    for k, v in prior['mean_attention'].items():
        print(f'  {k}: {v:.3f}')
