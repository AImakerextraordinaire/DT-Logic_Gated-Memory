"""Batched scoring must equal one-question-at-a-time scoring."""
import torch

from dmodel.data.facts import Question
from dmodel.eval import batch_choice_logprobs, confidences


class _Tok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **_):
        return {"input_ids": [1 + (ord(c) % 50) for c in text]}


class _LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(64, 16)
        self.rnn = torch.nn.GRU(16, 16, batch_first=True)
        self.head = torch.nn.Linear(16, 64)

    def forward(self, input_ids, attention_mask=None):
        h, _ = self.rnn(self.emb(input_ids))
        return type("O", (), {"logits": self.head(h)})()


def test_batched_equals_single_and_confidence_sums_to_one():
    tok, lm = _Tok(), _LM().eval()
    qs = [Question("Q: color?\nA:", [" red", " blue", " teal"], 2, None, "seen"),
          Question("Q: pet of Zed?\nA:", [" cat", " hedgehog"], 0, None, "seen")]
    batched = batch_choice_logprobs(lm, tok, qs, "cpu", max_seqs=3)
    single = [batch_choice_logprobs(lm, tok, [q], "cpu")[0] for q in qs]
    for a, b in zip(batched, single):
        assert all(abs(x - y) < 1e-4 for x, y in zip(a, b))
    c = confidences(lm, tok, qs, "cpu")
    assert all(0.0 <= x <= 1.0 for x in c)
    assert c[0] < 0.5  # teal is not the best-scoring choice under this random LM
