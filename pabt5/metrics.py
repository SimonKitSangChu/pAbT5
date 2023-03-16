from typing import List, Any, Optional, Dict

import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
import torch
from tqdm import tqdm

from .util import spaceout
from .dataset import decode2string


def accuracy_from_ids(label_ids: np.array, output_ids: np.array) -> float:
    assert label_ids.ndim == 1 and output_ids.ndim == 1
    label_ids = label_ids.tolist()
    output_ids = output_ids.tolist()
    accuracy = sum(a == b for a, b in zip(label_ids, output_ids)) / len(label_ids)
    return accuracy


def accuracy_from_string(predictions: List[str], labels: List[str], return_mean: bool = True):
    accuracy = []
    for prediction, label in zip(predictions, labels):
        match = sum(p == l for l, p in zip(list(label), list(prediction)))  # sum up to length of label
        accuracy.append(match / len(label))

    accuracy = np.array(accuracy)
    if return_mean:
        return float(accuracy.mean())
    else:
        return accuracy


def accuracy_from_logits(logits: np.array, label_ids: np.array) -> float:
    prediction_ids = np.argmax(logits, axis=-1)
    return accuracy_from_ids(label_ids, prediction_ids)


def entropy_from_logits(logits: np.array, base: float = 2, return_mean: bool = True):
    probs = softmax(logits, axis=-1)
    s = entropy(probs, axis=-1, base=base)
    if return_mean:
        return float(s.mean())
    else:
        return s


def perplexity_from_logits(logits: np.array, base: float = 2, return_mean: bool = True, return_entropy: bool = False):
    s = entropy_from_logits(logits=logits, base=base, return_mean=False)
    perplexity = base ** s

    if return_mean and return_entropy:
        return {'perplexity': float(perplexity.mean()), 'entropy': float(s.mean())}
    elif return_mean and not return_entropy:
        return float(perplexity.mean())
    elif not return_mean and return_entropy:
        return {'perplexity': perplexity, 'entropy': s}
    else:
        return perplexity


def evaluate_model(
        model,
        tokenizer,
        input_ids: torch.Tensor,
        label_ids: torch.Tensor,
        output_attentions: bool = False,
        output_last_hidden_state: bool = False,
        return_dict: bool = True,
        skip_special_tokens: bool = True,
        **forward_kwargs,
) -> Dict[str, Any]:
    assert input_ids.ndim == 2
    assert input_ids.shape[0] == 1, 'only support single sequence evaluation due to loss and perplexity '
    'calculations, but got input_ids.shape[0] != 1'
    assert return_dict, 'only support return_dict = True'

    device = model.device
    input_ids = input_ids.to(device)
    label_ids = label_ids.to(device)
    for k, v in forward_kwargs.items():
        if issubclass(v.__class__, torch.Tensor):
            forward_kwargs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=label_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_last_hidden_state,
            return_dict=True,
            **forward_kwargs,
        )

    loss = outputs.loss.cpu().item()
    perplexity = np.exp(loss)

    logits = outputs.logits.cpu().numpy()
    output_ids = np.argmax(logits, axis=-1)
    output_sequence = decode2string(tokenizer, output_ids, skip_special_tokens=skip_special_tokens, squeeze=True)
    label_sequence = decode2string(tokenizer, label_ids, skip_special_tokens=skip_special_tokens, squeeze=True)
    encoder_sequence = decode2string(tokenizer, input_ids, skip_special_tokens=skip_special_tokens, squeeze=True)

    return_dict = {
        'encoder_sequence': encoder_sequence,
        'label_sequence': label_sequence,
        'output_sequence': output_sequence,
        'loss': loss,
        'perplexity': perplexity,
        'logits': logits,
        'accuracy': accuracy_from_logits(logits[0], label_ids[0]),
    }

    if output_attentions:
        if hasattr(outputs, 'encoder_attentions'):
            return_dict['encoder_attentions'] = torch.cat(outputs.encoder_attentions, dim=0).cpu().numpy()
        if hasattr(outputs, 'decoder_attentions'):
            return_dict['decoder_attentions'] = torch.cat(outputs.decoder_attentions, dim=0).cpu().numpy()
        if hasattr(outputs, 'cross_attentions'):
            return_dict['cross_attentions'] = torch.cat(outputs.cross_attentions, dim=0).cpu().numpy()

    if output_last_hidden_state:
        if hasattr(outputs, 'decoder_hidden_states'):
            return_dict['decoder_last_hidden_state'] = outputs.decoder_hidden_states[-1].cpu().numpy()
        if hasattr(outputs, 'encoder_last_hidden_state'):
            return_dict['encoder_last_hidden_state'] = outputs.encoder_last_hidden_state.cpu().numpy()
        if hasattr(outputs, 'last_hidden_state'):
            return_dict['last_hidden_state'] = outputs.last_hidden_state.cpu().numpy()

    return return_dict


def evaluate_generate_outputs(
        model,
        tokenizer,
        generate_outputs,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        output_last_hidden_state: bool = False,
        drop_duplicates: bool = False,
        skip_special_tokens: bool = True,
        **forward_kwargs,
) -> Dict[str, Any]:
    evaluations = []
    generate_sequences = []

    for generate_idx, generate_ids_ in enumerate(generate_outputs[0]):
        generate_sequence = tokenizer.decode(generate_ids_, skip_special_tokens=skip_special_tokens)  # clean tokens
        generate_sequence = generate_sequence.replace(' ', '')
        generate_ids = tokenizer.encode(spaceout(generate_sequence))
        generate_ids = torch.tensor([generate_ids], device=model.device)

        duplicate = any(generate_sequence == evaluation['generate_sequence'] for \
                        evaluation in evaluations)
        if drop_duplicates and duplicate:
            continue

        evaluation = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            label_ids=generate_ids,
            output_attentions=output_attentions,
            output_last_hidden_state=output_last_hidden_state,
            **forward_kwargs,
        )
        del evaluation['output_sequence']

        evaluation['generate_sequence'] = generate_sequence
        try:
            score = generate_outputs.scores
            score = score.cpu().numpy()
            evaluation['score'] = score
        except AttributeError:
            pass

        evaluations.append(evaluation)

    evaluations = {k: [i[k] for i in evaluations] for k in evaluations[0].keys()}
    return evaluations


def pseudolikelihood_esm(
        model,
        sequence: str,
        batch_size: int = 1,
        mask_token: str = '<mask>',
        device: str = 'cpu',
        alphabet: Any = None,
) -> float:
    if batch_size == -1:
        batch_size = len(sequence)

    if type(model) == str:
        model, alphabet = torch.hub.load('facebookresearch/esm:main', model)
        model = model.eval()

    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    sequences = []
    for i in range(len(sequence)):
        sequence_ = sequence[:i] + mask_token + sequence[i + 1:]
        sequences.append((i, sequence_))

    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_labels = torch.tensor(batch_labels)

    log_probs = []
    for subbatch_labels, subbatch_tokens in zip(batch_labels.split(batch_size), batch_tokens.split(batch_size)):
        subbatch_tokens = subbatch_tokens.to(device)
        with torch.no_grad():
            token_probs = torch.log_softmax(model(subbatch_tokens)['logits'], dim=-1)

        for i_batch, i in enumerate(subbatch_labels):
            i = i.item()
            subbatch_log_prob = token_probs[i_batch, i+1, alphabet.get_idx(sequence[i])]  # +1 for BOS
            subbatch_log_prob = subbatch_log_prob.cpu().item()
            log_probs.append(subbatch_log_prob)

    return -sum(log_probs) / len(log_probs)
