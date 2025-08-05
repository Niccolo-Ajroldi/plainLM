
import torch

from itertools import chain
from typing import Dict, List, Any


def intra_doc_causal_mask(doc_boundaries: list, max_seq_length: int, device='cpu') -> torch.Tensor:
    """Create a block diagonal causal mask for the intra-document segments."""

    if sum(doc_boundaries) != max_seq_length:
        raise ValueError("Sum of doc_boundaries does not match max_seq_length.")

    # Create a causal mask for each segment in doc_boundaries (lower triangle is True, upper is False)
    sub_masks_bool = []
    for segment_length in doc_boundaries:
        segment_causal_mask_bool = torch.tril(torch.ones(
            (segment_length, segment_length), dtype=torch.bool, device=device
        ))
        sub_masks_bool.append(segment_causal_mask_bool)

    # Use torch.block_diag to combine single masks into a block diagonal mask.
    # This creates a matrix where sub_masks are on the diagonal, and off-diagonal blocks are False.
    block_diagonal_mask_bool = torch.block_diag(*sub_masks_bool)

    return block_diagonal_mask_bool

def _get_docs_boundaries(doc_lengths: List[int], n_chunks: int, max_seq_length: int) -> List[List[int]]:
    """
    Get the boundaries of documents in the concatenated chunks.
    
    A list of documents has been concatenated and chunked into `n_chunks` of `max_seq_length`.
    Each original document had a different length, defined in `doc_lengths`.

    We need to determine how many tokens from each document are present in each chunk.
    This function returns a list of lists, where each inner list contains the lengths of document
    segments that are present in the corresponding chunk.
    
    For example, if `doc_lengths = [10, 20, 40]`, `n_chunks = 2`, and `max_seq_length = 30`, 
    the output will be:
    [
        [10, 20],  # First chunk contains 10 tokens from the first doc and 20 from the second doc.
        [30]       # Second chunk contains 30 tokens from the third doc.
    ]
    """

    # doc_boundaries[i] will contain the length of document segments within the i-th chunk
    doc_boundaries = [[] for _ in range(n_chunks)]

    doc_idx = 0 # Index of the current document being processed
    current_doc_remainder = 0 # Remaining length of the current document being processed

    # Iterate through each generated chunk
    for chunk_idx in range(n_chunks):
        current_chunk_filled_length = 0

        # Fill the current chunk
        while current_chunk_filled_length < max_seq_length:
            if current_doc_remainder == 0:
                # If the previous document part is fully consumed, get the next document
                if doc_idx < len(doc_lengths):
                    current_doc_remainder = doc_lengths[doc_idx]
                    doc_idx += 1
                else:
                    # No more documents to add. Fill the rest of the chunk with padding if necessary.
                    # Given 'total_length' truncation, this loop should ideally only break when
                    # current_chunk_filled_length == max_seq_length, or if it's the very last
                    # (possibly incomplete) chunk if total_length wasn't a perfect multiple.
                    # Since we truncated total_length to be a multiple, this `break` handles
                    # any edge cases where the last chunk might not be fully filled by documents.
                    break

            # Calculate how much of the current document can fit into the remaining space of the chunk
            space_in_chunk = max_seq_length - current_chunk_filled_length
            amount_to_add = min(current_doc_remainder, space_in_chunk)

            doc_boundaries[chunk_idx].append(amount_to_add)
            current_chunk_filled_length += amount_to_add
            current_doc_remainder -= amount_to_add

    return doc_boundaries


# Main data processing function that will concatenate all texts 
# from tokenized dataset and generate chunks of max_seq_length.
# NOTE: expected token loss by batched concat_chunk, 
# it truncates leftover tokens that don't fill a full max_seq_length chunk.
def concat_chunck(examples: Dict[str, List[Any]], max_seq_length) -> Dict[str, List[Any]]:
    """Concatenate all texts and split them into chunks of max_seq_length."""

    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length

    # Split by chunks of max_len.
    result = {
      k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] 
      for k, t in concatenated_examples.items()
    }

    # Lengths of original documents in the input examples.
    original_docs_lengths = [len(example) for example in examples['input_ids']]

    # Number of resulting chunks
    n_chunks = len(result['input_ids'])

    # Retrieve doc boundaries from original document lengths
    result['docs_lengths'] = _get_docs_boundaries(original_docs_lengths, n_chunks, max_seq_length)

    return result
