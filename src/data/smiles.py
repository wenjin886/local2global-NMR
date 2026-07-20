from typing import List


def tokenize_smiles_tokens(smiles: str) -> List[str]:
    """Tokenize SMILES with RXN and verify lossless reconstruction."""
    try:
        from rxn.chemutils.tokenization import tokenize_smiles
    except ImportError as error:
        raise ImportError(
            "SMILES tokenization requires the 'rxn-chem-utils' package"
        ) from error
    tokens = tokenize_smiles(smiles).split()
    if "".join(tokens) != smiles:
        raise ValueError(f"SMILES tokenization was not lossless: {smiles}")
    return tokens
