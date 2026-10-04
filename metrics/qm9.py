"""QM9 stability and validity for padded EFDM molecule samples."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from rdkit import Chem


_MODEL_ATOMS = ("C", "H", "N", "O", "F")
_QM9_ATOMS = ("H", "C", "N", "O", "F")
_VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}
_SINGLE = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}
_DOUBLE = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}
_TRIPLE = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}
_BOND_TYPES = (
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
)


def _bond_order(atom1: str, atom2: str, distance: Any) -> int:
    distance = 100 * distance
    if distance < _SINGLE[atom1][atom2] + 10:
        if atom1 in _DOUBLE and atom2 in _DOUBLE[atom1]:
            if distance < _DOUBLE[atom1][atom2] + 5:
                if atom1 in _TRIPLE and atom2 in _TRIPLE[atom1]:
                    if distance < _TRIPLE[atom1][atom2] + 3:
                        return 3
                return 2
        return 1
    return 0


def _stability(positions: torch.Tensor, atom_types: torch.Tensor) -> tuple[bool, int, int]:
    positions_np = positions.numpy()
    types_np = atom_types.numpy()
    bonds = np.zeros(len(types_np), dtype="int")
    for i in range(len(types_np)):
        for j in range(i + 1, len(types_np)):
            displacement = positions_np[i] - positions_np[j]
            distance = np.sqrt(np.sum(displacement**2))
            order = _bond_order(_QM9_ATOMS[types_np[i]], _QM9_ATOMS[types_np[j]], distance)
            bonds[i] += order
            bonds[j] += order
    stable_atoms = sum(
        int(_VALENCE[_QM9_ATOMS[atom]] == degree)
        for atom, degree in zip(types_np, bonds)
    )
    return stable_atoms == len(types_np), stable_atoms, len(types_np)


def _build_molecule(positions: torch.Tensor, atom_types: torch.Tensor) -> Chem.Mol:
    molecule = Chem.RWMol()
    for atom in atom_types:
        molecule.AddAtom(Chem.Atom(_QM9_ATOMS[int(atom)]))
    distances = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0), p=2).squeeze(0)
    for i in range(len(atom_types)):
        for j in range(i):
            first, second = sorted((int(atom_types[i]), int(atom_types[j])))
            order = _bond_order(_QM9_ATOMS[first], _QM9_ATOMS[second], distances[i, j])
            if order:
                molecule.AddBond(i, j, _BOND_TYPES[order])
    return molecule


def _mol_to_smiles(molecule: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(molecule)
    except ValueError:
        return None
    return Chem.MolToSmiles(molecule)


def _is_valid(positions: torch.Tensor, atom_types: torch.Tensor) -> bool:
    molecule = _build_molecule(positions, atom_types)
    if _mol_to_smiles(molecule) is None:
        return False
    fragments = Chem.rdmolops.GetMolFrags(molecule, asMols=True)
    largest = max(fragments, default=molecule, key=lambda mol: mol.GetNumAtoms())
    return _mol_to_smiles(largest) is not None


def evaluate_qm9(sample_payload: dict) -> dict:
    """Score all requested molecules, including empty and nonfinite failures.

    Input is the dictionary saved by ``sample.py``. ``keep`` determines the
    declared atom count; it must not be inferred from rendered coordinates.
    """
    samples = sample_payload["samples"]
    keep = sample_payload["keep"]
    if not isinstance(samples, torch.Tensor) or not isinstance(keep, torch.Tensor):
        raise TypeError("samples and keep must be tensors")
    if samples.ndim != 3 or samples.shape[-1] != 8 or keep.shape != samples.shape[:2]:
        raise ValueError("QM9 samples must have shape [N, K, 8] and keep [N, K]")
    if len(samples) == 0:
        raise ValueError("At least one sample is required")
    samples = samples.detach().cpu()
    keep = keep.detach().to(device="cpu", dtype=torch.bool)

    processed = []
    empty_indices = []
    nonfinite_indices = []
    failed_declared_atoms = 0
    for index, (sample, mask) in enumerate(zip(samples, keep)):
        declared = int(mask.sum())
        if declared == 0:
            empty_indices.append(index)
            continue
        candidate = sample[mask]
        if not bool(torch.isfinite(candidate).all()):
            nonfinite_indices.append(index)
            failed_declared_atoms += declared
            continue
        positions = candidate[:, :3].to(torch.float32)
        model_types = candidate[:, 3:8].argmax(dim=1)
        qm9_types = torch.tensor(
            [_QM9_ATOMS.index(_MODEL_ATOMS[int(atom)]) for atom in model_types],
            dtype=torch.long,
        )
        processed.append((positions, qm9_types))

    stable_molecules = stable_atoms = total_atoms = valid_molecules = 0
    for positions, atom_types in processed:
        molecule_stable, atom_stable, atom_count = _stability(positions, atom_types)
        stable_molecules += int(molecule_stable)
        stable_atoms += atom_stable
        total_atoms += atom_count
        valid_molecules += int(_is_valid(positions, atom_types))
    total_atoms += failed_declared_atoms
    n_requested = len(samples)
    n_processed = len(processed)
    return {
        "atom_stability": stable_atoms / max(total_atoms, 1),
        "molecule_stability": stable_molecules / n_requested,
        "validity": valid_molecules / n_requested,
        "molecule_stability_processed_only": stable_molecules / max(n_processed, 1),
        "validity_processed_only": valid_molecules / max(n_processed, 1),
        "stable_atoms": stable_atoms,
        "total_atoms": total_atoms,
        "stable_molecules": stable_molecules,
        "valid_molecules": valid_molecules,
        "n_requested": n_requested,
        "n_processed_nonempty": n_processed,
        "n_empty": len(empty_indices),
        "empty_indices": empty_indices,
        "n_nonfinite_molecules": len(nonfinite_indices),
        "nonfinite_molecule_indices": nonfinite_indices,
        "nonfinite_declared_atoms": failed_declared_atoms,
    }
