"""RDKit-based molecule rendering to SVG with optional atom highlighting."""

from __future__ import annotations

from io import BytesIO
import base64

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

_PALETTE = {
    "highlight": (0.769, 0.404, 0.310),   # terracotta
    "background": (0.980, 0.969, 0.949),  # ivory
}


def render_svg(smiles: str, highlight_atoms: list[int] | None = None, size: int = 420) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size, int(size * 0.75))
    opts = drawer.drawOptions()
    opts.setBackgroundColour(_PALETTE["background"])
    opts.bondLineWidth = 2.2
    opts.padding = 0.12
    hl = list(highlight_atoms) if highlight_atoms else []
    if hl:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=hl,
            highlightAtomColors={i: _PALETTE["highlight"] for i in hl},
            highlightBonds=[],
            highlightBondColors={},
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def svg_to_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def png_from_smiles(smiles: str, size: int = 300) -> bytes | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    bio = BytesIO()
    bio.write(drawer.GetDrawingText().encode() if isinstance(drawer.GetDrawingText(), str) else drawer.GetDrawingText())
    return bio.getvalue()
