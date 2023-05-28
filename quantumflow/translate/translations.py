# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Type

from ..circuits import Circuit
from ..ops import Gate
from ..stdgates import S_H, T_H, CNot, H, I, Ph, S, T, YPow, ZPow

__all__ = [
    "TRANSLATIONS",
    "register_translation",
    "select_translations",
    "circuit_translate",
    "translation_source_gate",
    "translation_target_gates",
]

TRANSLATIONS: List[Callable] = []
"""A list of all registered gate translations"""


def register_translation(translation: Callable) -> Callable:
    """A function decorator used to register gate translations.

    All registered translations are listed in TRANSLATIONS

    """
    TRANSLATIONS.append(translation)

    return translation


def translation_source_gate(trans: Callable) -> Type[Gate]:
    return trans.__annotations__["gate"]


def translation_target_gates(trans: Callable) -> Tuple[Type[Gate]]:
    try:
        ret = trans.__annotations__["return"].__args__[0]
    except KeyError:  # pragma: no cover   # FIXME
        raise ValueError("Translation missing return type annotation")

    if hasattr(ret, "__args__"):  # Union
        gates = ret.__args__
    else:
        gates = (ret,)

    return gates


def select_translations(
    target_gates: Iterable[Type[Gate]],
    translations: Optional[Iterable[Callable]] = None,
) -> List[Callable]:
    """Return a list of translations that will translate source gates to target
    gates.

    If no translations are specified, we use all QuantumFlow translations
    listed in qf.TRANSLATIONS

    For example, to convert a circuit to use gates understood by QUIL:
    ::

        trans = qf.select_translators(qf.QUIL_GATES)
        circ = qf.transform(circ, trans)

    """
    # Warning: Black Voodoo magic. We use python's type annotations to figure
    # out the source gate and target gates of a translation.

    if translations is None:
        translations = TRANSLATIONS

    out_trans = []
    target_gates = set(target_gates)

    source_trans = set(translations)

    while source_trans:  # Loop until we run out of translations
        for trans in translations:
            if trans not in source_trans:
                continue

            from_gate = translation_source_gate(trans)
            to_gates = translation_target_gates(trans)
            if from_gate in target_gates:
                # If translation's source gates are already in targets
                # then we don't need this translation. Discard.
                source_trans.remove(trans)
                break

            if target_gates.issuperset(to_gates):
                # If target gate of translation are already in
                # target list, and source gate isn't, move
                # translation to output list and source gate to targets.
                target_gates.add(from_gate)
                out_trans.append(trans)
                source_trans.remove(trans)
                break
        else:
            # If we got here, none of the remaining translations can be
            # used. Break out of while and return results.
            break  # pragma: no cover

    return out_trans


# Deprecated
select_translators = select_translations


def circuit_translate(
    circ: Circuit,
    translators: Optional[Sequence] = None,
    targets: Optional[Iterable[Type[Gate]]] = None,
    recurse: bool = True,
) -> Circuit:
    """Apply a collection of translations to each gate in a circuit.
    If recurse, then apply translations to output of translations
    until translationally invariant.
    """
    if translators is not None and targets is not None:
        raise ValueError("Specify either targets or translators, not both")

    gates = list(reversed(list(circ)))
    translated: List[Gate] = []

    if translators is None:
        if targets is None:
            targets = [CNot, YPow, ZPow, I, Ph, H, T, S, T_H, S_H]  # FIXME
        translators = select_translations(targets)

    # Use type annotations to do dynamic dispatch
    gateclass_translation = {
        translation_source_gate(trans): trans for trans in translators
    }

    while gates:
        gate = gates.pop()
        if type(gate) in gateclass_translation:
            trans = gateclass_translation[type(gate)](gate)
            if recurse:
                gates.extend(reversed(list(trans)))
            else:
                translated.extend(trans)
        else:
            translated += gate

    return Circuit(translated)


# fin
