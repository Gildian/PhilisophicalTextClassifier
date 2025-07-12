#!/usr/bin/env python3
"""
Generate additional philosophical text samples for training data.
Creates balanced Continental and Analytic philosophy examples.
"""

import csv
import random

# Continental philosophy themes and concepts
continental_concepts = [
    "Being", "Dasein", "temporality", "authenticity", "thrownness", "facticity",
    "anxiety", "conscience", "resoluteness", "being-toward-death", "hermeneutics",
    "phenomenology", "intentionality", "lived experience", "embodiment", "intersubjectivity",
    "dialectic", "negation", "spirit", "absolute", "freedom", "recognition",
    "différance", "trace", "deconstruction", "logocentrism", "presence", "absence",
    "power", "discourse", "subject", "genealogy", "archaeology", "episteme",
    "unconscious", "desire", "signifier", "Other", "Real", "Symbolic", "Imaginary",
    "flesh", "chiasm", "reversibility", "depth", "invisible", "visible",
    "lifeworld", "horizon", "epoché", "reduction", "constitution", "synthesis"
]

continental_philosophers = [
    "Heidegger", "Sartre", "Merleau-Ponty", "Husserl", "Gadamer", "Ricoeur",
    "Hegel", "Marx", "Kierkegaard", "Nietzsche", "Foucault", "Derrida",
    "Levinas", "Lacan", "Deleuze", "Guattari", "Bataille", "Blanchot",
    "Beauvoir", "Irigaray", "Kristeva", "Butler", "Adorno", "Benjamin"
]

continental_verbs = [
    "reveals", "discloses", "conceals", "manifests", "emerges", "unfolds",
    "constitutes", "grounds", "undermines", "transforms", "negates", "preserves",
    "discovers", "encounters", "experiences", "dwells", "inhabits", "gathers"
]

# Analytic philosophy themes and concepts
analytic_concepts = [
    "proposition", "truth conditions", "logical form", "validity", "soundness",
    "necessity", "possibility", "counterfactuals", "rigid designation", "natural kinds",
    "functionalism", "multiple realizability", "computational theory", "intentionality",
    "qualia", "consciousness", "zombie argument", "knowledge argument", "explanatory gap",
    "externalism", "internalism", "reliabilism", "justification", "skepticism",
    "reference", "sense", "meaning", "use", "speech acts", "implicature",
    "causation", "laws of nature", "properties", "universals", "particulars",
    "identity", "persistence", "composition", "modal logic", "possible worlds"
]

analytic_philosophers = [
    "Russell", "Moore", "Wittgenstein", "Carnap", "Quine", "Davidson", "Putnam",
    "Kripke", "Lewis", "Chalmers", "Jackson", "Searle", "Dennett", "Fodor",
    "Dretske", "Millikan", "Burge", "McDowell", "Brandom", "Sellars",
    "Gettier", "Goldman", "Nozick", "Plantinga", "van Fraassen", "Fine"
]

analytic_verbs = [
    "entails", "implies", "follows", "demonstrates", "proves", "refutes",
    "establishes", "shows", "argues", "claims", "maintains", "holds",
    "defines", "analyzes", "reduces", "eliminates", "identifies", "distinguishes"
]

def generate_continental_text():
    """Generate a Continental philosophy text sample."""
    templates = [
        f"The phenomenon of {random.choice(continental_concepts)} {random.choice(continental_verbs)} the {random.choice(continental_concepts)} that {random.choice(continental_verbs)} in {random.choice(continental_concepts)}.",
        f"{random.choice(continental_philosophers)}'s analysis of {random.choice(continental_concepts)} {random.choice(continental_verbs)} how {random.choice(continental_concepts)} {random.choice(continental_verbs)} the structure of {random.choice(continental_concepts)}.",
        f"The {random.choice(continental_concepts)} of {random.choice(continental_concepts)} cannot be understood apart from the {random.choice(continental_concepts)} that {random.choice(continental_verbs)} {random.choice(continental_concepts)}.",
        f"Authentic {random.choice(continental_concepts)} {random.choice(continental_verbs)} through the {random.choice(continental_concepts)} that {random.choice(continental_verbs)} {random.choice(continental_concepts)} from its fallen state.",
        f"The temporal structure of {random.choice(continental_concepts)} {random.choice(continental_verbs)} the {random.choice(continental_concepts)} between {random.choice(continental_concepts)} and {random.choice(continental_concepts)}.",
    ]
    return random.choice(templates)

def generate_analytic_text():
    """Generate an Analytic philosophy text sample."""
    templates = [
        f"The {random.choice(analytic_concepts)} argument {random.choice(analytic_verbs)} that {random.choice(analytic_concepts)} {random.choice(analytic_verbs)} {random.choice(analytic_concepts)} in all possible worlds.",
        f"{random.choice(analytic_philosophers)}'s theory of {random.choice(analytic_concepts)} {random.choice(analytic_verbs)} how {random.choice(analytic_concepts)} can be reduced to {random.choice(analytic_concepts)}.",
        f"If {random.choice(analytic_concepts)} {random.choice(analytic_verbs)} {random.choice(analytic_concepts)}, then {random.choice(analytic_concepts)} must {random.choice(analytic_verbs)} the {random.choice(analytic_concepts)} conditions.",
        f"The problem of {random.choice(analytic_concepts)} {random.choice(analytic_verbs)} difficulties for theories that {random.choice(analytic_verbs)} {random.choice(analytic_concepts)} with {random.choice(analytic_concepts)}.",
        f"Modal {random.choice(analytic_concepts)} provides formal tools for analyzing {random.choice(analytic_concepts)} and its relation to {random.choice(analytic_concepts)}.",
    ]
    return random.choice(templates)

def generate_additional_samples(num_samples=600):
    """Generate additional philosophical text samples."""
    samples = []
    
    for i in range(num_samples // 2):
        # Continental sample
        continental_text = generate_continental_text()
        samples.append([continental_text, "Continental"])
        
        # Analytic sample
        analytic_text = generate_analytic_text()
        samples.append([analytic_text, "Analytic"])
    
    return samples

if __name__ == "__main__":
    # Read existing samples
    existing_samples = []
    try:
        with open('sample_data.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            existing_samples = list(reader)
        print(f"Found {len(existing_samples)} existing samples")
    except FileNotFoundError:
        existing_samples = [['text', 'label']]  # Header only
    
    # Generate new samples
    new_samples = generate_additional_samples(600)
    print(f"Generated {len(new_samples)} new samples")
    
    # Combine and write
    all_samples = existing_samples + new_samples
    
    with open('sample_data.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_samples)
    
    print(f"Total samples in file: {len(all_samples)}")
    print(f"Continental samples: {sum(1 for row in all_samples[1:] if len(row) > 1 and row[1] == 'Continental')}")
    print(f"Analytic samples: {sum(1 for row in all_samples[1:] if len(row) > 1 and row[1] == 'Analytic')}")
