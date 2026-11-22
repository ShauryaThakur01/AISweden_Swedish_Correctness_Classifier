"""Build matched training sets and one group-disjoint validation split."""

import itertools
import json
import random
from pathlib import Path

SEED = 42
DATA = Path("prepared")


def read_rows(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def synthetic_rows(original, reserved):
    # Every lexical choice works with each sentence frame.
    people = ["läraren", "eleven", "studenten", "forskaren", "läkaren",
              "chefen", "grannen", "journalisten", "tränaren", "teknikern"]
    objects = ["brev", "avtal", "schema", "protokoll", "meddelande", "dokument"]
    times = ["I går", "Förra veckan", "I måndags", "Efter lunch", "I morse", "På mötet"]
    actions = ["läsa", "skriva", "granska", "hitta"]
    contexts = list(itertools.product(times, people, objects, actions))
    rng = random.Random(SEED)
    rng.shuffle(contexts)
    labels = [row["label"] for row in original]
    rng.shuffle(labels)
    verbs = {"läsa": "läste", "skriva": "skrev", "granska": "granskade", "hitta": "hittade"}
    seen = set(reserved)
    result = []
    for time, person, noun, action in contexts:
        label = labels[len(result)]
        poor = label == "poor"
        # Sample the same frames for either label. Within each frame,
        # incorrect forms differ from the grammatical sentence at known sites.
        frame = rng.randrange(3)
        error_count = rng.randint(1, 2) if poor else 0
        sites = set(rng.sample(["clause", "agreement"], error_count))
        phrase = f"{'ett ny' if 'agreement' in sites else 'ett nytt'} {noun}"
        if frame == 0:
            verb = verbs[action]
            clause = f"{person} {verb}" if "clause" in sites else f"{verb} {person}"
            text = f"{time} {clause} {phrase}."
            clause_error = "verb-second order"
        elif frame == 1:
            verb = verbs[action] if "clause" in sites else action
            text = f"{person.capitalize()} ville {verb} {phrase} {time.lower()}."
            clause_error = "finite verb after modal"
        else:
            verb = verbs[action] if "clause" in sites else action
            text = f"{time} försökte {person} att {verb} {phrase}."
            clause_error = "finite verb after infinitive marker"
        key = " ".join(text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        errors = [name for site, name in (("clause", clause_error), ("agreement", "adjective agreement")) if site in sites]
        index = len(result)
        result.append(dict(id=index, group=index, text=text, label=label,
                           score=[0.95, 0.45, 0.25][error_count],
                           error=", ".join(errors) or None))
        if len(result) == len(labels):
            return result
    raise ValueError("Template inventory too small for this training set.")


def main():
    rows = read_rows("data/train.jsonl")
    groups = sorted({row["group"] for row in rows})
    random.Random(SEED).shuffle(groups)
    held_out = set(groups[:round(0.2 * len(groups))])
    train = [r for r in rows if r["group"] not in held_out]
    valid = [r for r in rows if r["group"] in held_out]
    if any({r["label"] for r in split} != {"good", "poor"} for split in (train, valid)):
        raise ValueError("Both splits require good and poor examples.")
    reserved = {" ".join(r["text"].casefold().split()) for r in rows}
    generated = synthetic_rows(train, reserved)
    DATA.mkdir(exist_ok=True)
    for name, split in (("supplied", train), ("generated", generated), ("validation", valid)):
        write_rows(DATA / f"{name}.jsonl", split)
        print(f"{name}: {len(split)} rows, {sum(r['label'] == 'good' for r in split)} good")


if __name__ == "__main__":
    main()
