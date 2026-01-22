"""Lightweight in-code datasets for batch experiments."""

from __future__ import annotations

import random
from typing import Dict, List

MATH_DATASET: List[Dict[str, str]] = [
    {"question": "John has 3 apples and buys 2 more. How many apples does he have now?", "answer": "5"},
    {"question": "What is 10 + 15?", "answer": "25"},
    {"question": "A box has 12 eggs. If you eat 5, how many are left?", "answer": "7"},
    {"question": "Convert 10 kilograms to pounds. (Use 1 kg ≈ 2.2 lbs)", "answer": "22"},
    {"question": "A rectangle is 4 cm by 6 cm. What is its area?", "answer": "24"},
    {"question": "Sarah had 20 candies and gave away 8. How many does she have now?", "answer": "12"},
    {"question": "What is 9 times 7?", "answer": "63"},
    {"question": "If a car travels 60 miles per hour for 2 hours, how far does it go?", "answer": "120"},
    {"question": "Simplify: 45 - 18", "answer": "27"},
    {"question": "What is the perimeter of a square with side length 5?", "answer": "20"},
    {"question": "What is 100 divided by 4?", "answer": "25"},
    {"question": "Add 3.5 and 2.25", "answer": "5.75"},
    {"question": "A shirt costs $35 and is on sale for $5 off. What is the sale price?", "answer": "30"},
    {"question": "If a pizza is cut into 8 slices and you eat 3, how many slices remain?", "answer": "5"},
    {"question": "Convert 5 miles to kilometers (1 mile ≈ 1.6 km).", "answer": "8"},
    {"question": "What is 12 + 6 + 4?", "answer": "22"},
    {"question": "If a book has 240 pages and you read 30 each day, how many days to finish?", "answer": "8"},
    {"question": "Simplify: 9 + 8 - 3", "answer": "14"},
    {"question": "What is 50% of 80?", "answer": "40"},
    {"question": "If you triple 7, what do you get?", "answer": "21"},
]

REASONING_DATASET: List[Dict[str, str]] = [
    {"question": "Can a person safely drive a car while sleeping?", "answer": "no"},
    {"question": "If it is raining, are the streets likely to be wet?", "answer": "yes"},
    {"question": "Does a banana usually grow underground?", "answer": "no"},
    {"question": "Is water typically a liquid at room temperature?", "answer": "yes"},
    {"question": "Can a glass easily break if you drop it on the floor?", "answer": "yes"},
    {"question": "Would you expect to find snow in the Sahara Desert regularly?", "answer": "no"},
    {"question": "Does the sun rise in the east?", "answer": "yes"},
    {"question": "Is it safe to stare directly at the sun?", "answer": "no"},
    {"question": "Do most birds have feathers?", "answer": "yes"},
    {"question": "Can a fish breathe air like humans without special adaptation?", "answer": "no"},
    {"question": "Is boiling water typically hot?", "answer": "yes"},
    {"question": "Would you expect a rock to float in water?", "answer": "no"},
    {"question": "Can a person usually lift a car by hand?", "answer": "no"},
    {"question": "Does a tree need sunlight to grow?", "answer": "yes"},
    {"question": "Is the sky green on most days?", "answer": "no"},
    {"question": "Can a human breathe underwater without equipment?", "answer": "no"},
    {"question": "Are penguins birds?", "answer": "yes"},
    {"question": "Does fire typically feel cold?", "answer": "no"},
    {"question": "Is chocolate generally sweet?", "answer": "yes"},
    {"question": "Can a laptop run without any power source or battery?", "answer": "no"},
]


def _sample(dataset: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    items = dataset.copy()
    random.shuffle(items)
    return items[:n]


def sample_math(n: int) -> List[Dict[str, str]]:
    return _sample(MATH_DATASET, n)


def sample_reasoning(n: int) -> List[Dict[str, str]]:
    return _sample(REASONING_DATASET, n)
