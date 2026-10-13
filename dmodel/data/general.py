"""Small general-knowledge regression set.

Used to check that training the memory does not damage what the frozen base
model already knows. Two probes:
  - GENERAL_QA: multiple-choice questions (accuracy should not drop)
  - GENERAL_TEXT: plain sentences (mean NLL should not rise noticeably)
"""
from __future__ import annotations

from .facts import Question

_QA = [
    ("What is the capital of France?", ["Paris", "London", "Berlin", "Madrid"], 0),
    ("What is the capital of Japan?", ["Seoul", "Tokyo", "Beijing", "Bangkok"], 1),
    ("What is the largest planet in the solar system?", ["Mars", "Earth", "Jupiter", "Venus"], 2),
    ("How many legs does a spider have?", ["six", "eight", "ten", "four"], 1),
    ("What gas do plants absorb from the air?", ["oxygen", "nitrogen", "helium", "carbon dioxide"], 3),
    ("What is the chemical symbol for gold?", ["Au", "Ag", "Gd", "Go"], 0),
    ("What is the freezing point of water in Celsius?", ["0", "10", "32", "100"], 0),
    ("Which ocean is the largest?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
    ("Who wrote Romeo and Juliet?", ["Dickens", "Shakespeare", "Tolstoy", "Austen"], 1),
    ("What is the square root of 81?", ["7", "8", "9", "11"], 2),
    ("Which planet is known as the Red Planet?", ["Mars", "Saturn", "Neptune", "Mercury"], 0),
    ("What is the boiling point of water in Celsius?", ["50", "90", "100", "212"], 2),
    ("How many continents are there?", ["five", "six", "seven", "nine"], 2),
    ("What is the hardest natural substance?", ["gold", "iron", "diamond", "quartz"], 2),
    ("What language is mainly spoken in Brazil?", ["Spanish", "Portuguese", "French", "Italian"], 1),
    ("What is the largest mammal?", ["elephant", "blue whale", "giraffe", "hippo"], 1),
    ("How many days are in a leap year?", ["364", "365", "366", "367"], 2),
    ("What organ pumps blood through the body?", ["lungs", "liver", "heart", "kidney"], 2),
    ("What is 12 times 12?", ["124", "144", "132", "154"], 1),
    ("Which element has the atomic number 1?", ["helium", "oxygen", "hydrogen", "carbon"], 2),
    ("What is the capital of Italy?", ["Rome", "Milan", "Naples", "Venice"], 0),
    ("What do bees make?", ["silk", "honey", "milk", "wax paper"], 1),
    ("What is the longest river in Africa?", ["Congo", "Niger", "Nile", "Zambezi"], 2),
    ("How many minutes are in an hour?", ["30", "60", "90", "100"], 1),
    ("What color do you get by mixing blue and yellow?", ["green", "purple", "orange", "brown"], 0),
    ("Which animal is known as the king of the jungle?", ["tiger", "lion", "bear", "wolf"], 1),
    ("What is the opposite of hot?", ["cold", "warm", "wet", "bright"], 0),
    ("What star is at the center of our solar system?", ["Sirius", "the Sun", "Polaris", "Vega"], 1),
    ("How many sides does a hexagon have?", ["five", "six", "seven", "eight"], 1),
    ("What is the capital of Canada?", ["Toronto", "Vancouver", "Ottawa", "Montreal"], 2),
]

GENERAL_QA: list[Question] = [
    Question(prompt=f"Q: {q}\nA:", choices=[" " + c for c in cs], answer_idx=a,
             fact=None, kind="general")  # type: ignore[arg-type]
    for q, cs, a in _QA
]

GENERAL_TEXT: list[str] = [
    "The sun rises in the east and sets in the west.",
    "Water is made of hydrogen and oxygen atoms.",
    "A bicycle has two wheels, a frame, and a pair of pedals.",
    "Many people drink coffee in the morning to help them wake up.",
    "The library was quiet except for the sound of turning pages.",
    "Rain fell softly on the roof throughout the night.",
    "She opened the window to let the fresh air into the room.",
    "Computers store information as sequences of ones and zeros.",
    "The children laughed as they played in the park after school.",
    "Mountains are often covered in snow during the winter months.",
    "He wrote a letter to his grandmother to thank her for the gift.",
    "The train arrived at the station exactly on time.",
    "Bread is usually made from flour, water, yeast, and salt.",
    "The museum displayed paintings from several different centuries.",
    "Birds build nests in trees to protect their eggs.",
]
