"""The fixed word lists given to the detector. Kept apart from the detector itself so
that search and the tests can use them without loading a model library."""

# Who makes a moment worth indexing.
SUBJECTS = ["person"]

# Things worth tagging in a home, whatever the caption says. Every key frame is checked
# for these once, at indexing time.
HOME_VOCAB = [
    "person", "refrigerator", "oven", "microwave", "stove", "sink", "cabinet", "drawer", "door", "window",
    "table", "chair", "sofa", "bed", "tv", "laptop", "phone", "bottle", "cup", "plate", "bowl", "pan", "pot",
    "knife", "egg carton", "milk carton", "food", "bag", "backpack", "box", "package", "key", "wallet",
    "remote control", "book", "shoe", "jacket", "umbrella", "broom", "trash can", "dog", "cat",
]
