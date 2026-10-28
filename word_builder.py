"""
ECHO HANDS — Word Builder (Phase 1)
Accurately accumulates recognized letters into words with hold-to-confirm debouncing.
Supports all letters including 'E' and 'A' without conflicts.
"""

import time


class WordBuilder:
    def __init__(self, hold_seconds=0.7, grace_seconds=0.3):
        self.word = ""
        self.candidate_letter = ""
        self.letter_start_time = None
        self.last_seen_time = None
        self.committed = False
        self.hold_seconds = hold_seconds
        self.grace_seconds = grace_seconds
        self.progress = 0.0

    def update(self, letter, fingers=None):
        """
        Updates the candidate letter tracking.
        When a letter is held steadily for `hold_seconds`, it is committed to `self.word`.
        """
        now = time.time()

        if letter:
            if letter != self.candidate_letter:
                # Switched to a new letter -> start fresh hold cycle
                self.candidate_letter = letter
                self.letter_start_time = now
                self.last_seen_time = now
                self.committed = False
                self.progress = 0.0
            else:
                # Same letter sustained
                self.last_seen_time = now
                if not self.committed and self.letter_start_time is not None:
                    elapsed = now - self.letter_start_time
                    self.progress = min(1.0, elapsed / self.hold_seconds)
                    if elapsed >= self.hold_seconds:
                        self.word += letter
                        self.committed = True
                        self.progress = 1.0

        else:
            # No letter recognized in this frame (hand relaxed or transition)
            if self.last_seen_time and (now - self.last_seen_time < self.grace_seconds):
                # Within grace window -> tolerate single frame drops
                pass
            else:
                # Grace period expired -> reset candidate
                self.candidate_letter = ""
                self.letter_start_time = None
                self.committed = False
                self.progress = 0.0

        return self.word

    def add_space(self):
        """Adds a space to separate words."""
        if self.word and not self.word.endswith(" "):
            self.word += " "
        self.candidate_letter = ""
        self.letter_start_time = None
        self.committed = False
        self.progress = 0.0
        return self.word

    def backspace(self):
        """Deletes the last character."""
        if self.word:
            self.word = self.word[:-1]
        self.candidate_letter = ""
        self.letter_start_time = None
        self.committed = False
        self.progress = 0.0
        return self.word

    def clear(self):
        """Clears the entire word text."""
        self.word = ""
        self.candidate_letter = ""
        self.letter_start_time = None
        self.last_seen_time = None
        self.committed = False
        self.progress = 0.0
        return self.word
