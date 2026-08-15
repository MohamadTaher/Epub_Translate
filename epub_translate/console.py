"""
The terminal: what a run prints, and the one question it asks.

All escape codes live here. `confirm` sits beside the logger rather than in the
CLI because it writes to the same terminal in the same colours, and a run's
output should be describable by naming one module.
"""

# level -> (label, label color, message color)
_LEVELS = {
    'system':      ('SYSTEM',      '96',    '97'),
    'api':         ('API',         '94',    '97'),
    'progress':    ('PROGRESS',    '92',    '97'),
    'concurrency': ('CONCURRENCY', '95',    '97'),
    'rate_limit':  ('RATE_LIMIT',  '93',    '33'),
    'interrupt':   ('INTERRUPT',   '41;97', '91'),
    'error':       ('ERROR',       '91',    '31'),
    'warning':     ('WARNING',     '93',    '33'),
    'success':     ('SUCCESS',     '92',    '32'),
    'info':        ('INFO',        '96',    '97'),
}


class ColoredLogger:
    """Logger with colored output for different message types."""

    def log(self, level: str, message: str):
        """
        Print at a level named at runtime.

        Public because a caller relaying events carries the level as a string:
        looking the name up in `_LEVELS` makes that table the authority on which
        levels exist, rather than whichever of the methods below happen to be
        defined.
        """
        label, label_color, message_color = _LEVELS[level]
        print(f"\033[{label_color}m[{label}]\033[0m \033[{message_color}m{message}\033[0m")

    def system(self, message: str): self.log('system', message)
    def api(self, message: str): self.log('api', message)
    def progress(self, message: str): self.log('progress', message)
    def concurrency(self, message: str): self.log('concurrency', message)
    def rate_limit(self, message: str): self.log('rate_limit', message)
    def interrupt(self, message: str): self.log('interrupt', message)
    def error(self, message: str): self.log('error', message)
    def warning(self, message: str): self.log('warning', message)
    def success(self, message: str): self.log('success', message)
    def info(self, message: str): self.log('info', message)


def confirm(question: str) -> bool:
    """Ask a yes/no question at the terminal."""
    return input(f"\n\033[93m{question} (yes/no): \033[0m").strip().lower() == 'yes'


logger = ColoredLogger()
