import logging
import sys
import threading
import time


class StreamToLogger:
    """
    A wrapper that redirects sys.stdout and sys.stderr to a logger.
    This ensures that print() statements and exceptions are captured.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, message: str):
        # Process the incoming message chunk by chunk
        while message:
            # Find the first occurrence of either \n or \r
            nl_index = message.find('\n')
            cr_index = message.find('\r')

            # Find the index of the *first* delimiter
            if nl_index == -1 and cr_index == -1:
                # Case 1: No delimiters. Append to buffer and stop processing.
                self.linebuf += message
                break  # Exit the while loop

            if cr_index != -1 and (cr_index < nl_index or nl_index == -1):
                # Case 2: \r is the first delimiter
                index = cr_index
                delimiter_char = '\r'
            else:
                # Case 3: \n is the first delimiter
                index = nl_index
                delimiter_char = '\n'

            # Extract the part of the message *before* the delimiter
            line_part = message[:index]

            # Combine with the existing buffer, log it, and reset the buffer
            line_to_log = self.linebuf + line_part
            if line_to_log:
                # Log the complete line (without the delimiter)
                self.logger.log(self.level, line_to_log)

            # Reset the line buffer because we've hit a terminator
            # Both \n and \r signal the end of a "line" in this context.
            self.linebuf = ""

            # Continue processing the rest of the message *after* the delimiter
            message = message[index + 1:]

    def flush(self):
        # Flush the remaining content in the buffer.
        # This is important for cases where the output doesn't end with a newline.
        if self.linebuf:
            self.logger.log(self.level, self.linebuf)
            self.linebuf = ''


def setup_logging_v1(
        log_file: str = "training.log",
        console_level: int = logging.INFO,
        file_level: int = logging.INFO,
        capture_print: bool = True
):
    """
    Configures a root logger with a console handler and a file handler.
    Optionally redirects sys.stdout and sys.stderr to the logger.

    Args:
        log_file (str): The path to the log file.
        console_level (int): The logging level for the console output.
        file_level (int): The logging level for the file output.
        capture_print (bool): If True, redirects sys.stdout and sys.stderr
                              to the logger to capture print() statements.

    Returns:
        logging.Logger: The configured root logger.
    """
    # Get the root logger
    logger = logging.getLogger(__name__)
    # Set the minimum level for the logger to receive all messages
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicate output if the function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler, it will not use the redirected sys.stdout below, so no infinite loop
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    # Define the formatter for console handlers
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    # Define the formatter for file handlers
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Redirect sys.stdout and sys.stderr to the logger
    if capture_print:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger

def setup_logging(
        log_file: str = "training.log",
        console_level: int = logging.INFO,
        file_level: int = logging.INFO,
        capture_print: bool = True
):
    """
    Configures a logger that is safe to call multiple times.
    Prevents recursion by binding console handler to sys.__stdout__.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Remove and close existing handlers to prevent duplicates and FD leaks
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.flush()
            h.close()
        except Exception:
            pass

    # Always use original std streams for console handler
    orig_stdout = sys.__stdout__
    orig_stderr = sys.__stderr__

    console_handler = logging.StreamHandler(orig_stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Redirect stdout/stderr only once to avoid recursion
    if capture_print:
        if not isinstance(sys.stdout, StreamToLogger):
            sys.stdout = StreamToLogger(logger, logging.INFO)
        if not isinstance(sys.stderr, StreamToLogger):
            sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger


if __name__ == "__main__":
    # Configure logging and capture print
    # The relative path will be handled by the OS
    logger = setup_logging("../../results/test.log", capture_print=True)

    # Now, all prints will be written to the log file + console!
    print("🌟 This print will be logged!")
    print("🚀 Starting training...")

    # You can also directly use logging
    logger.info("This is from logger.info")
    logger.warning("This is a warning")
    logger.error("This is an error")


    # multiprocessing
    def worker(name):
        for i in range(3):
            print(f"🧵 Thread {name} - Step {i}")
            time.sleep(0.1)
            if i == 1:
                logger.error(f"🔥 Error in thread {name}")
                print(f'i am worker {i}')


    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(f"Worker-{i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # The flush method on LoggerWriter will ensure any pending output is logged
    # before the script exits.
    sys.stdout.flush()
    sys.stderr.flush()

    print("✅ All threads finished.")
