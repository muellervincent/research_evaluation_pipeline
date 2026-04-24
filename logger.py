import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

console = Console()

def setup_logging(log_file: str = None) -> logging.Logger:
    """Sets up the custom Rich logger and mutes generic noisy SDK logs."""
    
    # Mute google-genai logs and AFC warnings
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    handlers = [RichHandler(console=console, rich_tracebacks=True, markup=True)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        handlers.append(file_handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger("agent")

def print_cognitive_step(title: str, content: str):
    """Outputs a visually appealing panel representing the agent's internal reasoning."""
    console.print(Panel(content, title=f"[bold green]{title}[/bold green]", border_style="green"))

def highlight_print(message: str):
    """Outputs important messages."""
    console.print(f"[bold cyan]{message}[/bold cyan]")
