"""Alexandria CLI — entry point for all commands."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from alexandria.config import Config

console = Console()


@click.group()
@click.version_option(package_name="alexandria")
def main() -> None:
    """Alexandria — semantic code search for AI agents."""


@main.command()
@click.option("--context", "-c", required=True, help="Name for this index context.")
@click.option("--chunk-lines", default=50, show_default=True, help="Lines per chunk (sliding window).")
@click.option("--chunk-overlap", default=10, show_default=True, help="Overlap lines between chunks.")
@click.option("--follow-symlinks", is_flag=True, default=False, help="Follow symbolic links.")
@click.argument("path", type=click.Path(exists=True, file_okay=False))
def index(context: str, chunk_lines: int, chunk_overlap: int, follow_symlinks: bool, path: str) -> None:
    """Index a codebase into a named vector database context."""
    from alexandria.chunker import chunk_file
    from alexandria.discovery import discover_files
    from alexandria.embedder import Embedder
    from alexandria.store import Store

    config = Config(
        chunk_lines=chunk_lines,
        chunk_overlap=chunk_overlap,
        follow_symlinks=follow_symlinks,
    )

    repo_root = Path(path).resolve()

    # Verify services
    store = Store(config)
    if not store.is_available():
        console.print("[red]Error:[/red] Cannot connect to Qdrant at {}.".format(config.qdrant_url))
        console.print("Start Qdrant first: [bold]qdrant[/bold]")
        sys.exit(1)

    embedder = Embedder(config)
    if not embedder.is_available():
        console.print("[red]Error:[/red] Ollama model '{}' not available.".format(config.embed_model))
        console.print("Run: [bold]alexandria setup[/bold]")
        sys.exit(1)

    # Discover files
    console.print(f"[bold]Discovering files[/bold] in {repo_root}...")
    files = discover_files(repo_root, follow_symlinks=follow_symlinks)
    console.print(f"  Found [cyan]{len(files)}[/cyan] files")

    # Get existing file hashes for change detection
    existing_hashes = store.get_indexed_file_hashes(context)

    # Determine which files need (re-)indexing
    files_to_index: list[Path] = []
    files_skipped = 0
    for f in files:
        rel = str(f.relative_to(repo_root))
        try:
            current_hash = hashlib.sha256(f.read_bytes()).hexdigest()
        except (OSError, PermissionError):
            continue
        if rel in existing_hashes and existing_hashes[rel] == current_hash:
            files_skipped += 1
        else:
            files_to_index.append(f)

    if files_skipped > 0:
        console.print(f"  Skipping [dim]{files_skipped}[/dim] unchanged files")

    if not files_to_index:
        console.print("[green]All files up to date.[/green] Nothing to index.")
        return

    console.print(f"  Indexing [cyan]{len(files_to_index)}[/cyan] files into context [bold]{context}[/bold]")

    # Phase 1: Chunk all files
    from alexandria.chunker import Chunk

    all_chunks: list[Chunk] = []
    files_with_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking files...", total=len(files_to_index))
        for f in files_to_index:
            rel = str(f.relative_to(repo_root))
            # Delete old points for this file before re-indexing
            if rel in existing_hashes:
                store.delete_file_points(context, rel)

            chunks = chunk_file(f, config, repo_root)
            if chunks:
                all_chunks.extend(chunks)
                files_with_chunks += 1
            progress.advance(task)

    console.print(f"  Produced [cyan]{len(all_chunks)}[/cyan] chunks from [cyan]{files_with_chunks}[/cyan] files")

    if not all_chunks:
        console.print("[yellow]Warning:[/yellow] No chunks produced. Nothing to embed.")
        return

    # Phase 2: Embed all chunks
    texts = [c.text for c in all_chunks]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(texts))
        vectors: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vectors = embedder.embed_batch(batch, batch_size=len(batch))
            vectors.extend(batch_vectors)
            progress.advance(task, len(batch))

    # Phase 3: Store in Qdrant
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Storing vectors...")
        n_stored = store.upsert_chunks(context, all_chunks, vectors)
        progress.advance(task)

    console.print(f"\n[green]Done![/green] Indexed [bold]{n_stored}[/bold] chunks into context [cyan]{context}[/cyan]")


@main.command()
def serve() -> None:
    """Start the MCP server (stdio transport)."""
    import sys
    from rich.console import Console as RichConsole
    from alexandria.mcp_server import run_stdio

    # MCP stdio server writes to stdout, so only log to stderr
    err_console = RichConsole(stderr=True)
    err_console.print("[bold]Starting Alexandria MCP server[/bold] (stdio)")
    run_stdio()


@main.command()
@click.option("--context", "-c", required=True, help="Context to drop.")
@click.confirmation_option(prompt="Are you sure you want to drop this context?")
def drop(context: str) -> None:
    """Delete a context's collection from the vector database."""
    from alexandria.store import Store

    config = Config()
    store = Store(config)

    if not store.is_available():
        console.print("[red]Error:[/red] Cannot connect to Qdrant.")
        sys.exit(1)

    if store.drop_context(context):
        console.print(f"[green]Dropped[/green] context [cyan]{context}[/cyan]")
    else:
        console.print(f"[yellow]Context '{context}' not found.[/yellow]")


@main.command(name="list")
def list_contexts() -> None:
    """Show all indexed contexts and their stats."""
    from alexandria.store import Store

    config = Config()
    store = Store(config)

    if not store.is_available():
        console.print("[red]Error:[/red] Cannot connect to Qdrant.")
        sys.exit(1)

    contexts = store.list_contexts()

    if not contexts:
        console.print("[dim]No contexts indexed yet.[/dim]")
        console.print("Run: [bold]alexandria index --context NAME /path/to/code[/bold]")
        return

    table = Table(title="Indexed Contexts")
    table.add_column("Context", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")

    for ctx in sorted(contexts):
        stats = store.get_context_stats(ctx)
        table.add_row(ctx, str(stats["points"]), str(stats["status"]))

    console.print(table)


@main.command()
def setup() -> None:
    """Pull the Ollama embedding model and verify Qdrant connectivity."""
    from alexandria.embedder import Embedder
    from alexandria.store import Store

    config = Config()

    # Check Qdrant
    console.print("[bold]Checking Qdrant...[/bold]", end=" ")
    store = Store(config)
    if store.is_available():
        console.print("[green]OK[/green]")
    else:
        console.print("[red]FAILED[/red]")
        console.print(f"  Cannot connect to Qdrant at {config.qdrant_url}")
        console.print("  Start Qdrant: [bold]qdrant[/bold]")

    # Check Ollama and pull model
    console.print(f"[bold]Checking Ollama model '{config.embed_model}'...[/bold]", end=" ")
    embedder = Embedder(config)
    if embedder.is_available():
        console.print("[green]already pulled[/green]")
    else:
        console.print("[yellow]pulling...[/yellow]")
        try:
            embedder.pull_model()
            console.print(f"  [green]Pulled {config.embed_model}[/green]")
        except Exception as e:
            console.print(f"  [red]Failed to pull model:[/red] {e}")
            console.print("  Make sure Ollama is running: [bold]ollama serve[/bold]")
            sys.exit(1)

    console.print("\n[green]Setup complete![/green]")


if __name__ == "__main__":
    main()
