"""Alexandria CLI — entry point for all commands."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from alexandria.config import Config

console = Console()


@click.group()
@click.version_option(package_name="alexandria")
def main() -> None:
    """Alexandria — semantic code search for AI agents."""


@main.command()
@click.option("--context", "-c", required=True, help="Name for this index context.")
@click.option(
    "--chunk-lines",
    default=50,
    show_default=True,
    help="Lines per chunk (sliding window).",
)
@click.option(
    "--chunk-overlap",
    default=10,
    show_default=True,
    help="Overlap lines between chunks.",
)
@click.option("--follow-symlinks", is_flag=True, default=False, help="Follow symbolic links.")
@click.argument("path", type=click.Path(exists=True, file_okay=False))
def index(
    context: str,
    chunk_lines: int,
    chunk_overlap: int,
    follow_symlinks: bool,
    path: str,
) -> None:
    """Index a codebase into a named vector database context."""
    from alexandria.chunker import chunk_file
    from alexandria.discovery import discover_files
    from alexandria.embedder import Embedder
    from alexandria.store import Store

    # Wire Python logging through Rich so embedder warnings render nicely.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[logging.StreamHandler(console.file)],
        force=True,
    )

    config = Config(
        chunk_lines=chunk_lines,
        chunk_overlap=chunk_overlap,
        follow_symlinks=follow_symlinks,
    )

    repo_root = Path(path).resolve()

    # Load project-level config (.alexandria.yml) if present
    from alexandria.config import PROJECT_CONFIG_FILE, load_project_config

    load_project_config(repo_root, config)
    if config.ignore_patterns:
        console.print(
            f"  Loaded [cyan]{len(config.ignore_patterns)}[/cyan] ignore pattern(s) "
            f"from [dim]{PROJECT_CONFIG_FILE}[/dim]"
        )

    # Verify services
    store = Store(config)
    if not store.is_available():
        console.print(f"[red]Error:[/red] Cannot connect to Qdrant at {config.qdrant_url}.")
        console.print("Start Qdrant first: [bold]qdrant[/bold]")
        sys.exit(1)

    embedder = Embedder(config)
    if not embedder.is_available():
        console.print(f"[red]Error:[/red] Ollama model '{config.embed_model}' not available.")
        console.print("Run: [bold]alexandria setup[/bold]")
        sys.exit(1)

    # Discover files
    console.print(f"[bold]Discovering files[/bold] in {repo_root}...")
    files = discover_files(
        repo_root,
        follow_symlinks=follow_symlinks,
        ignore_patterns=config.ignore_patterns,
    )
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

    console.print(
        f"  Indexing [cyan]{len(files_to_index)}[/cyan] files into context [bold]{context}[/bold]"
    )

    # Stream: chunk → embed → store in batches.
    # Instead of accumulating all chunks, then all vectors, then storing,
    # we process files in groups, embed each batch, and upsert immediately.
    # This reduces peak memory and overlaps work.
    from alexandria.chunker import Chunk

    embed_batch_size = 64
    total_chunks_stored = 0
    files_with_chunks = 0
    chunk_buffer: list[Chunk] = []

    def _flush_buffer(chunks: list[Chunk]) -> int:
        """Embed and store a buffer of chunks. Returns count stored."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        labels = [f"{c.file}:{c.start_line}-{c.end_line}" for c in chunks]
        vectors = embedder.embed_batch(
            texts,
            batch_size=embed_batch_size,
            labels=labels,
        )
        return store.upsert_chunks(context, chunks, vectors)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing files...", total=len(files_to_index))
        for f in files_to_index:
            rel = str(f.relative_to(repo_root))
            # Delete old points for this file before re-indexing
            if rel in existing_hashes:
                store.delete_file_points(context, rel)

            chunks = chunk_file(f, config, repo_root)
            if chunks:
                chunk_buffer.extend(chunks)
                files_with_chunks += 1

            # Flush when the buffer is large enough for an efficient embed call
            if len(chunk_buffer) >= embed_batch_size:
                total_chunks_stored += _flush_buffer(chunk_buffer)
                desc = f"Indexing files... ({total_chunks_stored} chunks)"
                progress.update(task, description=desc)
                chunk_buffer = []

            progress.advance(task)

        # Flush remaining chunks
        total_chunks_stored += _flush_buffer(chunk_buffer)

    console.print(
        f"\n[green]Done![/green] Indexed [bold]{total_chunks_stored}[/bold] chunks "
        f"from [cyan]{files_with_chunks}[/cyan] files into context [cyan]{context}[/cyan]"
    )


@main.command()
def serve() -> None:
    """Start the MCP server (stdio transport)."""
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
        sys.exit(1)

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
