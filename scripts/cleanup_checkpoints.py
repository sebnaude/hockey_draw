#!/usr/bin/env python
"""
Checkpoint cleanup and archival script.

Lists, archives, and deletes old solver checkpoint directories.

Usage:
    python scripts/cleanup_checkpoints.py                 # List all checkpoints
    python scripts/cleanup_checkpoints.py --keep 5        # Keep only 5 most recent
    python scripts/cleanup_checkpoints.py --archive       # Archive old to zip before deleting
    python scripts/cleanup_checkpoints.py --older-than 7  # Delete checkpoints older than 7 days
    python scripts/cleanup_checkpoints.py --dry-run       # Show what would be deleted
"""

import argparse
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, TypedDict


class CheckpointInfo(TypedDict):
    name: str
    path: Path
    modified: datetime
    size_bytes: int
    size_mb: float
    file_count: int


CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"


def get_checkpoint_info(checkpoint_dir: Path) -> CheckpointInfo:
    """Get info about a checkpoint directory."""
    stat = checkpoint_dir.stat()
    total_size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
    file_count = sum(1 for f in checkpoint_dir.rglob('*') if f.is_file())
    
    return {
        'name': checkpoint_dir.name,
        'path': checkpoint_dir,
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'size_bytes': total_size,
        'size_mb': total_size / (1024 * 1024),
        'file_count': file_count,
    }


def list_checkpoints() -> List[CheckpointInfo]:
    """List all checkpoint directories with metadata."""
    if not CHECKPOINTS_DIR.exists():
        return []
    
    checkpoints: List[CheckpointInfo] = []
    for item in CHECKPOINTS_DIR.iterdir():
        if item.is_dir():
            checkpoints.append(get_checkpoint_info(item))
    
    # Sort by modified time, newest first
    checkpoints.sort(key=lambda x: x['modified'], reverse=True)
    return checkpoints


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def print_checkpoints(checkpoints: List[CheckpointInfo]) -> None:
    """Print checkpoint list in a table."""
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    total_size = sum(c['size_bytes'] for c in checkpoints)
    
    print(f"\n{'Name':<20} {'Modified':<20} {'Size':<10} {'Files':<6}")
    print("-" * 60)
    
    for cp in checkpoints:
        print(f"{cp['name']:<20} {cp['modified'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{format_size(cp['size_bytes']):<10} {cp['file_count']:<6}")
    
    print("-" * 60)
    print(f"Total: {len(checkpoints)} checkpoints, {format_size(total_size)}")


def archive_checkpoint(checkpoint: CheckpointInfo, archive_dir: Path) -> Path:
    """Archive a checkpoint to a zip file."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    archive_name = f"{checkpoint['name']}_{checkpoint['modified'].strftime('%Y%m%d')}.zip"
    archive_path = archive_dir / archive_name
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in checkpoint['path'].rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(checkpoint['path'])
                zf.write(file_path, arcname)
    
    return archive_path


def delete_checkpoint(checkpoint: CheckpointInfo) -> None:
    """Delete a checkpoint directory."""
    shutil.rmtree(checkpoint['path'])


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup and archive solver checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--keep', type=int, metavar='N',
                        help='Keep only the N most recent checkpoints')
    parser.add_argument('--older-than', type=int, metavar='DAYS',
                        help='Delete checkpoints older than DAYS days')
    parser.add_argument('--archive', action='store_true',
                        help='Archive to zip before deleting')
    parser.add_argument('--archive-dir', type=str, default='checkpoints/archive',
                        help='Directory for archives (default: checkpoints/archive)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without doing it')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    checkpoints = list_checkpoints()
    
    # If no action specified, just list
    if args.keep is None and args.older_than is None:
        print_checkpoints(checkpoints)
        print("\nUse --keep N or --older-than DAYS to clean up.")
        return
    
    # Determine which to delete
    to_delete: List[CheckpointInfo] = []
    
    if args.keep is not None:
        if len(checkpoints) > args.keep:
            to_delete = checkpoints[args.keep:]
    
    if args.older_than is not None:
        cutoff = datetime.now() - timedelta(days=args.older_than)
        old_ones = [c for c in checkpoints if c['modified'] < cutoff]
        # Merge with existing to_delete
        existing_paths = {c['path'] for c in to_delete}
        for c in old_ones:
            if c['path'] not in existing_paths:
                to_delete.append(c)
    
    if not to_delete:
        print("No checkpoints to delete.")
        return
    
    # Show what will be deleted
    print(f"\nCheckpoints to {'archive and ' if args.archive else ''}delete:")
    print_checkpoints(to_delete)
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return
    
    # Confirm
    if not args.yes:
        response = input(f"\nDelete {len(to_delete)} checkpoint(s)? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Process deletions
    archive_dir = Path(args.archive_dir)
    
    for cp in to_delete:
        if args.archive:
            archive_path = archive_checkpoint(cp, archive_dir)
            print(f"Archived: {cp['name']} -> {archive_path}")
        
        delete_checkpoint(cp)
        print(f"Deleted: {cp['name']}")
    
    print(f"\nCleaned up {len(to_delete)} checkpoint(s).")
    
    # Show remaining
    remaining = list_checkpoints()
    if remaining:
        print(f"Remaining: {len(remaining)} checkpoint(s)")


if __name__ == "__main__":
    main()
