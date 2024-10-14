import click
from pathlib import Path
import numpy as np
from io import StringIO

@click.command()
@click.argument(
    "input_path", type=click.Path(exists=True, resolve_path=True)
)
@click.option("-f", "--overwrite", is_flag=True, help="overwrite existing output")
def cli(input_path, overwrite):
    """Convert AreTomo `aln` files to imod `xf` format. Supports both single file and directory inputs."""
    
    input_path = Path(input_path)
    
    # Check if the input is a directory
    if input_path.is_dir():
        external_dir = input_path / "external"
        
        if not external_dir.exists():
            raise FileNotFoundError(f"Expected 'external' folder not found in {input_path}")
        
        # Loop through all subdirectories in the 'external' folder
        for tomogram_dir in external_dir.iterdir():
            if tomogram_dir.is_dir():
                process_aln_files_in_dir(tomogram_dir, overwrite)
    
    # Check if the input is a single file
    elif input_path.is_file() and input_path.suffix == ".aln":
        # Process the single file
        process_aln_file(input_path, overwrite)
    else:
        raise ValueError(f"Invalid input: {input_path}. Provide a directory or a single .aln file.")
    

def process_aln_files_in_dir(tomogram_dir, overwrite):
    """Process all .aln files in a given directory."""
    aln_files = list(tomogram_dir.glob("*.aln"))
    if not aln_files:
        print(f"No .aln file found in {tomogram_dir}")
        return

    # Process each .aln file in the directory
    for aln_file in aln_files:
        process_aln_file(aln_file, overwrite)


def process_aln_file(aln_file, overwrite):
    """Process a single .aln file and convert it to a .xf file."""
    xf_file = aln_file.with_suffix(".xf")
    
    # Check if the output file exists and if overwrite is allowed
    if xf_file.exists() and not overwrite:
        print(f"{xf_file} already exists. Use --overwrite to replace it.")
        return

    # Read and process the .aln file
    txt = aln_file.read_text()
    if "# Local Alignment" in txt:
        txt = txt.partition("# Local Alignment")[0]
    data = np.loadtxt(StringIO(txt))
    
    # Convert angles and shifts
    angles = -np.radians(data[:, 1])
    shifts = data[:, [3, 4]]
    c, s = np.cos(angles), np.sin(angles)
    rot = np.empty((len(angles), 2, 2))
    rot[:, 0, 0] = c
    rot[:, 0, 1] = -s
    rot[:, 1, 0] = s
    rot[:, 1, 1] = c
    shifts_rot = np.einsum("ijk,ik->ij", rot, shifts)
    
    # Combine rotation and shift data
    out = np.concatenate([rot.reshape(-1, 4), -shifts_rot], axis=1)
    
    # Save the output to the .xf file
    np.savetxt(xf_file, out, ["%12.7f"] * 4 + ["%12.3f"] * 2)
    print(f"Converted {aln_file} to {xf_file}")


if __name__ == "__main__":
    cli()

