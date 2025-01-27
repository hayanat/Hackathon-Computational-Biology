import csv
import os
from Bio import SeqIO
import numpy as np
from alignments_algorithms_score import *


# ----------------------------- Input & Parsing -----------------------------

def parse_fasta_files(file_paths, sequence_type):
    """
    Parse multiple FASTA files and return a dictionary of sequences.
    :param file_paths: List of file paths to FASTA files.
    :param sequence_type: Type of sequence being parsed ('mRNA' or 'Protein').
    :return: Dictionary {species_name: DNA_sequence or Protein_sequence}
    """
    print(f"\n[INFO] Entering parse_fasta_files function for {sequence_type} sequences.")
    species_sequences = {}
    for file_path in file_paths:
        print(f"Processing {sequence_type} file: {file_path}")
        for record in SeqIO.parse(file_path, "fasta"):
            species_name = os.path.basename(file_path).split('.')[0]  # Use file name as species name
            species_sequences[species_name] = str(record.seq).upper()
            print(f"Parsed {sequence_type} sequence for species: {species_name}, Length: {len(record.seq)}")
    print(f"[INFO] Finished parse_fasta_files function for {sequence_type} sequences.\n")
    return species_sequences


# ----------------------------- Alignment to Time Conversion -----------------------------

def estimate_divergence_time(alignment_score, max_score, substitution_rate=1.2e-9, scaling_factor=68.47):
    """
    Estimates the divergence time between two species based on the alignment score.

    :param alignment_score: The raw alignment score from sequence alignment (float).
    :param max_score: The maximum alignment score (self-alignment score) (float).
    :param substitution_rate: Substitution rate per site per year (float). Default is 1.2e-9 for humans and chimps.
    :param scaling_factor: Scaling factor to convert raw score to substitution count (float). Default is 68.47.

    :return: Estimated divergence time in years (float).
    """

    # Step 1: Convert alignment score to total substitutions (K)
    K = (max_score - alignment_score) / scaling_factor

    # Handle potential negative substitutions
    if K < 0:
        print("[WARNING] Alignment score exceeds maximum score. Setting divergence time to infinity.")
        return float('inf')

    # Step 2: Calculate substitutions per site (d)
    d = K / max_score  # Using max_score as the total possible substitutions

    # Step 3: Estimate divergence time using the molecular clock
    T_years = d / (2 * substitution_rate)

    return T_years


# ----------------------------- Score Computation -----------------------------

def compute_alignment_with_human(mrna_sequences, protein_sequences, BLOSUM62, substitution_rate=1.2e-9,
                                 scaling_factor=68.47):
    """
    Compute DNA, Codon, and Protein alignment scores and evolutionary time
    between Human and other species.
    :param mrna_sequences: Dictionary {species_name: mRNA_sequence}.
    :param protein_sequences: Dictionary {species_name: Protein_sequence}.
    :param BLOSUM62: Substitution matrix for protein alignment.
    :param substitution_rate: Substitution rate per site per year (float). Default: 1.2e-9.
    :param scaling_factor: Scaling factor to convert raw score to substitution count (float). Default: 68.47.
    :return: None (prints scores and times directly).
    """
    print("\n[INFO] Entering compute_alignment_with_human function.")
    mrna_species = list(mrna_sequences.keys())
    protein_species = list(protein_sequences.keys())

    print("[INFO] mRNA Species:", mrna_species)
    print("[INFO] Protein Species:", protein_species)

    # Ensure human is present in the dataset
    human_species = mrna_species[0]  # The first file is the human genome (mRNA)
    human_mrna_sequence = mrna_sequences[human_species]
    human_protein_sequence = protein_sequences[protein_species[0]]  # The first file is the human genome (Protein)

    # Compute max scores for each alignment type (human self-alignment)
    max_dna_score = dna_alignment(human_mrna_sequence, human_mrna_sequence)
    max_codon_score = codon_alignment(human_mrna_sequence, human_mrna_sequence, BLOSUM62)
    max_protein_score = protein_alignment(human_protein_sequence, human_protein_sequence, BLOSUM62)
    print(f"[INFO] Max DNA score (Human vs Human): {max_dna_score}")
    print(f"[INFO] Max Codon score (Human vs Human): {max_codon_score}")
    print(f"[INFO] Max Protein score (Human vs Human): {max_protein_score}")

    # Align human genome with each other species using indices
    for i in range(1, len(mrna_species)):  # Start from index 1 to skip human (index 0)
        other_species = mrna_species[i]
        print(f"\n[INFO] Aligning Human ({human_species}) with {other_species}")

        # DNA alignment
        dna_score = dna_alignment(human_mrna_sequence, mrna_sequences[other_species])
        dna_time_years = estimate_divergence_time(dna_score, max_dna_score, substitution_rate=substitution_rate,
                                                  scaling_factor=scaling_factor)
        dna_time_mya = dna_time_years / 1e6  # Convert to million years ago
        print(f"DNA Alignment Score (Human vs {other_species}): {dna_score}")
        print(f"DNA Evolutionary Time (Human vs {other_species}): {dna_time_mya:.2f} million years ago (mya)")

        # Codon alignment
        codon_score_val = codon_alignment(human_mrna_sequence, mrna_sequences[other_species], BLOSUM62)
        codon_time_years = estimate_divergence_time(codon_score_val, max_codon_score,
                                                    substitution_rate=substitution_rate, scaling_factor=scaling_factor)
        codon_time_mya = codon_time_years / 1e6  # Convert to million years ago
        print(f"Codon Alignment Score (Human vs {other_species}): {codon_score_val}")
        print(f"Codon Evolutionary Time (Human vs {other_species}): {codon_time_mya:.2f} million years ago (mya)")

        # Protein alignment
        if i < len(protein_species):  # Ensure the index is within bounds for protein_species
            other_protein_species = protein_species[i]
            protein_score_val = protein_alignment(human_protein_sequence, protein_sequences[other_protein_species],
                                                  BLOSUM62)
            protein_time_years = estimate_divergence_time(protein_score_val, max_protein_score,
                                                          substitution_rate=substitution_rate,
                                                          scaling_factor=scaling_factor)
            protein_time_mya = protein_time_years / 1e6  # Convert to million years ago
            print(f"Protein Alignment Score (Human vs {other_protein_species}): {protein_score_val}")
            print(
                f"Protein Evolutionary Time (Human vs {other_protein_species}): {protein_time_mya:.2f} million years ago (mya)")
        else:
            print(f"[WARNING] Protein sequence for {other_species} not found.")

    print("\n[INFO] Finished compute_alignment_with_human function.\n")

def save_matrix_to_csv(matrix, species, filename):
    """
    Save a matrix to a CSV file.
    :param matrix: 2D NumPy array containing the matrix.
    :param species: List of species names (for row/column headers).
    :param filename: Name of the CSV file to save.
    """
    print(f"[INFO] Saving matrix to {filename}.")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header row
        writer.writerow(["Species"] + species)

        # Write each row
        for i, row in enumerate(matrix):
            writer.writerow([species[i]] + list(row))
    print(f"[INFO] Matrix saved to {filename}.")

def compute_pairwise_time_matrix(mrna_sequences, protein_sequences, BLOSUM62, k=0.01):
    """
    Compute the evolutionary time alignment matrix for all species based on DNA, Codon, and Protein alignments.
    :param mrna_sequences: Dictionary {species_name: mRNA_sequence}.
    :param protein_sequences: Dictionary {species_name: Protein_sequence}.
    :param BLOSUM62: Substitution matrix for protein alignment.
    :param k: Molecular clock rate (default: 0.01 substitutions per site per million years).
    :return: None (saves matrices to CSV files).
    """
    print("\n[INFO] Entering compute_pairwise_time_matrix function.")
    mrna_species = list(mrna_sequences.keys())
    protein_species = list(protein_sequences.keys())
    n = len(mrna_species)

    # Initialize matrices
    dna_time_matrix = np.zeros((n, n))
    codon_time_matrix = np.zeros((n, n))
    protein_time_matrix = np.zeros((n, n))

    # Compute max scores for each species (self-alignment)
    max_scores = {}
    for i in range(n):
        max_dna_score = dna_alignment(mrna_sequences[mrna_species[i]], mrna_sequences[mrna_species[i]])
        max_codon_score = codon_alignment(mrna_sequences[mrna_species[i]], mrna_sequences[mrna_species[i]], BLOSUM62)
        max_protein_score = protein_alignment(protein_sequences[protein_species[i]], protein_sequences[protein_species[i]], BLOSUM62)
        max_scores[mrna_species[i]] = {
            "max_dna_score": max_dna_score,
            "max_codon_score": max_codon_score,
        }
        max_scores[protein_species[i]] = {
            "max_protein_score": max_protein_score,
        }
        print(f"[INFO] Max scores for {mrna_species[i]}: DNA={max_dna_score}, Codon={max_codon_score}")
        print(f"[INFO] Max scores for {protein_species[i]}: Protein={max_protein_score}")

    # Compute pairwise evolutionary times
    for i in range(n):
        for j in range(i + 1, n):
            # DNA alignment
            dna_score = dna_alignment(mrna_sequences[mrna_species[i]], mrna_sequences[mrna_species[j]])
            dna_time = estimate_divergence_time(dna_score, max_scores[mrna_species[i]]["max_dna_score"], k)
            dna_time_matrix[i, j] = dna_time_matrix[j, i] = dna_time

            # Codon alignment
            codon_score = codon_alignment(mrna_sequences[mrna_species[i]], mrna_sequences[mrna_species[j]], BLOSUM62)
            codon_time = estimate_divergence_time(codon_score, max_scores[mrna_species[i]]["max_codon_score"], k)
            codon_time_matrix[i, j] = codon_time_matrix[j, i] = codon_time

            # Protein alignment
            protein_score = protein_alignment(protein_sequences[protein_species[i]], protein_sequences[protein_species[j]], BLOSUM62)
            protein_time = estimate_divergence_time(protein_score, max_scores[protein_species[i]]["max_protein_score"], k)
            protein_time_matrix[i, j] = protein_time_matrix[j, i] = protein_time

    # Save matrices to CSV files
    save_matrix_to_csv(dna_time_matrix, mrna_species, "dna_time_matrix.csv")
    save_matrix_to_csv(codon_time_matrix, mrna_species, "codon_time_matrix.csv")
    save_matrix_to_csv(protein_time_matrix, protein_species, "protein_time_matrix.csv")
    print("\n[INFO] Finished compute_pairwise_time_matrix function.\n")
# ----------------------------- Main -----------------------------

if __name__ == '__main__':
    # List of FASTA file paths for mRNA (Human must be the first file)
    mrna_fasta_files = [
        "final_data/gapdh/mRna_Homo_sapiens.fasta",
        "final_data/gapdh/mRna_Felis_catus.fasta",
        "final_data/gapdh/mRna_Columba_livia.fasta",
        "final_data/gapdh/mRna_Macaca_mulatta.fasta",
        "final_data/gapdh/mRna_Mus_musculus.fasta",
        "final_data/gapdh/mRna_Pongo_abelii.fasta",
        "final_data/gapdh/mRna_Rattus.fasta"
    ]

    # List of FASTA file paths for Protein (Human must be the first file)
    protein_fasta_files = [
        "final_data/gapdh/protien_Homo_sapiens.fasta",
        "final_data/gapdh/protien_Felis_catus.fasta",
        "final_data/gapdh/protien_Columba_livia.fasta",
        "final_data/gapdh/protien_Macaca_mulatta.fasta",
        "final_data/gapdh/protien_Mus_musculus.fasta",
        "final_data/gapdh/protien_Pongo_abelii.fasta",
        "final_data/gapdh/protien_Rattus.fasta"
    ]

    print("\n[INFO] Initializing program.")

    # Add stop penalties to the BLOSUM62 matrix
    print("[INFO] Adding stop penalties to BLOSUM62.")
    BLOSUM62 = add_stop_penalties(BLOSUM62_NO_STOP, stop_penalty=-5)

    # Parse FASTA files for mRNA and Protein sequences
    mrna_sequences = parse_fasta_files(mrna_fasta_files, "mRNA")
    protein_sequences = parse_fasta_files(protein_fasta_files, "Protein")

    # Compute alignment scores with Human
    compute_alignment_with_human(
        mrna_sequences,
        protein_sequences,
        BLOSUM62,
        substitution_rate=1.2e-9,  # Substitution rate per site per year
        scaling_factor=68.47  # Calibrated scaling factor
    )
    compute_pairwise_time_matrix(mrna_sequences, protein_sequences, BLOSUM62, k=0.01)


    print("[INFO] Program completed.")
