import numpy as np

# ----------------------------- Constants -----------------------------
# Scoring parameters
GAP_PENALTY = -2
MATCH_SCORE = 1
MISMATCH_SCORE = -1

FRAME_SHIFT_PENALTY = -10  # For 1- or 2-base shifts
CODON_GAP_PENALTY = -6    # For a full codon insertion/deletion

# Codon to amino acid map
CODON_TABLE = {
    # Stop codons directly in the same dictionary:
    "TAA": "*",
    "TAG": "*",
    "TGA": "*",
    # Standard codons:
    "TGG": "W",
    "TAC": "Y", "TAT": "Y",
    "TGC": "C", "TGT": "C",
    "GAA": "E", "GAG": "E",
    "AAA": "K", "AAG": "K",
    "CAA": "Q", "CAG": "Q",
    "AGC": "S", "AGT": "S", "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
    "TTA": "L", "TTG": "L", "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
    "AGA": "R", "AGG": "R", "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
    "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
    "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
    "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
    "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
    "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
    "ATA": "I", "ATC": "I", "ATT": "I",
    "TTC": "F", "TTT": "F",
    "GAC": "D", "GAT": "D",
    "CAC": "H", "CAT": "H",
    "AAC": "N", "AAT": "N",
    "ATG": "M"
}

# BLOSUM62 map
BLOSUM62_NO_STOP = {
    ('W', 'F'): 1, ('L', 'R'): -2, ('S', 'P'): -1, ('V', 'T'): 0,
    ('Q', 'Q'): 5, ('N', 'A'): -2, ('Z', 'Y'): -2, ('W', 'R'): -3,
    ('Q', 'A'): -1, ('S', 'D'): 0, ('H', 'H'): 8, ('S', 'H'): -1,
    ('H', 'D'): -1, ('L', 'N'): -3, ('W', 'A'): -3, ('Y', 'M'): -1,
    ('G', 'R'): -2, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3,
    ('Y', 'A'): -2, ('V', 'D'): -3, ('B', 'S'): 0, ('Y', 'Y'): 7,
    ('G', 'N'): 0, ('E', 'C'): -4, ('Y', 'Q'): -1, ('Z', 'Z'): 4,
    ('V', 'A'): 0, ('C', 'C'): 9, ('M', 'R'): -1, ('V', 'E'): -2,
    ('T', 'N'): 0, ('P', 'P'): 7, ('V', 'I'): 3, ('V', 'S'): -2,
    ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -2,
    ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -3, ('I', 'D'): -3,
    ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2,
    ('P', 'H'): -2, ('F', 'Q'): -3, ('Z', 'G'): -2, ('X', 'L'): -1,
    ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1, ('D', 'R'): -2,
    ('B', 'W'): -4, ('X', 'D'): -1, ('Z', 'K'): 1, ('F', 'A'): -2,
    ('Z', 'W'): -3, ('F', 'E'): -3, ('D', 'N'): 1, ('B', 'K'): 0,
    ('X', 'X'): -1, ('F', 'I'): 0, ('B', 'G'): -1, ('X', 'T'): 0,
    ('F', 'M'): 0, ('B', 'C'): -3, ('Z', 'I'): -3, ('Z', 'V'): -2,
    ('S', 'S'): 4, ('L', 'Q'): -2, ('W', 'E'): -3, ('Q', 'R'): 1,
    ('N', 'N'): 6, ('W', 'M'): -1, ('Q', 'C'): -3, ('W', 'I'): -3,
    ('S', 'C'): -1, ('L', 'A'): -1, ('S', 'G'): 0, ('L', 'E'): -3,
    ('W', 'Q'): -2, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 0,
    ('N', 'R'): 0, ('H', 'C'): -3, ('Y', 'N'): -2, ('G', 'Q'): -2,
    ('Y', 'F'): 3, ('C', 'A'): 0, ('V', 'L'): 1, ('G', 'E'): -2,
    ('G', 'A'): 0, ('K', 'R'): 2, ('E', 'D'): 2, ('Y', 'R'): -2,
    ('M', 'Q'): 0, ('T', 'I'): -1, ('C', 'D'): -3, ('V', 'F'): -1,
    ('T', 'A'): 0, ('T', 'P'): -1, ('B', 'P'): -2, ('T', 'E'): -1,
    ('V', 'N'): -3, ('P', 'G'): -2, ('M', 'A'): -1, ('K', 'H'): -1,
    ('V', 'R'): -3, ('P', 'C'): -3, ('M', 'E'): -2, ('K', 'L'): -2,
    ('V', 'V'): 4, ('M', 'I'): 1, ('T', 'Q'): -1, ('I', 'G'): -4,
    ('P', 'K'): -1, ('M', 'M'): 5, ('K', 'D'): -1, ('I', 'C'): -1,
    ('Z', 'D'): 1, ('F', 'R'): -3, ('X', 'K'): -1, ('Q', 'D'): 0,
    ('X', 'G'): -1, ('Z', 'L'): -3, ('X', 'C'): -2, ('Z', 'H'): 0,
    ('B', 'L'): -4, ('B', 'H'): 0, ('F', 'F'): 6, ('X', 'W'): -2,
    ('B', 'D'): 4, ('D', 'A'): -2, ('S', 'L'): -2, ('X', 'S'): 0,
    ('F', 'N'): -3, ('S', 'R'): -1, ('W', 'D'): -4, ('V', 'Y'): -1,
    ('W', 'L'): -2, ('H', 'R'): 0, ('W', 'H'): -2, ('H', 'N'): 1,
    ('W', 'T'): -2, ('T', 'T'): 5, ('S', 'F'): -2, ('W', 'P'): -4,
    ('L', 'D'): -4, ('B', 'I'): -3, ('L', 'H'): -3, ('S', 'N'): 1,
    ('B', 'T'): -1, ('L', 'L'): 4, ('Y', 'K'): -2, ('E', 'Q'): 2,
    ('Y', 'G'): -3, ('Z', 'S'): 0, ('Y', 'C'): -2, ('G', 'D'): -1,
    ('B', 'V'): -3, ('E', 'A'): -1, ('Y', 'W'): 2, ('E', 'E'): 5,
    ('Y', 'S'): -2, ('C', 'N'): -3, ('V', 'C'): -1, ('T', 'H'): -2,
    ('P', 'R'): -2, ('V', 'G'): -3, ('T', 'L'): -1, ('V', 'K'): -2,
    ('K', 'Q'): 1, ('R', 'A'): -1, ('I', 'R'): -3, ('T', 'D'): -1,
    ('P', 'F'): -4, ('I', 'N'): -3, ('K', 'I'): -3, ('M', 'D'): -3,
    ('V', 'W'): -3, ('W', 'W'): 11, ('M', 'H'): -2, ('P', 'N'): -2,
    ('K', 'A'): -1, ('M', 'L'): 2, ('K', 'E'): 1, ('Z', 'E'): 4,
    ('X', 'N'): -1, ('Z', 'A'): -1, ('Z', 'M'): -1, ('X', 'F'): -1,
    ('K', 'C'): -3, ('B', 'Q'): 0, ('X', 'B'): -1, ('B', 'M'): -3,
    ('F', 'C'): -2, ('Z', 'Q'): 3, ('X', 'Z'): -1, ('F', 'G'): -3,
    ('B', 'E'): 1, ('X', 'V'): -1, ('F', 'K'): -3, ('B', 'A'): -2,
    ('X', 'R'): -1, ('D', 'D'): 6, ('W', 'G'): -2, ('Z', 'F'): -3,
    ('S', 'Q'): 0, ('W', 'C'): -2, ('W', 'K'): -3, ('H', 'Q'): 0,
    ('L', 'C'): -1, ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4,
    ('W', 'S'): -3, ('S', 'E'): 0, ('H', 'E'): 0, ('S', 'I'): -2,
    ('H', 'A'): -2, ('S', 'M'): -1, ('Y', 'L'): -1, ('Y', 'H'): 2,
    ('Y', 'D'): -3, ('E', 'R'): 0, ('X', 'P'): -2, ('G', 'G'): 6,
    ('G', 'C'): -3, ('E', 'N'): 0, ('Y', 'T'): -2, ('Y', 'P'): -3,
    ('T', 'K'): -1, ('A', 'A'): 4, ('P', 'Q'): -1, ('T', 'C'): -1,
    ('V', 'H'): -3, ('T', 'G'): -2, ('I', 'Q'): -3, ('Z', 'T'): -1,
    ('C', 'R'): -3, ('V', 'P'): -2, ('P', 'E'): -1, ('M', 'C'): -1,
    ('K', 'N'): 0, ('I', 'I'): 4, ('P', 'A'): -1, ('M', 'G'): -3,
    ('T', 'S'): 1, ('I', 'E'): -3, ('P', 'M'): -2, ('M', 'K'): -1,
    ('I', 'A'): -1, ('P', 'I'): -3, ('R', 'R'): 5, ('X', 'M'): -1,
    ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 1, ('X', 'E'): -1,
    ('Z', 'N'): 0, ('X', 'A'): 0, ('B', 'R'): -1, ('B', 'N'): 3,
    ('F', 'D'): -3, ('X', 'Y'): -1, ('Z', 'R'): 0, ('F', 'H'): -1,
    ('B', 'F'): -3, ('F', 'L'): 0, ('X', 'Q'): -1, ('B', 'B'): 4
}
# ----------------------------- Alignments Functions -----------------------------

def dna_alignment(seq_a, seq_b):
    """
    Implements the Needleman-Wunsch (NW) algorithm for pairwise sequence alignment.

    :param seq_a: The first DNA sequence (string).
    :param seq_b: The second DNA sequence (string).
    :return: alignment_score: The optimal alignment score (float).
    """
    len_a, len_b = len(seq_a), len(seq_b)
    scoring_matrix = np.zeros((len_a + 1, len_b + 1), dtype=float)

    # Initialize gap penalties
    for i in range(1, len_a + 1):
        scoring_matrix[i][0] = i * GAP_PENALTY
    for j in range(1, len_b + 1):
        scoring_matrix[0][j] = j * GAP_PENALTY

    # Fill scoring matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            match = scoring_matrix[i - 1][j - 1] + (MATCH_SCORE if seq_a[i - 1] == seq_b[j - 1] else MISMATCH_SCORE)
            delete = scoring_matrix[i - 1][j] + GAP_PENALTY
            insert = scoring_matrix[i][j - 1] + GAP_PENALTY
            scoring_matrix[i][j] = max(match, delete, insert)

    return scoring_matrix[len_a][len_b]

def protein_alignment(protein_a, protein_b, substitution_matrix, gap_penalty=-2):
    """
    Perform Needleman–Wunsch alignment on two *DNA* sequences by:
      1) Translating each DNA sequence into its protein sequence.
      2) Aligning the resulting protein sequences using the given
         substitution matrix (e.g., BLOSUM62) and a gap penalty.
      3) Returning the aligned protein sequences and the alignment score.

    :param protein_a: First DNA sequence (string).
    :param protein_b: Second DNA sequence (string).
    :param substitution_matrix: Dictionary or 2D structure, e.g.:
                               substitution_matrix[(aa1, aa2)] = score
    :param gap_penalty: Penalty (integer) for introducing a gap.
    :return: alignment_score(float)
    """

    # 1) Translate DNA to protein
    # protein_a = translate_dna_to_protein(seq_a)
    # protein_b = translate_dna_to_protein(seq_b)

    # 2) Initialize DP matrix for Needleman-Wunsch
    len_a, len_b = len(protein_a), len(protein_b)
    scoring_matrix = np.zeros((len_a + 1, len_b + 1), dtype=float)

    # Fill first row/column with gap penalties
    for i in range(1, len_a + 1):
        scoring_matrix[i][0] = scoring_matrix[i - 1][0] + gap_penalty
    for j in range(1, len_b + 1):
        scoring_matrix[0][j] = scoring_matrix[0][j - 1] + gap_penalty

    # 3) Fill the scoring matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            aa_a = protein_a[i - 1]  # amino acid in protein_a
            aa_b = protein_b[j - 1]  # amino acid in protein_b

            # Look up the match/mismatch cost from the substitution matrix
            match_score = substitution_matrix.get((aa_a, aa_b), -1)  # default if missing

            diag = scoring_matrix[i - 1][j - 1] + match_score  # match/mismatch
            up = scoring_matrix[i - 1][j] + gap_penalty  # deletion
            left = scoring_matrix[i][j - 1] + gap_penalty  # insertion

            scoring_matrix[i][j] = max(diag, up, left)
    return scoring_matrix[len_a][len_b]

def codon_alignment(seq_a, seq_b, substitution_matrix):
    """
    Frameshift-aware, codon-based Needleman–Wunsch alignment.
    Allows partial codons (frameshifts), special penalties for stop codons, etc.

    :param seq_a: DNA sequence A (string)
    :param seq_b: DNA sequence B (string)
    :param substitution_matrix: e.g. BLOSUM62 (with stop penalties added)
    :return: alignment_score (float)
    """

    # 1) We'll do a DP matrix sized by raw nucleotide indices
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.full((len_a+1, len_b+1), float('-inf'))
    # For traceback, we store pointers (prev_i, prev_j) in a 2D list
    traceback = [[(0, 0) for _ in range(len_b+1)] for _ in range(len_a+1)]

    # 2) Initialize the DP boundaries
    dp[0][0] = 0.0
    # Fill first row (cost to align i bases of A to 0 bases of B)
    for i in range(3, len_a+1, 3):  # stepping by codons
        dp[i][0] = dp[i-3][0] + CODON_GAP_PENALTY
        traceback[i][0] = (i-3, 0)
    # If you want to allow frameshifts from the start (like 1 or 2 leftover),
    # you can also fill dp[1][0], dp[2][0] with frame shift penalties, etc.

    # Similarly fill first column for seq_b
    for j in range(3, len_b+1, 3):
        dp[0][j] = dp[0][j-3] + CODON_GAP_PENALTY
        traceback[0][j] = (0, j-3)

    # 3) Fill the DP matrix
    for i in range(len_a+1):
        for j in range(len_b+1):

            if i == 0 and j == 0:
                continue

            current_best = dp[i][j]  # Start with -inf or existing

            # -- Possible moves --
            #
            # (A) Codon vs. Codon (3 vs. 3) => normal alignment
            if i >= 3 and j >= 3:
                codonA = seq_a[i-3:i]
                codonB = seq_b[j-3:j]
                score_codon = codon_score(codonA, codonB, substitution_matrix)
                candidate = dp[i-3][j-3] + score_codon
                if candidate > current_best:
                    current_best = candidate
                    traceback[i][j] = (i-3, j-3)

            # (B) Frameshift in B: (3 bases in A vs. 1 or 2 in B)
            if i >= 3 and j >= 2:
                # 3 vs. 2
                candidate = dp[i-3][j-2] + FRAME_SHIFT_PENALTY
                if candidate > current_best:
                    current_best = candidate
                    traceback[i][j] = (i-3, j-2)
            if i >= 3 and j >= 1:
                # 3 vs. 1
                candidate = dp[i-3][j-1] + FRAME_SHIFT_PENALTY
                if candidate > current_best:
                    current_best = candidate
                    traceback[i][j] = (i-3, j-1)

            # (C) Frameshift in A: (1 or 2 in A vs. 3 in B)
            if i >= 2 and j >= 3:
                candidate = dp[i-2][j-3] + FRAME_SHIFT_PENALTY
                if candidate > current_best:
                    current_best = candidate
                    traceback[i][j] = (i-2, j-3)
            if i >= 1 and j >= 3:
                candidate = dp[i-1][j-3] + FRAME_SHIFT_PENALTY
                if candidate > current_best:
                    current_best = candidate
                    traceback[i][j] = (i-1, j-3)

            # (D) Codon-gap: 3 in A vs. 0 in B
            if i >= 3:
                candidate = dp[i-3][j] + CODON_GAP_PENALTY
                if candidate > current_best:
                    current_best = candidate
                    traceback[i][j] = (i-3, j)

            # (E) Gap-codon: 0 in A vs. 3 in B
            if j >= 3:
                candidate = dp[i][j-3] + CODON_GAP_PENALTY
                if candidate > current_best:
                    current_best = candidate
                    traceback[i][j] = (i, j-3)

            dp[i][j] = current_best

    return dp[len_a][len_b]


# ----------------------------- Helper Functions -----------------------------


def translate_dna_to_protein(dna_seq):
    """
    Translate DNA sequences into proteins.
    :param dna_seq: DNA sequence (string).
    :return: Protein sequence (string).
    """
    protein = []
    # We'll only loop up to the last full codon
    end = len(dna_seq) - (len(dna_seq) % 3)  # This strips off leftover bases, if any

    for i in range(0, end, 3):
        codon = dna_seq[i:i + 3].upper()
        if '-' in codon:
            protein.append('-')
        elif codon in CODON_TABLE:
            protein.append(CODON_TABLE[codon])
        else:
            protein.append('X')
    return "".join(protein)

def add_stop_penalties(substitution_matrix, stop_penalty=-5):
    """
    Adds entries to 'substitution_matrix' so that:
      - Aligning '*' with ANY amino acid => stop_penalty
    Returns the updated dictionary.
    """
    # Gather all existing amino-acid symbols found in the dictionary
    all_symbols = set()
    for (aa1, aa2) in substitution_matrix.keys():
        all_symbols.add(aa1)
        all_symbols.add(aa2)

    # Add '*' with each symbol to the matrix
    for symbol in all_symbols:
        substitution_matrix[('*', symbol)] = stop_penalty
        substitution_matrix[(symbol, '*')] = stop_penalty
    # If you want '*' vs. '*' to have the same penalty (or something else), do:
    substitution_matrix[('*', '*')] = stop_penalty

    return substitution_matrix

def codon_score(codon_a, codon_b, substitution_matrix):
    """
    Translate both codons to amino acids (or 'X')
    and look up in the substitution matrix.
    If either is '*' (stop codon), use special penalty (already in matrix or -5).
    Otherwise, default to -1 if not found.
    """
    aa_a = CODON_TABLE.get(codon_a.upper(), 'X')
    aa_b = CODON_TABLE.get(codon_b.upper(), 'X')
    return substitution_matrix.get((aa_a, aa_b), -1)


if __name__ == '__main__':

    BLOSUM62 = add_stop_penalties(BLOSUM62_NO_STOP, stop_penalty=-5)

    # Test the DNA alignment
    seq_a = "GATTACA"
    seq_b = "GCATGCU"
    dna_result = dna_alignment(seq_a, seq_b)
    print("DNA Alignment (Nucleotide-Level):")
    print(dna_result)

    # Test the protein alignment with the same sequences
    prot_score = protein_alignment(seq_a, seq_b, BLOSUM62, gap_penalty=-2)

    print("Protein Alignment (Translated from DNA):")
    print(prot_score)

    # Test the codon alignment with the same sequences
    codon_score = codon_alignment(seq_a, seq_b, BLOSUM62)
    print("Frameshift-Aware Codon Alignment:")
    print("Codon Alignment Score:", codon_score)
