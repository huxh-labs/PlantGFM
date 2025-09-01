from Bio import SeqIO
import sys

def extract_gene_sequences(fasta_file, gff_file, output_file, feature_type="gene"):
    """
    Extract sequences of specific features (default: genes) from a genome FASTA file
    using annotations from a GFF3 file.

    Parameters:
    fasta_file   : Path to the genome FASTA file
    gff_file     : Path to the GFF annotation file
    output_file  : Path to the output FASTA file
    feature_type : Feature type to extract (default = "gene")
    """
    # Load genome into dictionary for quick access
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    extracted_count = 0
    with open(output_file, "w") as out:
        with open(gff_file) as gff:
            for line in gff:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                chrom, source, feature, start, end, score, strand, phase, attributes = parts

                if feature.lower() == feature_type.lower():
                    try:
                        start, end = int(start), int(end)
                        seq_record = genome[chrom]
                        seq = seq_record.seq[start-1:end]  
                        header = f">{chrom}_{strand}_{start}-{end}"
                        out.write(f"{header}\n{seq}\n")
                        extracted_count += 1
                    except (ValueError, KeyError) as e:
                        print(f"Warning: error processing line {line.strip()}: {e}")
                        continue
    
    print(f"Extraction complete: {extracted_count} {feature_type} sequences extracted")
    print(f"Results saved in {output_file}")

if __name__ == "__main__":

    extract_gene_sequences("./demo.fasta", "./demo.gff3","./demo_gene.fasta")