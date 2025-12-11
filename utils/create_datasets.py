import os
import random
import zipfile
from Bio import SeqIO
from pathlib import Path
from io import StringIO


def create_simple_fasta_dataset(input_fasta, output_fasta,sequences):

    with open(input_fasta, 'r') as f:
        records = list(SeqIO.parse(f, "fasta"))

    random.seed(42)
    sampled_records = random.sample(records, sequences)


    SeqIO.write(sampled_records, output_fasta, "fasta")

    return output_fasta

def create_simple_fasta_dataset2(input_folder, output_zip, sequences_per_file=100):

    input_path = Path(input_folder)
    output_path = Path(output_zip)


    fasta_files = list(input_path.glob("*.fasta")) + list(input_path.glob("*.fa")) + list(input_path.glob("*.fna"))

    if not fasta_files:
        print(f"No se encontraron archivos FASTA en {input_folder}")
        return None

    print(f"Encontrados {len(fasta_files)} archivos FASTA en {input_folder}")

    random.seed(42)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        processed_count = 0

        for fasta_file in fasta_files:
            try:
                with open(fasta_file, 'r') as f:
                    records = list(SeqIO.parse(f, "fasta"))

                print(f"\nArchivo: {fasta_file.name}")
                print(f"  Total de secuencias: {len(records)}")

                if len(records) == 0:
                    print(f"  ⚠️  Archivo vacío, saltando...")
                    continue

                sample_size = min(sequences_per_file, len(records))

                if len(records) <= sequences_per_file:
                    print(f"  Usando todas las {len(records)} secuencias (menos que el límite)")
                    sampled_records = records
                else:
                    print(f"  Muestreando {sample_size} secuencias aleatorias")
                    sampled_records = random.sample(records, sample_size)

                base_name = fasta_file.stem
                processed_name = f"{base_name}.fasta"

                fasta_content = StringIO()
                SeqIO.write(sampled_records, fasta_content, "fasta")

                zipf.writestr(processed_name, fasta_content.getvalue())

                processed_count += 1
                print(f" Guardado como '{processed_name}' en el ZIP")

            except Exception as e:
                print(f"  Error procesando {fasta_file.name}: {str(e)}")

    print(f"\n{'='*60}")
    print(f"✅ Proceso completado exitosamente!")
    print(f"Archivos procesados: {processed_count}/{len(fasta_files)}")
    print(f"Archivo ZIP creado: {output_path}")
    print(f"Tamaño del ZIP: {output_path.stat().st_size / 1024:.2f} KB")

    return str(output_path)