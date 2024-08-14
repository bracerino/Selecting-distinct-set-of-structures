from fingerprint_calculator import StructureFingerprintCalculator
import time
chem_info2 = {
    "atomic_mass": {
        "Ti": 47.867,
        "Al": 26.98,
        "Cr": 51.9961,
        "Fe": 55.845,
        "Ni": 58.6934,
        "Cu": 63.546
    }
}

start_time = time.time()
calculator = StructureFingerprintCalculator('Vacancies_2/', 'dict_VACA_2.pkl', chem_info2)
calculator.process_structures()

end_time = time.time()  # End# timing
duration = end_time - start_time  # Calculate duration
print(f"Script completed in {duration} seconds")
