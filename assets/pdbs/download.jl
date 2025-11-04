using BioStructures

ids = ["1ake_A", "4ake_A", "1ysy_A", "2ahm_D", "1su4_A", "1iwo_A", "1q9x_B", "1q9y_A", "1ih7_A", "1ig9_A"]
for id in ids
    pdb_id, chain_id = split(id, '_')
    data = retrievepdb(pdb_id; dir = tempdir())[chain_id]
    write("$id.bin", Float32.(coordarray(data, calphaselector)))
end
