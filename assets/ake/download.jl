using BioStructures
using DelimitedFiles

data_1ake = retrievepdb("1ake"; dir = tempdir())["A"]
data_4ake = retrievepdb("4ake"; dir = tempdir())["A"]

writedlm("1ake.csv", coordarray(data_1ake, calphaselector), ',')
writedlm("4ake.csv", coordarray(data_4ake, calphaselector), ',')
