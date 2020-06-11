import warnings,sys,os
from pathlib import Path

warnings.filterwarnings("ignore")

if (len(sys.argv) <= 1):
    sys.stderr.write("No file specified on command line")
    sys.exit(2)

datafile = Path(sys.argv[1])

if (not datafile.is_file()):
    sys.stderr.write("Input file not found")
    sys.exit(2)

if (len(sys.argv) <= 2):
    sys.stderr.write("No output file specified on command line")
    sys.exit(3)

outfile = Path(sys.argv[2])
    
def getInputFile():
    return str(datafile.resolve()) 
    
def getOutputFile():
    return str(outfile.absolute()) 


#me = Path(__file__)
#dir = me.parent
#os.chdir(str(dir.resolve()))

